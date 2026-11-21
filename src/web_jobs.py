"""Job management for the CineComic browser interface and pipeline."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
UPLOAD_ROOT = PROJECT_ROOT / "data" / "uploads"
JOB_ROOT = PROJECT_ROOT / "data" / "jobs"
OUTPUT_ROOT = PROJECT_ROOT / "output"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MIN_VIDEO_DURATION_SECONDS = 1.0
MAX_VIDEO_DURATION_SECONDS = 300.0
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
STAGES = [
    "ingestion",
    "asr",
    "diarization",
    "phase4",
    "phase5",
    "phase6",
    "phase7",
    "phase8",
    "phase9",
    "phase10",
    "phase11",
    "phase12",
]

STAGE_METHODS = {
    "ingestion": "FFmpeg/FFprobe validation, video normalization, audio extraction, and metadata creation.",
    "asr": "Faster-Whisper speech recognition with timestamped utterance segmentation.",
    "diarization": "Pyannote speaker diarization to assign anonymous speaker turns.",
    "phase4": "Shot segmentation and candidate-frame quality filtering (blur, darkness, and redundancy checks).",
    "phase5": "InsightFace face detection, tracking, and character identity grouping.",
    "phase6": "Text-image semantic alignment for dialogue and candidate frames.",
    "phase7": "Temporal and visual evidence matching between diarized speakers and face tracks.",
    "phase8": "Narrative-aware selection of a compact set of comic keyframes.",
    "phase9": "Emotion estimation/fusion for dialogue and panel presentation.",
    "phase10": "Content-aware comic page and panel layout generation.",
    "phase11": "Panel image styling/cartoonization and speech balloon composition.",
    "phase12": "Final page rendering, PDF export, and output validation."
}

_PROCESS_LOCK = threading.Lock()
_PROCESSES: dict[str, tuple[subprocess.Popen, Any]] = {}
_STARTING_JOBS = 0


@contextmanager
def _reserve_job_slot():
    global _STARTING_JOBS
    with _PROCESS_LOCK:
        if _STARTING_JOBS or any(
            process.poll() is None
            for process, _ in _PROCESSES.values()
        ):
            raise RuntimeError(
                "Another pipeline job is already running. Wait for it to finish."
            )
        _STARTING_JOBS += 1
    try:
        yield
    finally:
        with _PROCESS_LOCK:
            _STARTING_JOBS -= 1


def sanitize_upload_filename(filename: str) -> str:
    """Return a portable, basename-only video filename."""
    basename = Path(filename.replace("\\", "/")).name
    suffix = Path(basename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported video format. Use MP4, MOV, MKV, AVI, or WEBM."
        )

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(basename).stem).strip("._")
    if not stem:
        raise ValueError("The uploaded filename does not contain a valid name.")
    return f"{stem}{suffix}"


def probe_video(video_path: Path) -> dict[str, Any]:
    """Read basic metadata and reject files without a valid video stream."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe is required to validate uploaded videos. "
            "Install FFmpeg and add it to PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Video metadata probing timed out.") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or "ffprobe could not read this file."
        raise ValueError(f"Invalid or unsupported video: {detail}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("ffprobe returned invalid video metadata.") from exc

    streams = payload.get("streams", [])
    video_stream = next(
        (
            stream for stream in streams
            if stream.get("codec_type") == "video"
        ),
        None,
    )
    if video_stream is None:
        raise ValueError("The uploaded file does not contain a video stream.")

    duration = payload.get("format", {}).get("duration")
    try:
        duration_s = float(duration)
    except (TypeError, ValueError):
        duration_s = None

    if duration_s is not None and duration_s < MIN_VIDEO_DURATION_SECONDS:
        raise ValueError("Video must be at least one second long.")
    if duration_s is not None and duration_s > MAX_VIDEO_DURATION_SECONDS:
        raise ValueError(
            "Videos longer than five minutes exceed the current pipeline limit."
        )

    return {
        "duration_s": duration_s,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "frame_rate": video_stream.get("avg_frame_rate"),
        "video_codec": video_stream.get("codec_name"),
        "size_bytes": video_path.stat().st_size,
    }


def probe_uploaded_bytes(payload: bytes, filename: str) -> dict[str, Any]:
    """Probe an in-memory upload using a temporary file that is always removed."""
    safe_name = sanitize_upload_filename(filename)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Video exceeds the 512 MB upload limit.")
    with tempfile.NamedTemporaryFile(
        suffix=Path(safe_name).suffix,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(payload)

    try:
        return probe_video(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validated_settings(settings: dict[str, Any]) -> dict[str, Any]:
    model_size = str(settings.get("model_size", "small"))
    if model_size not in {"tiny", "base", "small", "medium", "large-v3"}:
        raise ValueError("Choose a supported Whisper model size.")

    device = str(settings.get("device", "auto")).lower()
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("Device must be CPU, CUDA, or automatic.")
    if device == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise ValueError("CUDA was selected, but PyTorch is unavailable.") from exc
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA was selected, but no CUDA-compatible GPU is available. "
                "Choose CPU or automatic device selection."
            )

    language = str(settings.get("language", "en")).strip()
    if language.lower() != "auto" and not re.fullmatch(
        r"[A-Za-z-]{2,20}",
        language,
    ):
        raise ValueError(
            "Enter a language code such as en, es, fr, or auto."
        )
    panels_per_page = int(settings.get("panels_per_page", 6))
    if panels_per_page not in {4, 5, 6}:
        raise ValueError("Choose 4, 5, or 6 panels per page.")
    frame_scan = str(settings.get("frame_scan", "fast")).lower()
    if frame_scan not in {"fast", "detailed"}:
        raise ValueError("Frame scan must be fast or detailed.")

    return {
        "model_size": model_size,
        "device": device,
        "language": language or "auto",
        "panels_per_page": panels_per_page,
        "frame_scan": frame_scan,
    }


def _write_job_configs(
    config_dir: Path,
    settings: dict[str, Any],
) -> Path:
    source_dir = PROJECT_ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.glob("phase*.yaml")):
        shutil.copy2(source, config_dir / source.name)

    phase2_path = config_dir / "phase2.yaml"
    phase2 = yaml.safe_load(phase2_path.read_text(encoding="utf-8")) or {}
    asr = phase2.setdefault("asr", {})
    asr["model_size"] = settings["model_size"]
    asr["language"] = (
        None if settings["language"].lower() == "auto"
        else settings["language"]
    )
    asr["device"] = settings["device"]
    asr["compute_type"] = (
        "float16" if settings["device"] == "cuda"
        else "int8" if settings["device"] == "cpu"
        else "auto"
    )
    phase2_path.write_text(
        yaml.safe_dump(phase2, sort_keys=False),
        encoding="utf-8",
    )

    phase3_path = config_dir / "phase3.yaml"
    if phase3_path.exists():
        phase3 = yaml.safe_load(
            phase3_path.read_text(encoding="utf-8")
        ) or {}
        phase3.setdefault("diar", {})["device"] = settings["device"]
        phase3_path.write_text(
            yaml.safe_dump(phase3, sort_keys=False),
            encoding="utf-8",
        )

    phase5_path = config_dir / "phase5.yaml"
    if phase5_path.exists():
        phase5 = yaml.safe_load(
            phase5_path.read_text(encoding="utf-8")
        ) or {}
        detection = phase5.setdefault("phase5", {}).setdefault(
            "detection", {}
        )
        try:
            import onnxruntime as ort
            cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
        except ImportError:
            cuda_available = False
        detection["use_gpu"] = (
            cuda_available and settings["device"] in {"auto", "cuda"}
        )
        tracking = phase5.setdefault("phase5", {}).setdefault("tracking", {})
        if settings["frame_scan"] == "fast":
            # Face inference is the pipeline's largest measured CPU cost.
            # Halve the samples while allowing the tracker to bridge the
            # resulting two-second intervals.
            detection["det_fps"] = 0.5
            tracking["max_gap_s"] = 2.25
        else:
            detection["det_fps"] = 1.0
            tracking["max_gap_s"] = 1.5
        phase5_path.write_text(
            yaml.safe_dump(phase5, sort_keys=False),
            encoding="utf-8",
        )

    phase10_path = config_dir / "phase10.yaml"
    if phase10_path.exists():
        phase10 = yaml.safe_load(
            phase10_path.read_text(encoding="utf-8")
        ) or {}
        phase10.setdefault("phase10", {})["panels_per_page"] = (
            settings["panels_per_page"]
        )
        phase10_path.write_text(
            yaml.safe_dump(phase10, sort_keys=False),
            encoding="utf-8",
        )

    # The browser result viewer reads output/{job_id}; don't inherit a
    # developer's custom CLI output folder from the project-wide config.
    phase12_path = config_dir / "phase12.yaml"
    if phase12_path.exists():
        phase12 = yaml.safe_load(
            phase12_path.read_text(encoding="utf-8")
        ) or {}
        phase12["output_dir"] = "output"
        phase12["panels_per_page"] = settings["panels_per_page"]
        phase12["adaptive_pages"] = False
        # A two-column, three-row grid of 16:10 panels fills the square page.
        phase12["page_height"] = (
            1200 if settings["panels_per_page"] == 6 else 1600
        )
        phase12_path.write_text(
            yaml.safe_dump(phase12, sort_keys=False),
            encoding="utf-8",
        )

    return phase2_path


def _write_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(state_path)


def create_job(
    payload: bytes,
    filename: str,
    settings: dict[str, Any],
    hf_token: str | None = None,
    allow_model_downloads: bool = False,
) -> dict[str, Any]:
    with _reserve_job_slot():
        return _create_job(
            payload,
            filename,
            settings,
            hf_token,
            allow_model_downloads,
        )


def _create_job(
    payload: bytes,
    filename: str,
    settings: dict[str, Any],
    hf_token: str | None = None,
    allow_model_downloads: bool = False,
) -> dict[str, Any]:
    if not payload:
        raise ValueError("The uploaded video is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Video exceeds the 512 MB upload limit.")
    token = (hf_token or os.environ.get("HF_TOKEN", "")).strip()
    if not token:
        raise ValueError(
            "Hugging Face access is required for speaker diarization. "
            "Enter an HF token and accept access to pyannote/speaker-diarization-3.1 "
            "and pyannote/segmentation-3.0 before starting."
        )

    safe_name = sanitize_upload_filename(filename)
    validated = _validated_settings(settings)
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOB_ROOT / job_id
    upload_dir = UPLOAD_ROOT / job_id
    upload_dir.mkdir(parents=True, exist_ok=False)
    video_path = upload_dir / safe_name
    video_path.write_bytes(payload)

    try:
        video_info = probe_video(video_path)
    except (RuntimeError, ValueError):
        video_path.unlink(missing_ok=True)
        upload_dir.rmdir()
        raise

    config_path = _write_job_configs(job_dir / "configs", validated)
    log_path = job_dir / "pipeline.log"
    state_path = job_dir / "job.json"
    state = {
        "job_id": job_id,
        "video_id": job_id,
        "filename": safe_name,
        "status": "starting",
        "created_at": datetime.now().astimezone().isoformat(),
        "video_info": video_info,
        "settings": {
            **validated,
            "allow_model_downloads": allow_model_downloads,
        },
        "completed_stages": [],
        "progress": 0.0,
        "log_path": str(log_path),
        "video_path": str(video_path),
        "error": None,
    }
    _write_state(state_path, state)

    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "main.py"),
        "--config",
        str(config_path),
        "--video",
        str(video_path),
        "--video-id",
        job_id,
        "--stages",
        "all",
        "--force",
    ]
    log_file = log_path.open("ab")
    try:
        process_environment = os.environ.copy()
        process_environment["HF_TOKEN"] = token
        offline_mode = "0" if allow_model_downloads else "1"
        process_environment["HF_HUB_OFFLINE"] = offline_mode
        process_environment["TRANSFORMERS_OFFLINE"] = offline_mode
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=process_environment,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        log_file.close()
        state.update(status="failed", error=str(exc))
        _write_state(state_path, state)
        raise RuntimeError(f"Could not start the pipeline: {exc}") from exc

    with _PROCESS_LOCK:
        _PROCESSES[job_id] = (process, log_file)

    state.update(status="running", process_id=process.pid)
    _write_state(state_path, state)
    return state


def _completed_stages(log_text: str) -> list[str]:
    completed = set()
    for line in log_text.splitlines():
        match = re.match(
            r"^\[(ingestion|asr|diarization|phase(?:[4-9]|1[0-2]))\]",
            line,
        )
        if not match or "gate=FAIL" in line:
            continue
        stage = match.group(1)
        if "gate=PASS" in line or stage in {
            "phase5", "phase6", "phase7", "phase8",
            "phase10", "phase11", "phase12",
        }:
            completed.add(stage)
    return [stage for stage in STAGES if stage in completed]


def _stage_logs(log_text: str, running_stages: list[str]) -> dict[str, str]:
    """Group recent console output by stage logger or explicit stage tag."""
    grouped: dict[str, list[str]] = {stage: [] for stage in STAGES}
    for line in log_text.splitlines():
        lowered = line.lower()
        match_stage = None
        for stage in STAGES:
            stage_pattern = re.escape(stage)
            if re.search(
                rf"(?<![a-z0-9]){stage_pattern}(?![a-z0-9])",
                lowered,
            ):
                match_stage = stage
                break
        if match_stage is None and len(running_stages) == 1:
            # Older logger messages sometimes omit their logger name. During
            # a serial stage, these still belong to the only active phase.
            match_stage = running_stages[0]
        if match_stage:
            grouped[match_stage].append(line)

    return {
        stage: "\n".join(lines[-60:])
        for stage, lines in grouped.items()
        if lines
    }


def _scan_job_log_stages(log_path: Path, state: dict[str, Any]) -> list[str]:
    """Incrementally collect completed stages from the complete job log.

    The UI log tail is deliberately short, so using it for progress loses early
    stages on longer runs. Persist a byte cursor and scan only newly appended
    log data after the initial full scan.
    """
    if not log_path.is_file():
        return list(state.get("completed_stages", []))

    first_scan = "_log_scan_offset" not in state
    offset = 0 if first_scan else int(state.get("_log_scan_offset", 0))
    pending = "" if first_scan else str(state.get("_log_scan_pending", ""))
    try:
        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            end = log_file.tell()
            if end < offset:
                offset, pending, first_scan = 0, "", True
            log_file.seek(offset)
            chunk = log_file.read()
    except OSError:
        return list(state.get("completed_stages", []))

    decoded = pending + chunk.decode("utf-8", errors="replace")
    complete_text, separator, remainder = decoded.rpartition("\n")
    if separator:
        state["_log_scan_pending"] = remainder
    else:
        complete_text = ""
        state["_log_scan_pending"] = decoded
    state["_log_scan_offset"] = offset + len(chunk)

    statuses = dict(state.get("stage_statuses", {}))
    reasons = dict(state.get("stage_reasons", {}))
    if first_scan:
        for stage in _completed_stages(complete_text):
            statuses[stage] = "completed"

    for line in complete_text.splitlines():
        event = re.search(
            r"\[pipeline\]\s+stage_(started|completed|failed|skipped)="
            r"(ingestion|asr|diarization|phase(?:[4-9]|1[0-2]))(?:\s+reason=(.*))?",
            line,
        )
        if event:
            action, stage, reason = event.groups()
            statuses[stage] = {
                "started": "running", "completed": "completed",
                "failed": "failed", "skipped": "skipped",
            }[action]
            if reason:
                reasons[stage] = reason.strip()
            elif action == "completed":
                reasons.pop(stage, None)
        dialogue = re.search(r"\[pipeline\]\s+dialogue_detected=(true|false)", line, re.I)
        if dialogue:
            state["dialogue_detected"] = dialogue.group(1).lower() == "true"

    completed = {stage for stage, value in statuses.items() if value == "completed"}
    state["stage_statuses"] = {stage: statuses[stage] for stage in STAGES if stage in statuses}
    state["stage_reasons"] = {stage: reasons[stage] for stage in STAGES if stage in reasons}
    state["completed_stages"] = [stage for stage in STAGES if stage in completed]
    state["_running_stages"] = [
        stage for stage in STAGES if statuses.get(stage) == "running"
    ]
    return state["completed_stages"]


def _phase5_progress(log_text: str) -> dict[str, Any] | None:
    """Extract current face-scan progress and a rolling CPU-time estimate."""
    current = total = None
    inference_seconds: list[float] = []
    for line in log_text.splitlines():
        frame = re.search(r"Processing frame (\d+)/(\d+)", line)
        if frame:
            current, total = (int(value) for value in frame.groups())
        inference = re.search(r"Inference finished in ([\d.]+) seconds", line)
        if inference:
            inference_seconds.append(float(inference.group(1)))

    if not total:
        return None

    average = (
        sum(inference_seconds[-20:]) / len(inference_seconds[-20:])
        if inference_seconds
        else None
    )
    remaining_seconds = (
        max(0, total - (current or 0)) * average
        if average is not None
        else None
    )
    if remaining_seconds is None:
        eta = None
    elif remaining_seconds < 60:
        eta = f"about {int(remaining_seconds)} sec"
    elif remaining_seconds < 3600:
        eta = f"about {int(remaining_seconds // 60)} min"
    else:
        eta = f"about {remaining_seconds / 3600:.1f} hr"

    return {
        "current": current or 0,
        "total": total,
        "fraction": min(1.0, max(0.0, (current or 0) / total)),
        "average_inference_seconds": average,
        "eta": eta,
    }


def _failure_summary(log_text: str) -> tuple[str | None, str | None]:
    """Extract the failing stage and actionable final error from the log."""
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    for line in reversed(lines):
        event = re.search(
            r"\[pipeline\]\s+stage_failed=(ingestion|asr|diarization|phase(?:[4-9]|1[0-2]))\s+reason=(.*)",
            line,
        )
        if event:
            stage, reason = event.groups()
            return stage, f"{stage}: {reason or 'stage failed'}"

    for line in reversed(lines):
        gate = re.search(
            r"\[(ingestion|asr|diarization|phase(?:[4-9]|1[0-2]))\].*gate=FAIL(?:\s+reason=(.*))?",
            line,
        )
        if gate:
            stage, reason = gate.groups()
            detail = f": {reason}" if reason else " quality gate failed"
            return stage, f"{stage}{detail}"

    for line in reversed(lines):
        failed = re.search(
            r"^(INGESTION|ASR|DIARIZATION) FAILED:\s*(.+)$",
            line,
        )
        if failed:
            stage, detail = failed.groups()
            return stage.lower(), detail

    traceback_lines = [line for line in lines if line.startswith((
        "RuntimeError:", "ValueError:", "FileNotFoundError:",
        "ImportError:", "OSError:", "KeyError:", "AssertionError:",
    ))]
    if traceback_lines:
        return None, traceback_lines[-1]
    return None, None


def _tail_log(log_path: Path, max_lines: int = 120) -> str:
    if not log_path.is_file():
        return ""
    try:
        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            end = log_file.tell()
            log_file.seek(max(0, end - 1024 * 1024), os.SEEK_SET)
            tail = log_file.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(tail.splitlines()[-max_lines:])


def _pid_is_running(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except OSError:
        return False
    return True


def get_job(job_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        raise ValueError("Invalid job ID.")
    state_path = JOB_ROOT / job_id / "job.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Job {job_id} was not found.")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    process_entry = None
    with _PROCESS_LOCK:
        process_entry = _PROCESSES.get(job_id)

    if process_entry:
        process, log_file = process_entry
        exit_code = process.poll()
        if exit_code is not None:
            log_file.close()
            with _PROCESS_LOCK:
                _PROCESSES.pop(job_id, None)
            state["status"] = "completed" if exit_code == 0 else "failed"
            state["exit_code"] = exit_code
            state["error"] = None if exit_code == 0 else (
                f"Pipeline exited with status {exit_code}."
            )
    elif state.get("status") in {"starting", "running"}:
        process_id = state.get("process_id")
        if process_id and _pid_is_running(int(process_id)):
            pass
        elif (OUTPUT_ROOT / job_id / "page.json").is_file():
            state["status"] = "completed"
        else:
            state["status"] = "interrupted"
            state["error"] = (
                "The application restarted before this job recorded completion."
            )

    log_text = _tail_log(Path(state["log_path"]))
    completed = _scan_job_log_stages(Path(state["log_path"]), state)
    state["completed_stages"] = completed
    stage_statuses = state.get("stage_statuses", {})
    stage_reasons = state.get("stage_reasons", {})
    if state.get("status") in {"failed", "interrupted"}:
        for stage in state.get("_running_stages", []):
            stage_statuses[stage] = "failed"
            stage_reasons.setdefault(stage, "pipeline stopped before this stage completed")
    elif state.get("status") == "cancelled":
        for stage in state.get("_running_stages", []):
            stage_statuses[stage] = "skipped"
            stage_reasons.setdefault(stage, "processing was stopped by the user")
    state["stage_statuses"] = stage_statuses
    state["stage_reasons"] = stage_reasons
    processed = sum(
        stage_statuses.get(stage) in {"completed", "failed", "skipped"}
        for stage in STAGES
    )
    state["progress"] = processed / len(STAGES)
    if state.get("status") == "completed" and any(
        value == "failed" for value in stage_statuses.values()
    ):
        state["status"] = "completed_with_warnings"
    failed_reasons = [
        f"{stage}: {state.get('stage_reasons', {}).get(stage, 'stage failed')}"
        for stage, value in stage_statuses.items() if value == "failed"
    ]
    state["warning"] = "; ".join(failed_reasons) if failed_reasons else None
    if state["status"] in {"running", "starting"}:
        if not state.get("_running_stages"):
            fallback_stage = next(
                (stage for stage in STAGES if stage not in completed),
                None,
            )
            state["_running_stages"] = [fallback_stage] if fallback_stage else []
        state["current_stage"] = next(
            iter(state["_running_stages"]),
            next((stage for stage in STAGES if stage not in completed), "complete"),
        )
    else:
        state["_running_stages"] = []
        state["current_stage"] = "complete" if state["status"] in {"completed", "completed_with_warnings"} else next(
            (stage for stage in STAGES if stage not in completed),
            "complete",
        )
    state["phase5_progress"] = _phase5_progress(log_text)
    if state["status"] == "failed":
        failed_stage, summary = _failure_summary(log_text)
        failed_from_events = next(
            (stage for stage in STAGES if stage_statuses.get(stage) == "failed"),
            None,
        )
        state["failed_stage"] = failed_from_events or failed_stage or next(
            (stage for stage in STAGES if stage not in completed), None,
        )
        if state["failed_stage"] == "phase4":
            metrics_path = PROJECT_ROOT / "data" / "shots" / f"{job_id}.metrics.json"
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                reason = metrics.get("reason")
                if reason and reason != "all gates passed":
                    summary = f"phase4: {reason}"
            except (OSError, json.JSONDecodeError):
                pass
        if summary:
            state["error"] = summary
    else:
        state["failed_stage"] = None
    state["log_tail"] = log_text
    state["stage_logs"] = _stage_logs(log_text, state["_running_stages"])
    _write_state(state_path, state)
    return state


def stop_job(job_id: str) -> dict[str, Any]:
    """Stop a running job and its child processes, then persist cancellation."""
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        raise ValueError("Invalid job ID.")
    state_path = JOB_ROOT / job_id / "job.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Job {job_id} was not found.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    with _PROCESS_LOCK:
        process_entry = _PROCESSES.get(job_id)

    process = process_entry[0] if process_entry else None
    if process is not None and process.poll() is None:
        if os.name == "nt":
            stop_result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            if stop_result.returncode != 0 and process.poll() is None:
                process.terminate()
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    if process_entry:
        _, log_file = process_entry
        if not log_file.closed:
            log_file.close()
        with _PROCESS_LOCK:
            _PROCESSES.pop(job_id, None)

    if state.get("status") in {"starting", "running"}:
        state["status"] = "cancelled"
        state["stopped_by_user"] = True
        state["error"] = "Processing was stopped by you."
        log_text = _tail_log(Path(state.get("log_path", "")))
        completed = _scan_job_log_stages(Path(state.get("log_path", "")), state)
        stage_statuses = state.get("stage_statuses", {})
        processed = sum(
            stage_statuses.get(stage) in {"completed", "failed", "skipped"}
            for stage in STAGES
        )
        state["progress"] = processed / len(STAGES)
        state["current_stage"] = next(
            (stage for stage in STAGES if stage not in completed), "complete"
        )
        state["log_tail"] = log_text
        _write_state(state_path, state)
    return state


def build_processing_report(job_id: str) -> str:
    """Build a self-contained Markdown report from persisted job artifacts."""
    state = get_job(job_id)
    started = state.get("created_at", "unknown")
    video = state.get("video_info", {})
    settings = state.get("settings", {})
    completed = set(state.get("completed_stages", []))
    current = state.get("current_stage")
    status = state.get("status", "unknown")
    lines = [
        f"# CineComic processing report — {job_id}", "",
        f"- **Status:** {status}", f"- **Started:** {started}",
        f"- **Input:** {state.get('filename', 'unknown')}",
        f"- **Duration:** {video.get('duration_s', 'unknown')} seconds",
        f"- **Resolution:** {video.get('width', '?')} × {video.get('height', '?')}",
        f"- **Codec:** {video.get('video_codec', 'unknown')}",
        f"- **Settings:** Whisper `{settings.get('model_size', 'unknown')}`, language `{settings.get('language', 'unknown')}`, device `{settings.get('device', 'unknown')}`, face scan `{settings.get('frame_scan', 'fast')}`, {settings.get('panels_per_page', '?')} panels/page",
        "", "## Pipeline architecture and stage record", "",
        "The application runs a dependency-ordered Python pipeline. Independent phases run concurrently in small groups; downstream phases wait for their required artifacts. The Flask web server launches the pipeline as a child process and records its state and live log.", "",
        "| Stage | Run status | Detail | Method |", "|---|---|---|---|",
    ]
    for stage in STAGES:
        stage_status = state.get("stage_statuses", {}).get(stage)
        stage_status = stage_status or (
            "completed" if stage in completed else
            "running" if stage in state.get("_running_stages", []) and status in {"running", "starting"} else
            "current / incomplete" if stage == current and status in {"running", "starting"} else
            "stopped before stage" if status == "cancelled" else
            "failed / incomplete" if status in {"failed", "interrupted"} else
            "not reached"
        )
        detail = state.get("stage_reasons", {}).get(stage, "")
        detail = detail.replace("|", "\\|")
        lines.append(f"| `{stage}` | {stage_status.title()} | {detail} | {STAGE_METHODS[stage]} |")
    out_dir = OUTPUT_ROOT / job_id
    artifacts = [
        ("Transcript", PROJECT_ROOT / "data" / "subtitles" / f"{job_id}.json"),
        ("Selected keyframes", PROJECT_ROOT / "data" / "keyframes" / f"{job_id}.json"),
        ("Dialogue/frame alignment", out_dir / "dialogue_frame_alignment.json"),
        ("Comic page data and validation metrics", out_dir / "page.json"),
        ("Comic PDF", out_dir / "comic.pdf"),
    ]
    lines.extend(["", "## Artifacts", "", "| Artifact | Availability | Location |", "|---|---|---|"])
    for label, path in artifacts:
        lines.append(f"| {label} | {'Present' if path.is_file() else 'Not produced'} | `{path.relative_to(PROJECT_ROOT)}` |")
    page_path = out_dir / "page.json"
    if page_path.is_file():
        try:
            page = json.loads(page_path.read_text(encoding="utf-8-sig"))
            metrics = page.get("metrics", {}) if isinstance(page, dict) else {}
            if metrics:
                lines.extend(["", "## Final output metrics", ""])
                for key, value in metrics.items():
                    lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        except (OSError, json.JSONDecodeError):
            lines.extend(["", "The page metrics file exists but could not be parsed."])
    if state.get("error"):
        lines.extend(["", "## Run message", "", str(state["error"])])
    lines.extend(["", "## Processing log", "", "```text", state.get("log_tail", "(no log recorded)"), "```", ""])
    return "\n".join(lines)
