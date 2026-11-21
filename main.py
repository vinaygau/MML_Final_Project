"""CineComic-MML entry point."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from src.common.utils import (
    get_logger,
    load_pipeline_config,
    set_seed,
)
from src.ingestion.metadata import (
    IngestionError,
)
from src.ingestion.video_ingestion import (
    ingest,
)


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


def _stage_config(base_config: str | Path, stage_name: str) -> Path:
    p = Path(base_config)
    stage_yaml = p.parent / f"{stage_name}.yaml"
    if stage_yaml.exists():
        return stage_yaml
    return p


def _normalize_stages(stages):
    """Expand the special 'all' stage while preserving pipeline order."""
    if not stages:
        return ["ingestion"]

    if "all" in stages:
        return list(STAGES)

    ordered = []
    for name in STAGES:
        if name in stages:
            ordered.append(name)
    return ordered


def _run_stage_group(stage_functions, max_workers):
    """Run independent selected stages concurrently, preserving named results."""
    if len(stage_functions) <= 1 or max_workers <= 1:
        return {
            name: function()
            for name, function in stage_functions.items()
        }

    with ThreadPoolExecutor(
        max_workers=min(max_workers, len(stage_functions)),
        thread_name_prefix="cinecomic-stage",
    ) as executor:
        futures = {
            name: executor.submit(function)
            for name, function in stage_functions.items()
        }
        return {
            name: future.result()
            for name, future in futures.items()
        }


def _run_named_stage(stage_name, function, validator=None):
    """Run one stage, record its outcome, and let independent work continue."""
    print(f"[pipeline] stage_started={stage_name}", flush=True)
    try:
        value = function()
        if validator is not None:
            passed, reason = validator(value)
        elif isinstance(value, tuple) and value and isinstance(value[0], bool):
            passed = value[0]
            reason = str(value[1]) if len(value) > 1 else ""
        elif isinstance(value, bool):
            passed, reason = value, ""
        else:
            passed, reason = True, ""
    except Exception as exc:
        logging.exception("Pipeline stage %s failed", stage_name)
        reason = str(exc).replace("\r", " ").replace("\n", " ").strip()
        print(
            f"[pipeline] stage_failed={stage_name} reason={reason or type(exc).__name__}",
            flush=True,
        )
        return False, None, reason or type(exc).__name__

    if passed:
        print(f"[pipeline] stage_completed={stage_name}", flush=True)
        return True, value, ""

    reason = reason or "quality gate failed"
    reason = reason.replace("\r", " ").replace("\n", " ").strip()
    print(f"[pipeline] stage_failed={stage_name} reason={reason}", flush=True)
    return False, value, reason


def _skip_stage(stage_name, reason):
    reason = str(reason).replace("\r", " ").replace("\n", " ").strip()
    print(f"[pipeline] stage_skipped={stage_name} reason={reason}", flush=True)


def _transcript_has_dialogue(data_dir, video_id):
    transcript_path = Path(data_dir) / "subtitles" / f"{video_id}.json"
    try:
        payload = json.loads(transcript_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return False
    segments = payload.get("segments", []) if isinstance(payload, dict) else []
    return any(
        isinstance(segment, dict)
        and str(segment.get("text", "")).strip()
        for segment in segments
    ) if isinstance(segments, list) else False


def main(argv=None) -> int:

    ap = argparse.ArgumentParser(
        description="CineComic-MML pipeline"
    )

    ap.add_argument(
        "--config",
        required=True,
        help="YAML config"
    )

    ap.add_argument(
        "--video",
        default=None,
        help="video file path or URL"
    )

    ap.add_argument(
        "--video-id",
        default=None,
        help="video ID"
    )

    ap.add_argument(
        "--stages",
        nargs="+",
        default=["ingestion"],
        choices=STAGES + ["all"],
        help="stages to run, in order; use 'all' for the full pipeline",
    )

    ap.add_argument(
        "--force",
        action="store_true",
        help="ignore cache"
    )

    ap.add_argument(
        "--max-parallel-stages",
        type=int,
        default=2,
        help="upper limit for concurrent stages (default: 2; use 1 for sequential execution)",
    )

    args = ap.parse_args(argv)
    if args.max_parallel_stages < 1:
        ap.error("--max-parallel-stages must be at least 1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        force=True,
    )

    stages = _normalize_stages(args.stages)

    cfg = load_pipeline_config(
        args.config
    )

    set_seed(
        cfg["seed"]
    )

    video_id = args.video_id
    status: dict[str, str] = {}
    reasons: dict[str, str] = {}

    def skipped(stage_name: str, reason: str) -> None:
        if stage_name in stages:
            _skip_stage(stage_name, reason)
            status[stage_name] = "skipped"
            reasons[stage_name] = reason

    def run(stage_name: str, function, validator=None):
        if stage_name not in stages:
            return True, None
        ok, value, reason = _run_named_stage(stage_name, function, validator)
        status[stage_name] = "completed" if ok else "failed"
        if reason:
            reasons[stage_name] = reason
        return ok, value

    if "ingestion" in stages:
        if not args.video:
            ap.error("--video is required for the ingestion stage")

        def do_ingestion():
            try:
                return ingest(cfg, args.video, video_id, force=args.force)
            except IngestionError as exc:
                raise RuntimeError(str(exc)) from exc

        ok, value = run(
            "ingestion",
            do_ingestion,
            lambda result: (bool(result[0]), "ingestion quality gate failed"),
        )
        if ok:
            _, video_id = value
        if not ok:
            for stage_name in stages:
                if stage_name != "ingestion":
                    skipped(stage_name, "video ingestion did not produce usable input")
            return 1
    elif not video_id:
        ap.error("--video-id is required when ingestion is not in --stages")

    if "asr" in stages:
        from src.audio.whisper_asr import AsrStage

        log = get_logger(video_id, cfg["log_dir"])
        asr_ok, _ = run(
            "asr",
            lambda: AsrStage(cfg, video_id, log).execute(force=args.force),
        )
        dialogue_available = asr_ok and _transcript_has_dialogue(cfg["data_dir"], video_id)
        print(f"[pipeline] dialogue_detected={str(dialogue_available).lower()}", flush=True)
        if not dialogue_available:
            reasons.setdefault("asr", "No dialogue detected; visual-only comic generation will continue")
    else:
        dialogue_available = _transcript_has_dialogue(cfg["data_dir"], video_id)
        print(f"[pipeline] dialogue_detected={str(dialogue_available).lower()}", flush=True)

    prep_stage_functions = {}
    if "diarization" in stages:
        if not dialogue_available:
            skipped("diarization", "no dialogue was detected; speaker diarization is not needed")
        else:
            from src.audio.diarization import DiarStage

            log = get_logger(video_id, cfg["log_dir"])
            prep_stage_functions["diarization"] = lambda: _run_named_stage(
                "diarization",
                lambda: DiarStage(cfg, video_id, log).execute(force=args.force),
            )

    if "phase4" in stages:
        from src.video.phase4 import Phase4Stage

        phase4_stage = Phase4Stage(cfg, video_id)

        def do_phase4():
            passed = phase4_stage.execute(force=args.force)
            try:
                metrics = json.loads(phase4_stage.metrics_path.read_text(encoding="utf-8"))
                reason = metrics.get("reason", "")
            except (OSError, json.JSONDecodeError):
                reason = ""
            print(
                f"[phase4] video_id={video_id} gate={'PASS' if passed else 'FAIL'}"
                + (f" reason={reason}" if reason else ""),
                flush=True,
            )
            return passed, reason or ("scene detection quality gate failed" if not passed else "")

        prep_stage_functions["phase4"] = lambda: _run_named_stage("phase4", do_phase4)

    # These stages read separate artifacts after ASR: audio/transcript for
    # diarization and video/metadata for scene detection. Run them together.
    prep_results = _run_stage_group(prep_stage_functions, args.max_parallel_stages)
    prep_ok = {}
    for name, result in prep_results.items():
        ok, _value, reason = result
        status[name] = "completed" if ok else "failed"
        prep_ok[name] = ok
        if reason:
            reasons[name] = reason
    diarization_ok = prep_ok.get("diarization", True)
    phase4_ok = prep_ok.get("phase4", True)

    cfg["force"] = args.force
    visual_stage_functions = {}
    if "phase5" in stages:
        from src.vision.phase5 import run_phase5
        if phase4_ok:
            visual_stage_functions["phase5"] = lambda: _run_named_stage(
                "phase5", lambda: run_phase5(
                    video_id=video_id, config=cfg,
                    video_path=Path(args.video) if args.video else None,
                ),
            )
        else:
            skipped("phase5", "scene detection failed, so candidate frames are unavailable")
    if "phase6" in stages:
        from src.vision.phase6 import run_phase6
        if phase4_ok:
            visual_stage_functions["phase6"] = lambda: _run_named_stage(
                "phase6", lambda: run_phase6(
                    project_root=Path(__file__).resolve().parent,
                    video_id=video_id,
                    config_path=_stage_config(args.config, "phase6"),
                ),
            )
        else:
            skipped("phase6", "scene detection failed, so candidate frames are unavailable")

    visual_results = _run_stage_group(visual_stage_functions, args.max_parallel_stages)
    visual_ok = {name: result[0] for name, result in visual_results.items()}
    for name, result in visual_results.items():
        ok, value, reason = result
        status[name] = "completed" if ok else "failed"
        if reason:
            reasons[name] = reason
        if ok and isinstance(value, dict):
            detail = f"tracks={value.get('tracks', 0)}" if name == "phase5" else f"segments={value.get('num_segments', 0)}"
            print(f"[{name}] video_id={video_id} {detail}")

    phase7_ok = True
    if "phase7" in stages:
        if not dialogue_available:
            phase7_ok = False
            skipped("phase7", "no dialogue is available for active speaker matching")
        elif not diarization_ok or ("phase5" in stages and not visual_ok.get("phase5", False)):
            phase7_ok = False
            skipped("phase7", "diarization or character tracking failed")
        else:
            from src.speaker.phase7 import run_phase7
            phase7_ok, _ = run("phase7", lambda: run_phase7(
                config_path=_stage_config(args.config, "phase7"), video_id=video_id,
            ))

    multimodal_functions = {}
    if "phase8" in stages:
        if phase4_ok:
            from src.keyframes.phase8 import run_phase8
            multimodal_functions["phase8"] = lambda: _run_named_stage(
                "phase8", lambda: run_phase8(str(_stage_config(args.config, "phase8")), video_id),
                lambda result: (bool(result.get("gate_passed", True)), result.get("reason", "")),
            )
        else:
            skipped("phase8", "scene detection failed, so keyframe selection cannot run")
    if "phase9" in stages:
        if not dialogue_available:
            skipped("phase9", "no dialogue is available for emotion analysis")
        else:
            from src.emotion.phase9 import Phase9Stage
            multimodal_functions["phase9"] = lambda: _run_named_stage(
                "phase9", lambda: Phase9Stage(cfg, video_id).execute(force=args.force),
            )

    multimodal_results = _run_stage_group(multimodal_functions, args.max_parallel_stages)
    multimodal_ok = {}
    for name, result in multimodal_results.items():
        ok, value, reason = result
        status[name] = "completed" if ok else "failed"
        multimodal_ok[name] = ok
        if reason:
            reasons[name] = reason
        if ok and name == "phase8":
            count = value.get("selected_panels") or value.get("num_keyframes", 0)
            print(f"[phase8] video_id={video_id} keyframes={count}")

    phase8_ok = multimodal_ok.get("phase8", True)
    phase10_ok = True
    if "phase10" in stages:
        if not phase8_ok:
            phase10_ok = False
            skipped("phase10", "keyframe selection failed")
        else:
            from src.layout.phase10 import run_phase10
            phase10_ok, result = run("phase10", lambda: run_phase10(video_id, _stage_config(args.config, "phase10")))
            if phase10_ok:
                count = result.get("metrics", {}).get("panel_count", result.get("num_panels", 0))
                print(f"[phase10] video_id={video_id} panels={count}")

    phase11_ok = True
    if "phase11" in stages:
        if not phase10_ok:
            phase11_ok = False
            skipped("phase11", "comic layout failed")
        else:
            from src.comic.phase11 import run_phase11
            phase11_ok, result = run("phase11", lambda: run_phase11(video_id, _stage_config(args.config, "phase11")))
            if phase11_ok:
                count = result.get("metrics", {}).get("successful_panels", result.get("num_panels", 0))
                print(f"[phase11] video_id={video_id} panels={count}")

    phase12_ok = True
    if "phase12" in stages:
        if not phase10_ok:
            phase12_ok = False
            skipped("phase12", "comic layout failed")
        else:
            from src.comic.phase12 import run_phase12
            phase12_ok, result = run("phase12", lambda: run_phase12(video_id, _stage_config(args.config, "phase12")))
            if phase12_ok:
                count = result.get("metrics", {}).get("page_count", result.get("num_pages", 0))
                print(f"[phase12] video_id={video_id} pages={count}")

    failed_stages = [name for name, outcome in status.items() if outcome == "failed"]
    skipped_stages = [name for name, outcome in status.items() if outcome == "skipped"]
    print(
        f"[pipeline] summary completed={sum(v == 'completed' for v in status.values())} "
        f"failed={len(failed_stages)} skipped={len(skipped_stages)}",
        flush=True,
    )
    if "phase12" in stages:
        return 0 if phase12_ok else 1
    return 1 if failed_stages else 0


if __name__ == "__main__":
    sys.exit(main())
