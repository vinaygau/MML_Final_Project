import json
import subprocess
from pathlib import Path

import pytest
import yaml

import src.web_jobs as web_jobs


def test_sanitize_upload_filename_removes_path_components():
    assert web_jobs.sanitize_upload_filename(
        r"..\folder\my clip.MP4"
    ) == "my_clip.mp4"


def test_probe_video_returns_stream_metadata(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"video")

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({
                "format": {"duration": "12.5"},
                "streams": [{
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 640,
                    "height": 360,
                    "avg_frame_rate": "24/1",
                }],
            }),
            stderr="",
        )

    monkeypatch.setattr(web_jobs.subprocess, "run", fake_run)

    metadata = web_jobs.probe_video(video_path)

    assert metadata["duration_s"] == 12.5
    assert metadata["width"] == 640
    assert metadata["video_codec"] == "h264"


def test_probe_uploaded_bytes_removes_temporary_file(
    monkeypatch,
):
    probed_paths = []

    def fake_probe(path):
        probed_paths.append(path)
        return {"duration_s": 1.0}

    monkeypatch.setattr(web_jobs, "probe_video", fake_probe)

    assert web_jobs.probe_uploaded_bytes(b"video", "clip.mp4") == {
        "duration_s": 1.0
    }
    assert len(probed_paths) == 1
    assert not probed_paths[0].exists()


def test_probe_video_rejects_files_without_video_stream(
    tmp_path: Path,
    monkeypatch,
):
    video_path = tmp_path / "audio-only.mp4"
    video_path.write_bytes(b"media")
    monkeypatch.setattr(
        web_jobs.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"format": {}, "streams": [{"codec_type": "audio"}]}',
            stderr="",
        ),
    )

    with pytest.raises(ValueError, match="does not contain a video stream"):
        web_jobs.probe_video(video_path)


def test_job_config_overrides_are_isolated_per_job(
    tmp_path: Path,
    monkeypatch,
):
    project_root = tmp_path / "project"
    source_configs = project_root / "configs"
    source_configs.mkdir(parents=True)
    (source_configs / "phase2.yaml").write_text(
        "asr:\n  model_size: small\n  device: cpu\n  language: en\n",
        encoding="utf-8",
    )
    (source_configs / "phase3.yaml").write_text(
        "diar:\n  device: auto\n",
        encoding="utf-8",
    )
    (source_configs / "phase10.yaml").write_text(
        "phase10:\n  panels_per_page: 5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_jobs, "PROJECT_ROOT", project_root)

    phase2_path = web_jobs._write_job_configs(
        tmp_path / "job" / "configs",
        {
            "model_size": "base",
            "language": "auto",
            "device": "cuda",
            "panels_per_page": 6,
        },
    )

    phase2 = yaml.safe_load(phase2_path.read_text(encoding="utf-8"))
    phase3 = yaml.safe_load(
        (phase2_path.parent / "phase3.yaml").read_text(encoding="utf-8")
    )
    phase10 = yaml.safe_load(
        (phase2_path.parent / "phase10.yaml").read_text(encoding="utf-8")
    )
    source_phase2 = yaml.safe_load(
        (source_configs / "phase2.yaml").read_text(encoding="utf-8")
    )

    assert phase2["asr"] == {
        "model_size": "base",
        "device": "cuda",
        "language": None,
        "compute_type": "float16",
    }
    assert phase3["diar"]["device"] == "cuda"
    assert phase10["phase10"]["panels_per_page"] == 6
    assert source_phase2["asr"]["model_size"] == "small"


def test_job_creation_requires_hugging_face_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(ValueError, match="Hugging Face access is required"):
        web_jobs.create_job(b"video", "clip.mp4", {})


def test_job_creation_passes_token_only_to_pipeline_environment(
    tmp_path: Path,
    monkeypatch,
):
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "phase2.yaml").write_text("asr: {}\n", encoding="utf-8")
    (config_dir / "phase3.yaml").write_text("diar: {}\n", encoding="utf-8")
    (config_dir / "phase10.yaml").write_text("phase10: {}\n", encoding="utf-8")
    monkeypatch.setattr(web_jobs, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(web_jobs, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.setattr(web_jobs, "JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(web_jobs, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(web_jobs, "_PROCESSES", {})
    monkeypatch.setattr(
        web_jobs,
        "probe_video",
        lambda _path: {"duration_s": 1.0},
    )

    captured = {}

    class FakeProcess:
        pid = 12345

        @staticmethod
        def poll():
            return None

    def fake_popen(_command, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(web_jobs.subprocess, "Popen", fake_popen)

    state = web_jobs.create_job(
        b"video",
        "clip.mp4",
        {},
        hf_token="hf_test_secret",
        allow_model_downloads=True,
    )
    persisted_state = json.loads(
        (tmp_path / "jobs" / state["job_id"] / "job.json").read_text(
            encoding="utf-8"
        )
    )
    web_jobs._PROCESSES[state["job_id"]][1].close()
    web_jobs._PROCESSES.pop(state["job_id"])

    assert captured["env"]["HF_TOKEN"] == "hf_test_secret"
    assert captured["env"]["HF_HUB_OFFLINE"] == "0"
    assert captured["env"]["TRANSFORMERS_OFFLINE"] == "0"
    assert state["settings"]["allow_model_downloads"] is True
    assert "hf_test_secret" not in json.dumps(persisted_state)
    assert "hf_token" not in persisted_state


def test_completed_stage_parser_uses_real_pipeline_markers():
    log = (
        "[ingestion] video_id=sample gate=PASS\n"
        "[asr] video_id=sample gate=PASS\n"
        "[phase8] video_id=sample keyframes=12\n"
    )

    assert web_jobs._completed_stages(log) == [
        "ingestion",
        "asr",
        "phase8",
    ]
