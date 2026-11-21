"""Phase 3 diarization checks."""

from __future__ import annotations

import json
import sys
import types
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.audio.diarization import (
    DiarStage,
    _diarize_pyannote,
    _read_pyannote_audio,
)
from src.common.utils import load_config

from tests.conftest import ROOT


@pytest.fixture
def diar_cfg(tmp_path):
    cfg = load_config(ROOT / "configs" / "phase3.yaml")
    cfg["data_dir"] = str(tmp_path / "data")
    cfg["results_dir"] = str(tmp_path / "results")
    cfg["log_dir"] = str(tmp_path / "logs")
    return cfg


def test_phase3_config_loads():
    cfg = load_config(ROOT / "configs" / "phase3.yaml")
    assert cfg["diar"]["backend"] == "pyannote"
    assert cfg["diar_gates"]["der_max"] == 0.25


def test_main_registers_diarization_stage():
    import main

    assert "diarization" in main.STAGES


def test_pyannote_audio_reader_loads_pcm_waveform_without_codec(
    tmp_path: Path,
):
    audio_path = tmp_path / "mono.wav"
    source_samples = np.asarray([0, 16384, -16384, 32767], dtype="<i2")
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(source_samples.tobytes())

    audio = _read_pyannote_audio(str(audio_path))

    assert audio["sample_rate"] == 16000
    assert tuple(audio["waveform"].shape) == (1, 4)
    assert str(audio["waveform"].dtype) == "torch.float32"
    assert audio["waveform"][0, 1].item() == pytest.approx(0.5)
    assert audio["waveform"][0, 2].item() == pytest.approx(-0.5)


def test_pyannote_backend_passes_decoded_waveform_to_pipeline(
    tmp_path: Path,
    monkeypatch,
):
    audio_path = tmp_path / "mono.wav"
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(np.asarray([0, 1024], dtype="<i2").tobytes())

    captured = {}

    class FakeAnnotation:
        @staticmethod
        def itertracks(yield_label):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=0.5), None, "speaker_1"

    class FakePipeline:
        @staticmethod
        def from_pretrained(_model, token):
            assert token == "hf_test_token"
            return FakePipeline()

        def __call__(self, audio):
            captured["audio"] = audio
            return FakeAnnotation()

    pyannote_module = types.ModuleType("pyannote")
    pyannote_module.__path__ = []
    audio_module = types.ModuleType("pyannote.audio")
    audio_module.Pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio_module)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    turns = _diarize_pyannote(
        str(audio_path),
        {"model": "test/model", "device": "auto"},
    )

    assert turns == [{"start": 0.0, "end": 0.5, "speaker": "speaker_1"}]
    assert captured["audio"]["sample_rate"] == 16000
    assert tuple(captured["audio"]["waveform"].shape) == (1, 2)


def test_diar_stage_runs_and_uses_mock_backend(diar_cfg, tmp_path):
    data_dir = Path(diar_cfg["data_dir"])
    (data_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (data_dir / "subtitles").mkdir(parents=True, exist_ok=True)
    (data_dir / "diarization").mkdir(parents=True, exist_ok=True)

    (data_dir / "metadata" / "demo.json").write_text(
        json.dumps({
            "audio_path": str(tmp_path / "demo.wav"),
            "audio_duration_s": 2.0,
            "has_audio": True,
        }),
        encoding="utf-8",
    )

    (data_dir / "subtitles" / "demo.json").write_text(
        json.dumps({
            "segments": [
                {"segment_id": "s0", "start": 0.0, "end": 1.0},
                {"segment_id": "s1", "start": 1.0, "end": 2.0},
            ]
        }),
        encoding="utf-8",
    )

    def fake_backend(wav, dcfg, log=None):
        return [
            {"start": 0.0, "end": 1.0, "speaker": "A"},
            {"start": 1.0, "end": 2.0, "speaker": "B"},
        ]

    stage = DiarStage(diar_cfg, "demo", diarize_fn=fake_backend)
    stage.run()

    metrics = stage.evaluate()
    ok, reason = stage.gate(metrics)

    assert ok is True
    assert reason.startswith("structure OK")
    assert (data_dir / "diarization" / "demo.raw.json").exists()
    assert (data_dir / "diarization" / "demo.rttm").exists()
    assert metrics["n_turns"] == 2
