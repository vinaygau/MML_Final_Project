import json
from pathlib import Path

import src.keyframes.phase8 as phase8
import yaml
import numpy as np
from PIL import Image
from src.keyframes.phase8 import (
    extract_dialogue_timestamp_frames,
    extract_frame_records,
    prioritize_dialogue_coverage,
    run_phase8,
    temporal_scene_fallbacks,
)


def test_extract_frame_records_uses_nested_shot_timestamps(tmp_path: Path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    (frame_dir / "s02_f001.jpg").write_bytes(b"")

    records = extract_frame_records(
        frame_dir,
        {
            "shots": [
                {
                    "shot_id": 2,
                    "start": 8.0,
                    "end": 12.0,
                    "frames": [
                        {
                            "frame_id": "s02_f001",
                            "t": 9.5,
                            "path": "frames/s02_f001.jpg",
                        }
                    ],
                }
            ]
        },
    )

    assert len(records) == 1
    assert records[0]["t"] == 9.5
    assert records[0]["shot_id"] == 2
    assert records[0]["shot_start"] == 8.0
    assert records[0]["shot_end"] == 12.0


def test_temporal_fallback_stays_within_the_matching_scene():
    records = [
        {
            "frame_id": "inside",
            "t": 20.0,
            "shot_start": 19.0,
            "shot_end": 22.0,
        },
        {
            "frame_id": "other_scene",
            "t": 20.2,
            "shot_start": 22.0,
            "shot_end": 25.0,
        },
    ]

    result = temporal_scene_fallbacks(
        records,
        [{"segment_id": 1, "start": 20.4, "end": 20.8}],
        {},
    )

    assert result == {"inside": ["1"]}


def test_temporal_fallback_does_not_use_a_distant_frame_from_same_shot():
    records = [{
        "frame_id": "far_inside_same_shot",
        "t": 1.0,
        "shot_start": 0.0,
        "shot_end": 30.0,
    }]

    result = temporal_scene_fallbacks(
        records,
        [{"segment_id": 1, "start": 20.0, "end": 21.0}],
        {},
        max_gap_s=2.0,
    )

    assert result == {}


def test_missing_frame_uses_the_source_video_at_dialogue_midpoint(
    tmp_path: Path,
    monkeypatch,
):
    class Capture:
        def __init__(self):
            self.position_ms = None

        def isOpened(self):
            return True

        def set(self, _property, value):
            self.position_ms = value
            return True

        def read(self):
            return True, object()

        def release(self):
            pass

    capture = Capture()
    monkeypatch.setattr(
        phase8.cv2,
        "VideoCapture",
        lambda _path: capture,
    )

    def write_image(path, _image):
        Path(path).write_bytes(b"synthetic frame")
        return True

    monkeypatch.setattr(phase8.cv2, "imwrite", write_image)
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"synthetic video")

    records, fallbacks = extract_dialogue_timestamp_frames(
        video_path,
        [{"segment_id": "4", "start": 2.0, "end": 4.0, "text": "Hello."}],
        set(),
        {"shots": [{"shot_id": 8, "start": 1.0, "end": 5.0}]},
        tmp_path / "frames",
    )

    assert capture.position_ms == 3000.0
    assert records[0]["timestamp_source"] == "transcript_midpoint_fallback"
    assert records[0]["shot_id"] == 8
    assert fallbacks == {records[0]["frame_id"]: ["4"]}


def test_panel_selection_preserves_all_dialogue_coverage():
    panels = [
        {
            "frame_id": "dialogue_1",
            "segment_ids": ["1"],
            "scores": {"total": 0.6},
        },
        {
            "frame_id": "dialogue_2",
            "segment_ids": ["2"],
            "scores": {"total": 0.7},
        },
        {
            "frame_id": "dialogue_3",
            "segment_ids": ["3"],
            "scores": {"total": 0.5},
        },
        {"frame_id": "visual", "segment_ids": [], "scores": {"total": 0.9}},
    ]

    selected = prioritize_dialogue_coverage(panels, page_budget=2)

    assert {sid for panel in selected for sid in panel["segment_ids"]} == {
        "1",
        "2",
        "3",
    }
    assert len(selected) == 3


def test_phase8_run_keeps_dialogue_and_extracts_missing_scene_frame(
    tmp_path: Path,
    monkeypatch,
):
    class Capture:
        def isOpened(self):
            return True

        def set(self, _property, _value):
            return True

        def read(self):
            return True, np.full((80, 120, 3), 120, dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(phase8.cv2, "VideoCapture", lambda _path: Capture())
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    Image.new("RGB", (120, 80), "#a08060").save(frame_dir / "frame_1.jpg")
    Image.new("RGB", (120, 80), "#607080").save(frame_dir / "frame_2.jpg")

    alignment_path = tmp_path / "alignment.json"
    alignment_path.write_text(
        json.dumps({
            "alignment": [
                {
                    "segment_id": "1",
                    "start": 1.0,
                    "end": 2.0,
                    "text": "First line.",
                    "candidates": [{
                        "frame_id": "frame_1",
                        "t": 1.5,
                        "shot_id": 1,
                        "clip_sim": 0.4,
                        "rank": 1,
                    }],
                },
                {
                    "segment_id": "2",
                    "start": 7.0,
                    "end": 8.0,
                    "text": "Second line.",
                    "candidates": [],
                    "status": "no_candidates",
                },
            ]
        }),
        encoding="utf-8",
    )
    shots_path = tmp_path / "shots.json"
    shots_path.write_text(
        json.dumps({
            "shots": [
                {
                    "shot_id": 1,
                    "start": 0.0,
                    "end": 5.0,
                    "frames": [
                        {
                            "frame_id": "frame_1",
                            "t": 1.5,
                            "path": str(frame_dir / "frame_1.jpg"),
                        },
                        {
                            "frame_id": "frame_2",
                            "t": 3.0,
                            "path": str(frame_dir / "frame_2.jpg"),
                        },
                    ],
                },
                {
                    "shot_id": 2,
                    "start": 6.0,
                    "end": 9.0,
                    "frames": [],
                },
            ]
        }),
        encoding="utf-8",
    )
    metadata_path = tmp_path / "metadata.json"
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"synthetic video")
    metadata_path.write_text(
        json.dumps({"raw_path": str(video_path)}),
        encoding="utf-8",
    )
    speaker_path = tmp_path / "speaker.json"
    speaker_path.write_text("{}", encoding="utf-8")
    output_path = tmp_path / "keyframes.json"
    metrics_path = tmp_path / "metrics.json"
    config_path = tmp_path / "phase8.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "phase8": {
                "frames_dir_template": str(frame_dir),
                "alignment_template": str(alignment_path),
                "speaker_template": str(speaker_path),
                "shots_template": str(shots_path),
                "metadata_template": str(metadata_path),
                "image_embeddings_template": str(tmp_path / "embeddings.npy"),
                "output_template": str(output_path),
                "metrics_template": str(metrics_path),
                "page_budget": 1,
            }
        }),
        encoding="utf-8",
    )

    result = run_phase8(str(config_path), "synthetic")
    keyframes = json.loads(output_path.read_text(encoding="utf-8"))
    selected_segments = {
        segment_id
        for panel in keyframes["panels"]
        for segment_id in panel["segment_ids"]
    }

    assert selected_segments == {"1", "2"}
    assert result["dialogue_segment_coverage"] == 1.0
    assert result["dialogue_timestamp_fallback_segments"] == ["2"]
    assert result["page_budget_exceeded_for_dialogue_coverage"]
