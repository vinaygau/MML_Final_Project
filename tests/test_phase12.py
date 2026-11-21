import json
from pathlib import Path

import yaml
from PIL import Image

from src.comic.phase12 import (
    draw_bubble,
    layout_bubble_chunks,
    panel_segment_ids,
    run_phase12,
    speaker_display_labels,
    speaker_face_for_panel,
    write_html,
)


def test_panel_dialogue_requires_an_explicit_segment_alignment():
    transcript = {
        "1": {
            "segment_id": "1",
            "start": 10.0,
            "end": 11.0,
            "text": "A spoken sentence.",
        }
    }

    assert panel_segment_ids(
        {"timestamp": 10.5},
        transcript,
        {},
    ) == []
    assert panel_segment_ids(
        {"segment_ids": ["1"], "timestamp": 10.5},
        transcript,
        {},
    ) == ["1"]


def test_unknown_character_id_cannot_match_an_unknown_face_roi():
    assert speaker_face_for_panel(
        {
            "rois_panel_coordinates": [{
                "kind": "face",
                "character_id": "unknown",
                "track_id": "unknown",
                "box": [10, 10, 50, 60],
            }]
        },
        "unknown",
        "unknown",
    ) is None


def test_diarization_ids_get_stable_reader_labels():
    labels = speaker_display_labels([
        {"_start": 3.0, "_speaker_id": "SPEAKER_B"},
        {"_start": 4.0, "_speaker_id": "unknown"},
        {"_start": 5.0, "_speaker_id": "SPEAKER_A"},
    ])

    assert labels == {"SPEAKER_B": "Speaker 1", "SPEAKER_A": "Speaker 2"}


def test_bubble_label_is_drawn_with_the_dialogue():
    image = Image.new("RGB", (360, 220), "white")
    metadata = draw_bubble(
        image,
        "Hello there.",
        "neutral",
        "Speaker 1",
        (12, 12, 348, 208),
        {
            "bubble_padding": 10,
            "bubble_max_width_ratio": 0.75,
            "bubble_max_height_ratio": 0.7,
            "bubble_font_size": 18,
            "bubble_min_font_size": 10,
            "bubble_fill": "#fffdf7",
            "bubble_outline": "#171717",
            "bubble_text": "#111111",
            "bubble_outline_width": 2,
            "bubble_margin": 12,
        },
    )

    assert metadata["speaker_label"] == "Speaker 1"
    assert metadata["attributed"] is True


def test_dialogue_density_splits_by_bubble_area_without_dropping_turns():
    image = Image.new("RGB", (500, 300), "white")
    specs = [
        {"text": f"Line {index}.", "segment_ids": [str(index)], "_display_speaker_label": f"Speaker {index % 2 + 1}"}
        for index in range(8)
    ]
    config = {
        "bubble_margin": 12,
        "bubble_padding": 8,
        "bubble_row_gap": 8,
        "bubble_max_width_ratio": 0.75,
        "bubble_font_size": 17,
        "bubble_min_font_size": 10,
        "bubble_layout_max_coverage_ratio": 0.08,
    }

    chunks = layout_bubble_chunks(
        image,
        specs,
        caption_height=0,
        safe_zone=(12, 12, 488, 288),
        face_boxes=[],
        config=config,
    )
    rendered_specs = [spec for chunk, _ in chunks for spec in chunk]

    assert len(chunks) > 1
    assert rendered_specs == specs


def test_html_export_uses_relative_panel_image_paths(tmp_path: Path):
    output_dir = tmp_path / "comic"
    output_dir.mkdir()
    image_path = output_dir / "panels" / "panel_1.png"

    write_html(
        [{
            "page": 1,
            "panels": [{
                "panel_id": 1,
                "image": str(image_path),
                "bubbles": [],
            }],
        }],
        output_dir / "page.html",
    )

    content = (output_dir / "page.html").read_text(encoding="utf-8")
    assert "src='panels/panel_1.png'" in content
    assert str(image_path) not in content


def test_phase12_writes_alignment_report_and_unplaced_segment_warning(
    tmp_path: Path,
):
    video_id = "synthetic"
    transcript_dir = tmp_path / "subtitles"
    keyframes_dir = tmp_path / "keyframes"
    layout_dir = tmp_path / "layout"
    speaker_dir = tmp_path / "speaker"
    emotion_dir = tmp_path / "emotions"
    output_dir = tmp_path / "output"
    for directory in (
        transcript_dir,
        keyframes_dir,
        layout_dir,
        speaker_dir,
        emotion_dir,
    ):
        directory.mkdir()

    image_path = tmp_path / "frame.png"
    Image.new("RGB", (320, 240), "#8ca4b8").save(image_path)
    (transcript_dir / f"{video_id}.json").write_text(
        json.dumps({
            "segments": [
                {"segment_id": "1", "start": 1.0, "end": 2.0, "text": "Hello."},
                {"segment_id": "2", "start": 5.0, "end": 6.0, "text": "Goodbye."},
            ]
        }),
        encoding="utf-8",
    )
    (keyframes_dir / f"{video_id}.json").write_text(
        json.dumps({
            "panels": [{
                "frame_id": "frame_1",
                "segment_ids": ["1"],
                "alignment_method": "phase6_semantic",
                "alignment_confidence": 0.8,
            }]
        }),
        encoding="utf-8",
    )
    (layout_dir / f"{video_id}.json").write_text(
        json.dumps({
            "pages": [{
                "page": 1,
                "panels": [{
                    "panel_id": 1,
                    "frame_id": "frame_1",
                    "timestamp": 1.5,
                    "panel_image": str(image_path),
                }],
            }]
        }),
        encoding="utf-8",
    )

    config_path = tmp_path / "phase12.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "transcript_dir": str(transcript_dir),
            "keyframes_dir": str(keyframes_dir),
            "layout_dir": str(layout_dir),
            "speaker_dir": str(speaker_dir),
            "emotion_dir": str(emotion_dir),
            "styled_dir": str(tmp_path / "styled"),
            "output_dir": str(output_dir),
            "page_width": 400,
            "page_height": 500,
            "page_margin": 12,
            "panel_gap": 8,
            "bubble_font_size": 16,
            "bubble_min_font_size": 10,
        }),
        encoding="utf-8",
    )

    result = run_phase12(video_id, config_path)
    report_path = output_dir / video_id / "dialogue_frame_alignment.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert result["metrics"]["segments_placed"] == 1
    assert result["metrics"]["segments_not_placed"] == ["2"]
    assert report["alignments"][0]["dialogue"]["segment_id"] == "1"
    assert report["unplaced_segment_ids"] == ["2"]
