
"""
Phase 12: Dialogue-aware comic assembly and export.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps
from src.story.planning import supported_action_caption, validate_story_output

LOGGER = logging.getLogger("phase12")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS: dict[str, Any] = {
    "transcript_dir": "data/subtitles",
    "keyframes_dir": "data/keyframes",
    "layout_dir": "data/layout",
    "speaker_dir": "data/speaker",
    "emotion_dir": "data/emotions",
    "editorial_dir": "data/editorial",
    "styled_dir": "data/outputs",
    "output_dir": "output",
    "bubble_margin": 18,
    "bubble_padding": 12,
    "bubble_max_width_ratio": 0.72,
    "bubble_max_height_ratio": 0.36,
    "bubble_font_size": 20,
    "bubble_min_font_size": 12,
    "bubble_fill": "#fffdf7",
    "bubble_outline": "#171717",
    "bubble_text": "#111111",
    "caption_fill": "#ffdf42",
    "caption_outline": "#171717",
    "caption_text": "#111111",
    "speaker_label_color": "#555555",
    "visual_only_captions": False,
    "allow_editorial_dialogue_override": False,
    "panel_border_width": 3,
    "page_width": 1200,
    "page_height": 1200,
    "page_margin": 28,
    "panel_gap": 18,
    "max_bubbles_per_panel": 0,
    "bubble_layout_max_coverage_ratio": 0.42,
    "max_merged_dialogue_chars": 110,
    "bubble_row_gap": 10,
    "panels_per_page": 6,
    "adaptive_pages": False,
    "prefer_styled_panels": False,
    "allow_duplicate_segments": False,
    "fallback_nearest_segment_seconds": 1.5,
    "export_pdf": True,
    "export_cbz": True,
    "export_html": True,
}


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Could not read JSON %s: %s", path, exc)
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config = dict(DEFAULTS)
    if not path.exists():
        LOGGER.warning("Config not found: %s. Using defaults.", path)
        return config

    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            config.update(loaded)
    except ImportError:
        LOGGER.warning(
            "PyYAML is not installed. Install with: pip install pyyaml"
        )
    except Exception as exc:
        LOGGER.warning("Could not read config %s: %s", path, exc)

    return config


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def artifact_path(
    config: dict[str, Any],
    key: str,
    video_id: str,
) -> Path:
    base = resolve_path(str(config[key]))
    candidates = [
        base / f"{video_id}.json",
        base / video_id / f"{video_id}.json",
    ]

    found = first_existing(candidates)
    if found:
        return found

    if base.is_dir():
        matches = sorted(base.glob(f"{video_id}*.json"))
        if matches:
            return matches[0]

    return candidates[0]


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def first_value(
    record: dict[str, Any],
    keys: Iterable[str],
    default: Any = None,
) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return default


def unwrap_records(
    data: Any,
    preferred_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if not isinstance(data, dict):
        return []

    for key in preferred_keys:
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    for value in data.values():
        if isinstance(value, dict):
            for key in preferred_keys:
                records = value.get(key)
                if isinstance(records, list):
                    return [
                        item for item in records
                        if isinstance(item, dict)
                    ]

    return []


def normalize_id(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    return str(value)


def normalize_segment_ids(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (str, int, float)):
        return [str(value)]

    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict):
                candidate = first_value(
                    item,
                    ("segment_id", "segmentId", "id", "segment"),
                )
                if candidate is not None:
                    result.append(str(candidate))
            elif item is not None:
                result.append(str(item))
        return list(dict.fromkeys(result))

    return []


def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_transcript(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    records = unwrap_records(
        data,
        ("segments", "transcript", "utterances", "subtitles", "results"),
    )

    if not records and isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict) and any(
                field in value for field in ("text", "dialogue", "content")
            ):
                item = dict(value)
                item.setdefault("segment_id", key)
                records.append(item)

    transcript: dict[str, dict[str, Any]] = {}

    for index, record in enumerate(records):
        segment_id = first_value(
            record,
            ("segment_id", "segmentId", "id", "uid"),
            index,
        )
        start = to_float(
            first_value(record, ("start", "start_time", "startTime", "begin")),
            0.0,
        )
        end = to_float(
            first_value(record, ("end", "end_time", "endTime", "finish")),
            start,
        )
        text_value = first_value(
            record,
            ("text", "dialogue", "content", "sentence", "transcript"),
            "",
        )

        if isinstance(text_value, dict):
            text_value = first_value(
                text_value, ("text", "content"), ""
            )

        text_value = re.sub(
            r"\s+", " ", str(text_value or "")
        ).strip()

        if not text_value:
            continue

        transcript[str(segment_id)] = {
            "segment_id": str(segment_id),
            "start": float(start or 0.0),
            "end": float(end if end is not None else (start or 0.0)),
            "text": text_value,
            "raw": record,
        }

    LOGGER.info(
        "Loaded %d transcript segments from %s",
        len(transcript),
        path,
    )
    return transcript


def load_emotions(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    records = unwrap_records(
        data, ("segments", "emotions", "results", "predictions")
    )
    emotions: dict[str, dict[str, Any]] = {}

    if isinstance(data, dict) and not records:
        for segment_id, value in data.items():
            if not isinstance(value, dict):
                continue

            fused = as_dict(value.get("fused"))
            label = first_value(
                fused,
                ("label", "emotion"),
                first_value(
                    value,
                    ("label", "emotion", "predicted_label"),
                    None,
                ),
            )

            if label:
                confidence = to_float(
                    first_value(
                        fused,
                        ("conf", "confidence", "score"),
                        first_value(
                            value, ("confidence", "conf", "score"), 0.0
                        ),
                    ),
                    0.0,
                )
                emotions[str(segment_id)] = {
                    "label": str(label).lower(),
                    "confidence": float(confidence or 0.0),
                }

    for record in records:
        segment_id = first_value(
            record, ("segment_id", "segmentId", "id", "uid")
        )
        if segment_id is None:
            continue

        fused = as_dict(record.get("fused"))
        label = first_value(
            fused,
            ("label", "emotion"),
            first_value(
                record, ("label", "emotion", "predicted_label"), "neutral"
            ),
        )
        confidence = first_value(
            fused,
            ("conf", "confidence", "score"),
            first_value(record, ("confidence", "conf", "score"), 0.0),
        )

        emotions[str(segment_id)] = {
            "label": str(label or "neutral").lower(),
            "confidence": float(to_float(confidence, 0.0) or 0.0),
        }

    return emotions


def load_speaker_artifact(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    data = read_json(path)
    if not isinstance(data, dict):
        return {}, []

    segment_speakers: dict[str, dict[str, Any]] = {}
    turns: list[dict[str, Any]] = []

    for key in ("turn_matches", "speaker_turns", "turns", "matches", "segments"):
        value = data.get(key)
        if not isinstance(value, list):
            continue

        for item in value:
            if not isinstance(item, dict):
                continue

            if key in ("turn_matches", "speaker_turns", "turns", "matches"):
                turns.append(dict(item))

            segment_id = first_value(
                item,
                ("segment_id", "segmentId", "transcript_segment_id"),
            )
            speaker_id = first_value(
                item,
                ("speaker_id", "speaker", "speakerId", "track_id", "track"),
            )
            character_id = first_value(
                item,
                ("character_id", "character", "face_id", "person_id"),
            )

            if segment_id is not None and (
                speaker_id is not None or character_id is not None
            ):
                confidence = to_float(
                    first_value(
                        item, ("confidence", "score", "match_confidence"), 0.0
                    ),
                    0.0,
                )
                segment_speakers[str(segment_id)] = {
                    "speaker_id": normalize_id(speaker_id),
                    "character_id": normalize_id(character_id),
                    "track_id": normalize_id(first_value(item, ("matched_track_id", "track_id", "matched_track"))),
                    "confidence": float(confidence or 0.0),
                }

    for map_key in (
        "segment_speakers",
        "speaker_by_segment",
        "attributions",
        "segment_mapping",
    ):
        mapping = data.get(map_key)
        if not isinstance(mapping, dict):
            continue

        for segment_id, value in mapping.items():
            if isinstance(value, dict):
                speaker_id = first_value(
                    value, ("speaker_id", "speaker", "speakerId", "track_id")
                )
                character_id = first_value(
                    value, ("character_id", "character", "face_id", "person_id")
                )
                confidence = to_float(
                    first_value(value, ("confidence", "score"), 0.0), 0.0
                )
                segment_speakers[str(segment_id)] = {
                    "speaker_id": normalize_id(speaker_id),
                    "character_id": normalize_id(character_id),
                    "track_id": normalize_id(first_value(value, ("matched_track_id", "track_id", "matched_track"))),
                    "confidence": float(confidence or 0.0),
                }
            elif value is not None:
                segment_speakers[str(segment_id)] = {
                    "speaker_id": str(value),
                    "character_id": None,
                    "confidence": 0.0,
                }

    for turn in turns:
        turn["_start"] = to_float(
            first_value(turn, ("start", "start_time", "startTime", "turn_start"))
        )
        turn["_end"] = to_float(
            first_value(turn, ("end", "end_time", "endTime", "turn_end"))
        )
        turn["_speaker_id"] = normalize_id(
            first_value(turn, ("speaker_id", "speaker", "speakerId", "track_id", "track"))
        )
        turn["_character_id"] = normalize_id(
            first_value(turn, ("character_id", "character", "face_id", "person_id"))
        )
        turn["_track_id"] = normalize_id(
            first_value(turn, ("matched_track_id", "track_id", "matched_track"))
        )
        turn["_confidence"] = to_float(
            first_value(turn, ("speaker_confidence", "calibrated_confidence"))
        )
        turn["_match_score"] = to_float(
            first_value(turn, ("match_score", "score"))
        )
        turn["_decision"] = str(turn.get("decision", "unknown"))

    return segment_speakers, turns


def speaker_for_segment(
    segment: dict[str, Any],
    segment_speakers: dict[str, dict[str, Any]],
    speaker_turns: list[dict[str, Any]],
) -> dict[str, Any]:
    segment_id = str(segment["segment_id"])

    if segment_id in segment_speakers:
        direct = segment_speakers[segment_id]
        if direct.get("speaker_id") or direct.get("character_id"):
            return direct

    start = float(segment["start"])
    end = float(segment["end"])
    best_score = 0.0
    best_turn: dict[str, Any] | None = None

    for turn in speaker_turns:
        turn_start = turn.get("_start")
        turn_end = turn.get("_end")

        if turn_start is None or turn_end is None:
            continue

        overlap = max(
            0.0,
            min(end, turn_end) - max(start, turn_start),
        )
        score = overlap / max(0.001, end - start)

        if score <= 0:
            distance = min(
                abs(start - turn_end),
                abs(turn_start - end),
            )
            if distance <= 0.25:
                score = 0.01

        if score > best_score:
            best_score = score
            best_turn = turn

    if best_turn:
        character_id = best_turn.get("_character_id")
        track_id = best_turn.get("_track_id")
        if character_id is not None and str(character_id).casefold() in {"unknown", "none", ""}:
            character_id = None
        if track_id is not None and str(track_id).casefold() in {"unknown", "none", ""}:
            track_id = None
        return {
            "speaker_id": best_turn.get("_speaker_id"),
            "character_id": character_id,
            "track_id": track_id,
            "confidence": best_turn.get("_confidence"),
            "match_score": best_turn.get("_match_score"),
            "decision": best_turn.get("_decision", "unknown"),
        }

    return {
        "speaker_id": None,
        "character_id": None,
        "track_id": None,
        "confidence": None,
        "match_score": None,
        "decision": "unknown",
    }


def speaker_display_labels(speaker_turns: list[dict[str, Any]]) -> dict[str, str]:
    """Give diarization IDs stable, reader-friendly labels without naming people."""
    ordered_ids = []
    for turn in sorted(
        speaker_turns,
        key=lambda item: item.get("_start") if item.get("_start") is not None else float("inf"),
    ):
        speaker_id = turn.get("_speaker_id")
        if speaker_id is None or str(speaker_id).casefold() in {"unknown", "none", ""}:
            continue
        value = str(speaker_id)
        if value not in ordered_ids:
            ordered_ids.append(value)
    return {
        speaker_id: f"Speaker {index}"
        for index, speaker_id in enumerate(ordered_ids, start=1)
    }


def speaker_face_for_panel(
    panel: dict[str, Any],
    character_id: Any,
    track_id: Any,
) -> dict[str, Any] | None:
    """Find the Phase 10 face ROI associated with a Phase 7 speaker."""
    character_id = (
        None
        if character_id is not None and str(character_id).casefold() in {"unknown", "none", ""}
        else character_id
    )
    track_id = (
        None
        if track_id is not None and str(track_id).casefold() in {"unknown", "none", ""}
        else track_id
    )
    if character_id is None and track_id is None:
        return None
    for roi in as_list(panel.get("rois_panel_coordinates")):
        if not isinstance(roi, dict) or roi.get("kind") not in ("face", "speaker_face"):
            continue
        character_match = (
            character_id is not None
            and str(roi.get("character_id")) == str(character_id)
        )
        track_match = (
            track_id is not None
            and str(roi.get("track_id")) == str(track_id)
        )
        if character_match or track_match:
            return roi
    return None


def compact_speaker_face_box(
    roi: dict[str, Any],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Turn oversized person-track ROIs into a face-sized bubble anchor."""
    box = roi.get("box")
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    x1, y1, x2, y2 = [int(value) for value in box[:4]]
    mouth = roi.get("mouth_panel_coordinates")
    if not isinstance(mouth, (list, tuple)) or len(mouth) < 2:
        return x1, y1, x2, y2
    mouth_x, mouth_y = int(mouth[0]), int(mouth[1])
    box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
    if box_w <= width * 0.32 and box_h <= height * 0.48:
        return x1, y1, x2, y2
    face_w = min(box_w, max(40, int(width * 0.28)))
    face_h = min(box_h, max(44, int(height * 0.36)))
    face_x1 = max(0, min(width - face_w, mouth_x - face_w // 2))
    face_y1 = max(0, min(height - face_h, mouth_y - int(face_h * 0.72)))
    return face_x1, face_y1, face_x1 + face_w, face_y1 + face_h

def load_layout_panels(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, dict):
        return []

    panels: list[dict[str, Any]] = []

    for page_index, page in enumerate(as_list(data.get("pages")), 1):
        if not isinstance(page, dict):
            continue

        page_number = int(
            to_float(
                first_value(page, ("page", "page_number", "page_id")),
                page_index,
            ) or page_index
        )

        for panel_index, panel in enumerate(as_list(page.get("panels")), 1):
            if not isinstance(panel, dict):
                continue

            item = dict(panel)
            item.setdefault("page", page_number)
            item.setdefault(
                "panel_id",
                first_value(panel, ("panel_id", "id", "index"), panel_index),
            )
            panels.append(item)

    if not panels:
        records = unwrap_records(
            data, ("panels", "layout_panels", "panel_records")
        )
        for index, panel in enumerate(records, 1):
            item = dict(panel)
            item.setdefault("page", 1)
            item.setdefault(
                "panel_id",
                first_value(panel, ("panel_id", "id", "index"), index),
            )
            panels.append(item)

    LOGGER.info("Loaded %d layout panels from %s", len(panels), path)
    return panels


def load_keyframe_records(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    records = unwrap_records(
        data,
        (
            "panels",
            "keyframes",
            "selected_frames",
            "frames",
            "candidates",
            "results",
        ),
    )

    by_frame: dict[str, dict[str, Any]] = {}

    for record in records:
        frame_id = first_value(
            record,
            ("frame_id", "frameId", "keyframe_id", "id", "frame"),
        )
        if frame_id is not None:
            by_frame[str(frame_id)] = record

    if isinstance(data, dict):
        for key, value in data.items():
            if not isinstance(value, dict):
                continue

            if any(
                field in value
                for field in ("frame_id", "segment_ids", "timestamp")
            ):
                frame_id = first_value(
                    value,
                    ("frame_id", "frameId", "keyframe_id", "id"),
                    key,
                )
                by_frame[str(frame_id)] = value

    LOGGER.info("Loaded %d keyframe records from %s", len(by_frame), path)
    return by_frame


def join_keyframes(
    panels: list[dict[str, Any]],
    keyframes: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    matched = 0
    joined: list[dict[str, Any]] = []

    for original in panels:
        panel = dict(original)
        frame_id = first_value(
            panel, ("frame_id", "frameId", "keyframe_id", "keyframe")
        )
        frame = keyframes.get(str(frame_id)) if frame_id is not None else None
        # Phase 10 may choose a wider frame from the same Phase 4 shot to
        # retain speaker and listener faces. Keep the original Phase 8 record
        # as metadata context when that replacement is not itself a selected
        # keyframe; the rendered panel image remains the Phase 10 frame.
        if frame is None:
            original_frame_id = panel.get("original_frame_id")
            frame = keyframes.get(str(original_frame_id)) if original_frame_id is not None else None

        if frame:
            matched += 1
            field_mapping = {
                "segment_ids": (
                    "segment_ids",
                    "segments",
                    "transcript_segment_ids",
                    "aligned_segment_ids",
                ),
                "speaker_ids": ("speaker_ids", "speakers"),
                "timestamp": ("timestamp", "time", "start", "frame_time"),
                "character_id": ("character_id", "character", "face_id"),
                "shot_id": ("shot_id", "scene_id"),
                "shot_start": ("shot_start",),
                "shot_end": ("shot_end",),
                "story_beat_id": ("story_beat_id",),
                "story_role": ("story_role",),
                "story_importance": ("story_importance",),
                "visual_action_caption": ("visual_action_caption",),
                "visual_action_label": ("visual_action_label",),
                "alignment_method": ("alignment_method",),
                "alignment_confidence": ("alignment_confidence",),
                "image": (
                    "panel_image",
                    "image",
                    "image_path",
                    "styled_image",
                    "path",
                ),
            }

            for target, keys in field_mapping.items():
                if panel.get(target) in (None, "", []):
                    value = first_value(frame, keys)
                    if value is not None:
                        panel[target] = value

        panel["segment_ids"] = normalize_segment_ids(
            first_value(
                panel,
                (
                    "segment_ids",
                    "segments",
                    "transcript_segment_ids",
                    "aligned_segment_ids",
                ),
                [],
            )
        )
        joined.append(panel)

    return joined, matched


def image_candidates(
    panel: dict[str, Any],
    video_id: str,
    config: dict[str, Any],
) -> list[Path]:
    candidates: list[Path] = []

    panel_id = first_value(panel, ("panel_id", "id", "index"))
    prefer_styled = bool(config.get("prefer_styled_panels", False))
    styled_candidate = None
    if panel_id is not None:
        styled_root = resolve_path(str(config.get("styled_dir", "data/outputs")))
        styled_candidate = (
            styled_root / "panels_styled" / video_id / f"panel_{panel_id}.png"
        )
        if prefer_styled:
            candidates.append(styled_candidate)

    explicit = first_value(
        panel,
        (
            "panel_image",
            "image",
            "image_path",
            "styled_image",
            "path",
            "crop_path",
        ),
    )

    if explicit:
        path = Path(str(explicit))
        candidates.append(
            path if path.is_absolute() else PROJECT_ROOT / path
        )

    if not prefer_styled and styled_candidate is not None:
        # Preserve source-video characters and the exact frame coordinates
        # used by Phase 10 face tracking. Styled variants remain an opt-in.
        candidates.append(styled_candidate)

    frame_id = first_value(
        panel, ("frame_id", "frameId", "keyframe_id", "keyframe")
    )
    for key in ("styled_dir", "layout_dir", "keyframes_dir"):
        base = resolve_path(str(config[key]))

        candidates.extend([
            base / video_id / f"{panel_id}.png",
            base / video_id / f"{frame_id}.png"
            if frame_id is not None else base / "__missing__.png",
            base / f"{video_id}_{panel_id}.png",
            base / f"{video_id}_{frame_id}.png"
            if frame_id is not None else base / "__missing__.png",
            base / f"{video_id}.png",
        ])

    candidates.extend([
        PROJECT_ROOT / "data" / "layout" / video_id
        / f"panel_{panel_id}.png",
        PROJECT_ROOT / "data" / "layout"
        / f"{video_id}_panel_{panel_id}.png",
        PROJECT_ROOT / "data" / "outputs" / video_id
        / f"{frame_id}.png"
        if frame_id is not None else PROJECT_ROOT / "__missing__.png",
        PROJECT_ROOT / "data" / "outputs" / video_id
        / f"frame_{frame_id}.png"
        if frame_id is not None else PROJECT_ROOT / "__missing__.png",
        PROJECT_ROOT / "data" / "styled_frames" / video_id
        / f"{frame_id}.png"
        if frame_id is not None else PROJECT_ROOT / "__missing__.png",
    ])

    unique: list[Path] = []
    seen: set[str] = set()

    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)

    return unique


def find_panel_image(
    panel: dict[str, Any],
    video_id: str,
    config: dict[str, Any],
) -> Path:
    candidates = image_candidates(panel, video_id, config)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    panel_id = first_value(panel, ("panel_id", "id", "index"), "?")
    frame_id = first_value(
        panel, ("frame_id", "frameId", "keyframe_id", "keyframe"), "?"
    )
    tried = "\n  ".join(str(path) for path in candidates[:12])

    raise FileNotFoundError(
        f"No image found for panel_id={panel_id}, frame_id={frame_id}, "
        f"video_id={video_id}.\nChecked candidates:\n  {tried}\n"
        "Correct the image path in the Phase 10 layout JSON or update "
        "styled_dir in phase12.yaml."
    )


def load_font(
    size: int,
    bold: bool = False,
    italic: bool = False,
) -> ImageFont.ImageFont:
    if italic and bold:
        names = ["arialbi.ttf", "DejaVuSans-BoldOblique.ttf", "LiberationSans-BoldItalic.ttf"]
    elif italic:
        names = ["ariali.ttf", "DejaVuSans-Oblique.ttf", "LiberationSans-Italic.ttf"]
    elif bold:
        names = ["arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"]
    else:
        names = ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]

    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue

    return ImageFont.load_default()


def text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> float:
    try:
        return float(draw.textlength(text, font=font))
    except AttributeError:
        box = draw.textbbox((0, 0), text, font=font)
        return float(box[2] - box[0])


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = re.split(r"\s+", text.strip())
    lines: list[str] = []
    current = ""

    for word in words:
        if not word:
            continue

        candidate = word if not current else f"{current} {word}"

        if text_width(draw, candidate, font) <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        if text_width(draw, word, font) > max_width:
            piece = ""

            for char in word:
                if piece and text_width(draw, piece + char, font) > max_width:
                    lines.append(piece)
                    piece = char
                else:
                    piece += char

            current = piece
        else:
            current = word

    if current:
        lines.append(current)

    return lines or [""]


def normalize_safe_zone(
    panel: dict[str, Any],
    width: int,
    height: int,
    margin: int,
) -> tuple[int, int, int, int]:
    # Phase 10 saves its panel-local safe region under this key. Honor it so
    # balloons avoid faces and focal objects rather than defaulting to center.
    raw = first_value(panel, (
        "safe_zone",
        "safe_zones_panel_coordinates",
        "safe_bbox",
        "text_safe_zone",
    ))
    if isinstance(raw, list) and raw and isinstance(raw[0], (list, tuple, dict)):
        raw = raw[0]

    if isinstance(raw, dict):
        x1 = to_float(first_value(raw, ("x1", "left", "x")), margin)
        y1 = to_float(first_value(raw, ("y1", "top", "y")), margin)
        x2 = to_float(first_value(raw, ("x2", "right")), width - margin)
        y2 = to_float(first_value(raw, ("y2", "bottom")), height - margin)

    elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
        x1, y1, x2, y2 = [
            to_float(value, 0.0) or 0.0 for value in raw[:4]
        ]

        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height

    else:
        x1, y1, x2, y2 = margin, margin, width - margin, height - margin

    left = max(margin, min(width - margin, int(x1)))
    top = max(margin, min(height - margin, int(y1)))
    right = max(left + 1, min(width - margin, int(x2)))
    bottom = max(top + 1, min(height - margin, int(y2)))

    return left, top, right, bottom


def trim_letterbox_bars(
    image: Image.Image,
) -> tuple[Image.Image, int, int]:
    """Remove only solid near-black top/bottom bars from cinematic frames."""
    gray = image.convert("L")
    width, height = gray.size
    rows = [gray.crop((0, y, width, y + 1)).getdata() for y in range(height)]

    def is_bar(y: int) -> bool:
        values = list(rows[y])
        dark = sum(value < 30 for value in values) / max(1, width)
        return dark >= 0.90 and sum(values) / max(1, width) < 32

    top = 0
    while top < height // 3 and is_bar(top):
        top += 1
    bottom = height
    while bottom > height * 2 // 3 and is_bar(bottom - 1):
        bottom -= 1
    if top + (height - bottom) < max(8, int(height * 0.04)) or bottom - top < height * 0.55:
        return image, 0, 0
    return image.crop((0, top, width, bottom)), top, height - bottom


def draw_bubble(
    image: Image.Image,
    text: str,
    emotion: str,
    speaker_label: str | None,
    safe_zone: tuple[int, int, int, int],
    config: dict[str, Any],
    tail_target: tuple[float, float] | None = None,
    tail_enabled: bool = True,
    tail_to_edge: bool = False,
    position: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    draw = ImageDraw.Draw(image)
    sx1, sy1, sx2, sy2 = safe_zone

    safe_w = max(1, sx2 - sx1)
    safe_h = max(1, sy2 - sy1)
    padding = max(4, int(config["bubble_padding"]))

    # Let long lines wrap into a readable oval centered over the safe region;
    # using only the narrow safe-zone width made many ordinary lines overflow.
    max_w = max(32, min(
        safe_w,
        int(image.width * min(float(config["bubble_max_width_ratio"]), 0.82)),
    ))
    max_h = max(1, safe_h if position is not None else min(
        int(safe_h * float(config["bubble_max_height_ratio"])),
        safe_h,
    ))

    emotion_key = emotion.casefold()
    font_size = max(10, int(config["bubble_font_size"]))
    min_size = max(8, int(config["bubble_min_font_size"]))
    font = load_font(font_size, bold=True)
    label_font = load_font(max(8, int(font_size * 0.68)), bold=True) if speaker_label else None
    lines = wrap_text(draw, text, font, max_w - 2 * padding)

    def line_height(font_obj: ImageFont.ImageFont) -> int:
        try:
            box = draw.textbbox((0, 0), "Ag", font=font_obj)
            return max(1, box[3] - box[1] + 3)
        except Exception:
            return font_size + 4

    while font_size > min_size:
        height_per_line = line_height(font)
        label_font = load_font(max(8, int(font_size * 0.68)), bold=True) if speaker_label else None
        label_height = line_height(label_font) if label_font else 0
        widest_line = max(
            text_width(draw, line, font) for line in lines
        )
        label_width = text_width(draw, speaker_label, label_font) if label_font else 0

        if (
            len(lines) * height_per_line + label_height + (4 if label_font else 0) + 2 * padding <= max_h
            and max(widest_line, label_width) <= max_w - 2 * padding
        ):
            break

        font_size -= 1
        font = load_font(font_size, bold=True)
        lines = wrap_text(draw, text, font, max_w - 2 * padding)

    height_per_line = line_height(font)
    label_font = load_font(max(8, int(font_size * 0.68)), bold=True) if speaker_label else None
    label_height = line_height(label_font) if label_font else 0
    content_w = int(
        max(
            max(text_width(draw, line, font) for line in lines),
            text_width(draw, speaker_label, label_font) if label_font else 0,
        )
    )
    bubble_w = min(max_w, max(70, content_w + 2 * padding))
    bubble_h = min(
        max_h,
        max(38, len(lines) * height_per_line + label_height + (4 if label_font else 0) + 2 * padding),
    )

    safe_center_x = (sx1 + sx2) // 2
    if position is not None:
        px1, py1, px2, py2 = position
        align = str(config.get("bubble_slot_align", "center"))
        speaker_box = config.get("speaker_face_box")
        if isinstance(speaker_box, (list, tuple)) and len(speaker_box) >= 4:
            fx1, fy1, fx2, fy2 = [int(value) for value in speaker_box[:4]]
            face_on_right = (fx1 + fx2) / 2 >= image.width / 2
            target_y = (
                int(tail_target[1] * (image.height - 1))
                if tail_target is not None
                else (fy1 + fy2) // 2
            )
            max_y = max(py1, py2 - bubble_h)
            desired_x = fx1 - bubble_w - padding if face_on_right else fx2 + padding
            x_candidates = {
                px1,
                px2 - bubble_w,
                desired_x,
                fx1 - bubble_w - padding,
                fx2 + padding,
                (fx1 + fx2 - bubble_w) // 2,
            }
            y_candidates = set(range(py1, max_y + 1, 8))
            y_candidates.update((py1, max_y, target_y - bubble_h // 2,
                                 fy1 - bubble_h - padding, fy2 + padding))
            avoid_boxes = list(config.get("avoid_face_boxes", []))
            avoid_boxes.append((fx1, fy1, fx2, fy2))
            target_x = (
                int(tail_target[0] * (image.width - 1))
                if tail_target is not None
                else (fx1 + fx2) // 2
            )
            placements = []
            for candidate_x in x_candidates:
                candidate_x = max(px1, min(px2 - bubble_w, int(candidate_x)))
                for candidate_y in y_candidates:
                    candidate_y = max(py1, min(max_y, int(candidate_y)))
                    intersects_face = any(
                        candidate_x < ax2 + 4 and candidate_x + bubble_w > ax1 - 4
                        and candidate_y < ay2 + 4 and candidate_y + bubble_h > ay1 - 4
                        for ax1, ay1, ax2, ay2 in avoid_boxes
                    )
                    if intersects_face:
                        continue
                    gap_x = max(candidate_x - target_x, 0, target_x - (candidate_x + bubble_w))
                    gap_y = max(candidate_y - target_y, 0, target_y - (candidate_y + bubble_h))
                    distance = math.hypot(gap_x, gap_y)
                    preferred_side = candidate_x + bubble_w <= fx1 if face_on_right else candidate_x >= fx2
                    side_penalty = 0 if preferred_side else 8
                    score = distance + side_penalty + abs(candidate_y + bubble_h / 2 - target_y) * 0.05
                    placements.append((score, candidate_x, candidate_y))
            if placements:
                _, x1, y1 = min(placements)
            else:
                x1 = max(px1, min(px2 - bubble_w, desired_x))
                y1 = max(py1, min(max_y, target_y - bubble_h // 2))
        else:
            x1 = px1 if align == "left" else (
                px2 - bubble_w if align == "right" else px1 + (px2 - px1 - bubble_w) // 2
            )
            y1 = py1 + max(0, (py2 - py1 - bubble_h) // 2)
    else:
        x1 = max(
            int(config["bubble_margin"]),
            min(image.width - int(config["bubble_margin"]) - bubble_w,
                safe_center_x - bubble_w // 2),
        )
        y1 = min(sy2 - 2, max(sy1, int(config.get("bubble_top_offset", sy1))))
    x1 = max(0, min(image.width - bubble_w, x1))
    y1 = max(0, min(image.height - bubble_h, y1))
    x2 = x1 + bubble_w
    y2 = min(image.height, y1 + bubble_h)
    bubble_h = y2 - y1

    overflow = (
        len(lines) * height_per_line + label_height + (4 if label_font else 0) + 2 * padding > max_h
        or any(
            text_width(draw, line, font) > max_w - 2 * padding
            for line in lines
        )
    )

    emotion_palette = {
        "anger": ("#fff0ec", "#c84836"),
        "disgust": ("#f1f7e9", "#617b3b"),
        "fear": ("#eef4ff", "#536fa8"),
        "joy": ("#fff8dc", "#d29a21"),
        "sadness": ("#edf3fb", "#587699"),
        "surprise": ("#f6edff", "#8a58b8"),
        "neutral": (str(config.get("bubble_fill", "#fffdf7")), "#171717"),
    }
    fill, outline = emotion_palette.get(
        emotion_key,
        (str(config.get("bubble_fill", "#fffdf7")), "#171717"),
    )
    text_color = str(config.get("bubble_text", "#111111"))
    outline_width = max(2, int(config.get("bubble_outline_width", 3)))
    radius = max(8, int(min(bubble_w, bubble_h) * 0.12))

    # Draw a directional tail first so the bubble body covers its base and
    # makes the join look continuous. The tip is in panel coordinates and is
    # never arbitrarily clamped to the bottom edge.
    tail_side = None
    tail_tip = None
    if tail_enabled and tail_to_edge and tail_target is None:
        # An off-screen voice gets a short outward pointer. A long filled
        # triangle across the panel obscures the art and reads like a leader.
        if (x1 + x2) / 2 <= image.width / 2:
            tail_target = (min(image.width - 1, x2 + 24) / image.width, ((y1 + y2) / 2) / image.height)
        else:
            tail_target = (max(0, x1 - 24) / image.width, ((y1 + y2) / 2) / image.height)
    if tail_enabled and tail_target is not None:
        target_x = int(max(0.0, min(1.0, tail_target[0])) * (image.width - 1))
        target_y = int(max(0.0, min(1.0, tail_target[1])) * (image.height - 1))
        # Measure mouth distance from the nearest edge of the bubble, rather
        # than its center. A mouth beside a balloon is a valid short tail even
        # when the balloon itself is wide.
        gap_left = max(0, x1 - target_x)
        gap_right = max(0, target_x - x2)
        gap_top = max(0, y1 - target_y)
        gap_bottom = max(0, target_y - y2)
        horizontal_gap = max(gap_left, gap_right)
        vertical_gap = max(gap_top, gap_bottom)
        body_gap = math.hypot(horizontal_gap, vertical_gap)
        panel_diagonal = math.hypot(image.width, image.height)
        max_tail_distance = panel_diagonal * 0.24
        if body_gap <= max_tail_distance and body_gap > 1:
            if horizontal_gap and (not vertical_gap or horizontal_gap <= vertical_gap):
                tail_side = "left" if gap_left else "right"
                base_x = x1 if tail_side == "left" else x2
                base_y = int(max(y1 + radius, min(y2 - radius, target_y)))
                half_base = min(16, max(7, bubble_h // 3))
                base_a = (base_x, max(y1 + 3, base_y - half_base))
                base_b = (base_x, min(y2 - 3, base_y + half_base))
            else:
                tail_side = "top" if gap_top else "bottom"
                base_y = y1 if tail_side == "top" else y2
                base_x = int(max(x1 + radius, min(x2 - radius, target_x)))
                half_base = min(16, max(7, bubble_w // 10))
                base_a = (max(x1 + 3, base_x - half_base), base_y)
                base_b = (min(x2 - 3, base_x + half_base), base_y)

            # Keep long face pointers narrow so they remain speech tails
            # instead of becoming large wedges across the panel artwork.
            if body_gap > panel_diagonal * 0.16:
                half_base = min(half_base, 7)
            elif body_gap > panel_diagonal * 0.10:
                half_base = min(half_base, 7)
            if horizontal_gap and tail_side in ("left", "right"):
                base_a = (base_x, max(y1 + 3, base_y - half_base))
                base_b = (base_x, min(y2 - 3, base_y + half_base))
            else:
                base_a = (max(x1 + 3, base_x - half_base), base_y)
                base_b = (min(x2 - 3, base_x + half_base), base_y)

            # Stop just short of the mouth without crossing a balloon or face.
            distance = math.hypot(target_x - base_x, target_y - base_y)
            inset = min(5.0, max(0.0, distance - 2.0))
            if distance > 0:
                tip = (
                    int(round(target_x - (target_x - base_x) * inset / distance)),
                    int(round(target_y - (target_y - base_y) * inset / distance)),
                )
            else:
                tip = (target_x, target_y)
            # A tail aimed back into its own bubble is misleading.
            if not (x1 < tip[0] < x2 and y1 < tip[1] < y2):
                tail_tip = tip
                draw.polygon([base_a, base_b, tip], fill=fill, outline=outline)

    draw.rounded_rectangle(
        (x1, y1, x2, y2),
        radius=max(12, bubble_h // 2),
        fill=fill,
        outline=outline,
        width=outline_width,
    )

    y = y1 + padding
    if label_font and speaker_label:
        draw.text(
            (x1 + padding, y),
            speaker_label,
            fill=str(config.get("speaker_label_color", "#555555")),
            font=label_font,
        )
        y += label_height + 4

    for line in lines:
        line_w = text_width(draw, line, font)
        draw.text(
            (x1 + (bubble_w - line_w) // 2, y),
            line,
            fill=text_color,
            font=font,
        )
        y += height_per_line

    return {
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "text": text,
        "lines": lines,
        "font_size": font_size,
        "speaker_label": speaker_label,
        "emotion": emotion,
        "emotion_style": emotion_key,
        "text_overflow": bool(overflow),
        "attributed": bool(speaker_label),
        "placement": "top_safe_zone",
        "tail_side": tail_side if tail_tip is not None else None,
        "tail_tip": list(tail_tip) if tail_tip is not None else None,
    }


def measure_bubble(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
    config: dict[str, Any],
    speaker_label: str | None = None,
) -> dict[str, Any] | None:
    """Find the largest readable font whose wrapped text fits a layout slot."""
    padding = max(4, int(config["bubble_padding"]))
    if max_width <= padding * 2 + 8 or max_height <= padding * 2:
        return None
    max_font = max(8, int(config["bubble_font_size"]))
    min_font = max(8, int(config["bubble_min_font_size"]))
    for size in range(max_font, min_font - 1, -1):
        font = load_font(size, bold=True)
        lines = wrap_text(draw, text, font, max_width - 2 * padding)
        bounds = draw.textbbox((0, 0), "Ag", font=font)
        line_height = max(1, bounds[3] - bounds[1] + 3)
        width = max((text_width(draw, line, font) for line in lines), default=0)
        label_font = load_font(max(8, int(size * 0.68)), bold=True) if speaker_label else None
        label_height = 0
        label_width = 0
        if label_font is not None and speaker_label:
            label_bounds = draw.textbbox((0, 0), "Ag", font=label_font)
            label_height = max(1, label_bounds[3] - label_bounds[1] + 3)
            label_width = text_width(draw, speaker_label, label_font)
        height = len(lines) * line_height + label_height + (4 if label_font else 0) + 2 * padding
        if max(width, label_width) <= max_width - 2 * padding and height <= max_height:
            bubble_width = min(max_width, max(40, int(math.ceil(max(width, label_width) + 2 * padding))))
            return {
                "font_size": size,
                "lines": lines,
                "width": bubble_width,
                "height": max(32, int(height)),
            }
    return None


def dialogue_can_share_row(specs: list[dict[str, Any]]) -> bool:
    """Keep chronological dialogue readable when speakers swap sides."""
    for index, spec in enumerate(specs):
        speaker_side = spec.get("_speaker_side")
        expected_side = "left" if index % 2 == 0 else "right"
        # A known speaker on the other side needs a full-width row. Keeping
        # it in the opposing column creates the long, thin leader lines this
        # renderer is designed to avoid.
        if speaker_side is not None and speaker_side != expected_side:
            return False
    return True


def layout_bubble_chunks(
    image: Image.Image,
    specs: list[dict[str, Any]],
    caption_height: int,
    safe_zone: tuple[int, int, int, int],
    face_boxes: list[tuple[int, int, int, int]],
    config: dict[str, Any],
) -> list[tuple[list[dict[str, Any]], list[tuple[int, int, int, int]]]]:
    """Pack dialogue in reading order; split dense panels into continuations."""
    if not specs:
        return [([], [])]
    draw = ImageDraw.Draw(image)
    margin = max(8, int(config["bubble_margin"]))
    gap = max(6, int(config.get("bubble_row_gap", config["bubble_padding"] // 2)))
    top = max(margin, caption_height + gap)
    bottom = image.height - margin
    panel_width = image.width - 2 * margin
    max_coverage_ratio = max(
        0.05,
        min(0.9, float(config.get("bubble_layout_max_coverage_ratio", 0.42))),
    )
    expanded_specs: list[dict[str, Any]] = []
    full_width = min(
        panel_width,
        int(image.width * min(float(config["bubble_max_width_ratio"]), 0.82)),
    )
    for spec in specs:
        text = str(spec.get("text", ""))
        if measure_bubble(
            draw,
            text,
            full_width,
            bottom - top,
            config,
            spec.get("_display_speaker_label"),
        ) is not None:
            expanded_specs.append(spec)
            continue
        size = max(8, int(config["bubble_min_font_size"]))
        font = load_font(size, bold=True)
        lines = wrap_text(draw, text, font, max(24, full_width - 2 * int(config["bubble_padding"])))
        bounds = draw.textbbox((0, 0), "Ag", font=font)
        line_height = max(1, bounds[3] - bounds[1] + 3)
        max_lines = max(1, (bottom - top - 2 * int(config["bubble_padding"])) // line_height)
        for start in range(0, len(lines), max_lines):
            fragment = dict(spec)
            fragment["text"] = " ".join(lines[start:start + max_lines])
            fragment["_source_text"] = str(spec.get("_source_text", text))
            fragment["_continuation"] = start > 0
            fragment["_dialogue_fragment"] = True
            expanded_specs.append(fragment)
    specs = expanded_specs
    chunks: list[tuple[list[dict[str, Any]], list[tuple[int, int, int, int]]]] = []
    cursor = 0

    while cursor < len(specs):
        chosen: tuple[list[dict[str, Any]], list[tuple[int, int, int, int]]] | None = None
        upper = len(specs)
        for end in range(upper, cursor, -1):
            chunk = specs[cursor:end]
            matched_speaker_targets = any(
                spec.get("_tail_target") is not None
                or spec.get("_matched_face") is not None
                for spec in chunk
            )
            columns = (
                2
                if len(chunk) > 1
                and not matched_speaker_targets
                and dialogue_can_share_row(chunk)
                else 1
            )
            rows = math.ceil(len(chunk) / columns)
            column_gap = gap if columns == 2 else 0
            column_width = (panel_width - column_gap) // columns
            row_height_limit = (bottom - top - gap * (rows - 1)) // rows
            if row_height_limit < 36:
                continue

            measurements = []
            fits = True
            for spec in chunk:
                spec_width = column_width
                if spec.get("_tail_target") is not None:
                    # Speaker balloons need room to sit beside a face; a
                    # slightly narrower measure prevents wide balloons from
                    # covering the speaker and creating long tails.
                    ratio = float(spec.get("_bubble_width_ratio") or 0.56)
                    spec_width = min(spec_width, int(image.width * ratio))
                measured = measure_bubble(
                    draw,
                    str(spec.get("text", "")),
                    spec_width,
                    row_height_limit,
                    config,
                    spec.get("_display_speaker_label"),
                )
                if measured is None:
                    fits = False
                    break
                measurements.append(measured)
            if not fits:
                continue

            occupied_area = sum(
                item["width"] * item["height"]
                for item in measurements
            )
            panel_area = max(1, panel_width * (bottom - top))
            if len(chunk) > 1 and occupied_area / panel_area > max_coverage_ratio:
                continue

            row_heights = [
                max(measurements[index]["height"] for index in range(row * columns, min((row + 1) * columns, len(chunk))))
                for row in range(rows)
            ]
            stack_height = sum(row_heights) + gap * (rows - 1)
            max_stack_shift = max(0, bottom - top - stack_height)
            stack_shifts = range(0, max_stack_shift + 1, 8)
            if max_stack_shift not in stack_shifts:
                stack_shifts = list(stack_shifts) + [max_stack_shift]

            def stack_cost(shift: int) -> float:
                cost = abs((top + shift) - max(top, safe_zone[1])) / max(1, image.height) * 0.1
                y = top + shift
                for row in range(rows):
                    for column in range(columns):
                        index = row * columns + column
                        if index >= len(chunk):
                            continue
                        measured = measurements[index]
                        column_x1 = margin + column * (column_width + column_gap)
                        if column == 0:
                            bx1 = column_x1
                            bx2 = bx1 + measured["width"]
                        else:
                            bx2 = column_x1 + column_width
                            bx1 = bx2 - measured["width"]
                        for fx1, fy1, fx2, fy2 in face_boxes:
                            intersection = max(0, min(bx2, fx2) - max(bx1, fx1)) * max(
                                0, min(y + measured["height"], fy2) - max(y, fy1)
                            )
                            face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
                            cost += (intersection / face_area) * 8.0
                        target = chunk[index].get("_tail_target")
                        if target is not None:
                            target_x = float(target[0]) * (image.width - 1)
                            target_y = float(target[1]) * (image.height - 1)
                            gap_x = max(bx1 - target_x, 0.0, target_x - bx2)
                            gap_y = max(y - target_y, 0.0, target_y - (y + measured["height"]))
                            cost += math.hypot(gap_x, gap_y) / max(1.0, math.hypot(image.width, image.height)) * 3.0
                    y += row_heights[row] + gap
                return cost

            stack_y = top + min(stack_shifts, key=stack_cost)

            slots: list[tuple[int, int, int, int]] = []
            for row in range(rows):
                row_height = row_heights[row]
                row_start = stack_y
                for column in range(columns):
                    index = row * columns + column
                    if index >= len(chunk):
                        continue
                    x1 = margin + column * (column_width + column_gap)
                    x2 = x1 + column_width
                    slots.append((x1, row_start, x2, row_start + row_height))
                stack_y += row_height + gap

            chosen = (chunk, slots)
            break

        if chosen is None:
            # Keep an exceptionally long utterance intact in a full-width
            # continuation slot; draw_bubble will reduce its type before clipping.
            chunk = [specs[cursor]]
            slot = (margin, top, image.width - margin, bottom)
            chosen = (chunk, [slot])
        chunks.append(chosen)
        cursor += len(chosen[0])

    return chunks


def draw_caption(
    image: Image.Image,
    text: str,
    config: dict[str, Any],
    max_width_ratio: float = 0.86,
    avoid_boxes: list[tuple[int, int, int, int]] | None = None,
) -> int:
    """Draw a compact yellow narration box in the upper-left of a panel."""
    if not text.strip():
        return 0
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = load_font(max(12, min(16, int(height * 0.044))), bold=True, italic=True)
    padding_x, padding_y = 10, 6
    line_h = max(1, int(height * 0.05))
    avoid_boxes = avoid_boxes or []
    chosen = None
    for ratio in (max_width_ratio, 0.68, 0.56, 0.44):
        max_text_width = max(24, int(width * ratio) - 2 * padding_x)
        lines = wrap_text(draw, text, font, max_text_width)
        box_w = min(
            int(width * ratio),
            max(text_width(draw, line, font) for line in lines) + 2 * padding_x,
        )
        box_h = len(lines) * line_h + 2 * padding_y
        y1 = 8
        left_x = max(8, int(width * 0.09))
        right_x = max(8, width - box_w - 8)
        placements = []
        for x1 in dict.fromkeys((left_x, right_x)):
            x2, y2 = x1 + box_w, y1 + box_h
            overlap_area = sum(
                max(0, min(x2, fx2) - max(x1, fx1))
                * max(0, min(y2, fy2) - max(y1, fy1))
                for fx1, fy1, fx2, fy2 in avoid_boxes
            )
            placements.append((overlap_area, x1, x2, y2, box_w, box_h, lines))
        chosen = min(placements, key=lambda item: item[0])
        if chosen[0] == 0:
            break
    _, x1, x2, y2, box_w, box_h, lines = chosen
    y1 = 8
    draw.polygon(
        [(x1, y1), (x2, y1), (x2 - 12, y2), (x1, y2)],
        fill=str(config.get("caption_fill", "#ffdf42")),
    )
    draw.line(
        [(x1, y1), (x2, y1), (x2 - 12, y2), (x1, y2), (x1, y1)],
        fill=str(config.get("caption_outline", "#171717")),
        width=2,
        joint="curve",
    )
    for index, line in enumerate(lines):
        draw.text(
            (x1 + padding_x, y1 + padding_y + index * line_h),
            line,
            fill=str(config.get("caption_text", "#111111")),
            font=font,
        )
    return box_h


def continuation_artwork(
    panel: dict[str, Any],
    video_id: str,
    specs: list[dict[str, Any]],
    transcript: dict[str, dict[str, Any]],
    target_size: tuple[int, int],
    shot_frames: dict[int, list[dict[str, Any]]],
) -> tuple[Image.Image, str, float, str] | None:
    """Use a different sampled frame for a continued dialogue panel when possible."""
    # Phase 10 face coordinates belong to the original frame. Do not reuse
    # them on a different image, where a speech tail could identify the wrong
    # person or point at empty space.
    has_panel_face_rois = any(
        isinstance(roi, dict)
        and roi.get("kind") in ("face", "speaker_face")
        for roi in as_list(panel.get("rois_panel_coordinates"))
    )
    if has_panel_face_rois or any(
        spec.get("_matched_face") or spec.get("_tail_target")
        for spec in specs
    ):
        return None
    frame_id = str(first_value(panel, ("frame_id", "frameId", "keyframe_id"), ""))
    match = re.match(r"s(\d+)_f", frame_id)
    if not match:
        return None
    shot_id = int(match.group(1))
    candidates = shot_frames.get(shot_id, [])
    if not candidates:
        return None

    segment_ids = [
        str(segment_id)
        for spec in specs
        for segment_id in as_list(spec.get("segment_ids"))
        if str(segment_id) in transcript
    ]
    if not segment_ids:
        return None
    target_time = sum(
        (float(transcript[sid]["start"]) + float(transcript[sid]["end"])) / 2
        for sid in segment_ids
    ) / len(segment_ids)

    alternatives = []
    for candidate in candidates:
        candidate_id = str(candidate.get("frame_id", ""))
        try:
            candidate_time = float(candidate.get("t"))
        except (TypeError, ValueError):
            continue
        if candidate_id == frame_id or abs(candidate_time - target_time) < 0.2:
            continue
        candidate_path = resolve_path(str(candidate.get("path", "")))
        if candidate_path.is_file():
            alternatives.append((abs(candidate_time - target_time), candidate_time, candidate_id, candidate_path))
    if not alternatives:
        return None

    _, candidate_time, candidate_id, candidate_path = min(alternatives)
    try:
        import cv2
        from src.comic.phase11 import cartoonize, load_config as load_style_config
        import numpy as np

        frame = cv2.imread(str(candidate_path))
        if frame is None:
            return None
        crop = panel.get("crop")
        if isinstance(crop, (list, tuple)) and len(crop) >= 4:
            x1, y1, x2, y2 = [int(value) for value in crop[:4]]
            frame = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if frame.size == 0:
            return None
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        styled = cartoonize(resized, load_style_config(PROJECT_ROOT / "configs" / "phase11.yaml"))
        rgb = cv2.cvtColor(styled, cv2.COLOR_BGR2RGB)
        return Image.fromarray(np.asarray(rgb)), candidate_id, candidate_time, str(candidate_path)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        LOGGER.warning("Could not build alternate continuation frame %s: %s", candidate_id, exc)
        return None


def make_context_caption(
    segment_ids: list[str],
    transcript: dict[str, dict[str, Any]],
) -> str:
    """Add a factual narration bridge from the preceding transcript turn."""
    current = [transcript[sid] for sid in segment_ids if sid in transcript]
    if not current:
        return "The scene continues."
    start = min(float(segment["start"]) for segment in current)
    current_ids = set(segment_ids)
    earlier = [
        (float(segment["end"]), str(segment.get("text", "")).strip())
        for sid, segment in transcript.items()
        if sid not in current_ids
        and float(segment.get("end", 0.0)) <= start
        and str(segment.get("text", "")).strip()
    ]
    if not earlier:
        return "The scene opens."
    _, text = max(earlier, key=lambda item: item[0])
    text = re.sub(r"\s+", " ", text)
    if len(text) > 76:
        text = text[:73].rsplit(" ", 1)[0] + "..."
    return f"Earlier: {text}"

def panel_segment_ids(
    panel: dict[str, Any],
    transcript: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:
    ids = normalize_segment_ids(
        first_value(
            panel,
            (
                "segment_ids",
                "segments",
                "transcript_segment_ids",
                "aligned_segment_ids",
            ),
            [],
        )
    )
    ids = [sid for sid in ids if sid in transcript]

    return ids


def page_cell_size(config: dict[str, Any], panel_count: int) -> tuple[int, int]:
    """Return a wide lettering canvas; page layout is applied after lettering."""
    page_w = int(config["page_width"])
    margin = int(config["page_margin"])
    gap = int(config["panel_gap"])
    cell_w = (page_w - 2 * margin - gap) // 2
    aspect_ratio = max(1.0, float(config.get("panel_aspect_ratio", 1.6)))
    cell_h = int(round(cell_w / aspect_ratio))
    return cell_w, cell_h


def fit_panel_to_page_cell(
    image: Image.Image,
    panel: dict[str, Any],
    target_size: tuple[int, int],
    trim_top: int = 0,
) -> Image.Image:
    """Crop and size the art before lettering, preserving the page frame."""
    target_w, target_h = target_size
    source_w, source_h = image.size
    scale = max(target_w / source_w, target_h / source_h)
    scaled_w, scaled_h = source_w * scale, source_h * scale
    crop_left = (scaled_w - target_w) / 2
    crop_top = (scaled_h - target_h) / 2

    def convert_box(box: Any) -> list[int] | None:
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            return None
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
        return [
            max(0, int(round(x1 * scale - crop_left))),
            max(0, int(round(y1 * scale - crop_top))),
            min(target_w, int(round(x2 * scale - crop_left))),
            min(target_h, int(round(y2 * scale - crop_top))),
        ]

    zones = panel.get("safe_zones_panel_coordinates")
    if isinstance(zones, list):
        panel["safe_zones_panel_coordinates"] = [
            converted for zone in zones
            if (converted := convert_box(zone)) is not None
        ]
    direct_zone = panel.get("safe_zone")
    converted_direct = convert_box(direct_zone)
    if converted_direct is not None:
        panel["safe_zone"] = converted_direct

    transform = as_dict(panel.get("crop_transform"))
    roi_records = []
    for roi in as_list(panel.get("rois")):
        if not isinstance(roi, dict):
            continue
        box = roi.get("box")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
        if transform:
            x1 = (x1 - to_float(transform.get("offset_x"), 0.0)) * to_float(transform.get("scale_x"), 1.0)
            x2 = (x2 - to_float(transform.get("offset_x"), 0.0)) * to_float(transform.get("scale_x"), 1.0)
            y1 = (y1 - to_float(transform.get("offset_y"), 0.0)) * to_float(transform.get("scale_y"), 1.0) - trim_top
            y2 = (y2 - to_float(transform.get("offset_y"), 0.0)) * to_float(transform.get("scale_y"), 1.0) - trim_top
        converted = convert_box((x1, y1, x2, y2))
        if converted is not None:
            record = {**roi, "box": converted}
            mouth = roi.get("mouth_center")
            if isinstance(mouth, (list, tuple)) and len(mouth) >= 2:
                mx, my = float(mouth[0]), float(mouth[1])
                if transform:
                    mx = (mx - to_float(transform.get("offset_x"), 0.0)) * to_float(transform.get("scale_x"), 1.0)
                    my = (my - to_float(transform.get("offset_y"), 0.0)) * to_float(transform.get("scale_y"), 1.0) - trim_top
                point = convert_box((mx, my, mx, my))
                if point is not None:
                    record["mouth_panel_coordinates"] = [point[0], point[1]]
            roi_records.append(record)
    panel["rois_panel_coordinates"] = roi_records

    return ImageOps.fit(
        image,
        target_size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


STORY_SECTIONS = (
    (37.5, "opening", "Homecoming"),
    (79.1, "medical_crisis", "Palladium crisis"),
    (122.6, "pepper_arrival", "Pepper arrives"),
    (185.6, "expo_conflict", "Expo and relationship conflict"),
    (222.2, "decision", "A difficult decision"),
    (float("inf"), "succession", "Succession and resolution"),
)


def story_section(timestamp: Any) -> tuple[str, str]:
    """Group chronological panels around the scene turns in this source clip."""
    seconds = float(to_float(timestamp, 0.0) or 0.0)
    for boundary, section_id, title in STORY_SECTIONS:
        if seconds < boundary:
            return section_id, title
    return STORY_SECTIONS[-1][1], STORY_SECTIONS[-1][2]


def paginate_story_panels(
    panels: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build fixed-size pages without breaking the story's scene order."""
    if not panels:
        return []
    if not bool(config.get("adaptive_pages", True)):
        per_page = max(1, int(config.get("panels_per_page", 6)))
        return [
            {"page": page_number, "panels": panels[start:start + per_page]}
            for page_number, start in enumerate(range(0, len(panels), per_page), 1)
        ]

    sections: list[tuple[str, str, list[dict[str, Any]]]] = []
    for panel in panels:
        section_id, title = story_section(panel.get("timestamp"))
        if not sections or sections[-1][0] != section_id:
            sections.append((section_id, title, []))
        sections[-1][2].append(panel)

    page_specs: list[tuple[str, str, list[dict[str, Any]]]] = []
    max_panels = max(3, int(config.get("adaptive_page_max_panels", 5)))
    for section_id, title, section_panels in sections:
        dialogue_chars = [
            sum(len(str(bubble.get("text", ""))) for bubble in panel.get("bubbles", []))
            for panel in section_panels
        ]
        mean_chars = sum(dialogue_chars) / max(1, len(dialogue_chars))
        target_size = 5 if mean_chars < 50 else (4 if mean_chars < 88 else 3)
        target_size = min(target_size, max_panels)
        page_count = max(1, math.ceil(len(section_panels) / target_size))
        while page_count > 1 and len(section_panels) // page_count < 3:
            page_count -= 1
        base, remainder = divmod(len(section_panels), page_count)
        cursor = 0
        for page_index in range(page_count):
            size = base + (1 if page_index < remainder else 0)
            page_specs.append((section_id, title, section_panels[cursor:cursor + size]))
            cursor += size

    page_records = []
    section_page_counts: dict[str, int] = {}
    for page_number, (section_id, title, group) in enumerate(page_specs, 1):
        style = {
            "opening": "establishing",
            "medical_crisis": "dialogue",
            "pepper_arrival": "dialogue",
            "expo_conflict": "montage",
            "decision": "emotional",
            "succession": "dialogue",
        }.get(section_id, "dialogue")
        section_page_index = section_page_counts.get(section_id, 0)
        section_page_counts[section_id] = section_page_index + 1
        if section_page_index:
            alternate_styles = {
                "opening": "montage",
                "medical_crisis": "montage",
                "pepper_arrival": "emotional",
                "expo_conflict": "dialogue",
                "decision": "dialogue",
                "succession": "resolution",
            }
            style = alternate_styles.get(section_id, style)
        page_records.append({
            "page": page_number,
            "section": title,
            "layout_style": style,
            "panels": group,
        })
    return page_records


def build_pages(
    page_records: list[dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
) -> list[Path]:
    page_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_page_files = {
        f"page_{int(page['page']):02d}.png" for page in page_records
    }
    for existing_page in output_dir.glob("page_*.png"):
        if re.fullmatch(r"page_\d{2,}\.png", existing_page.name) and existing_page.name not in expected_page_files:
            existing_page.unlink()

    page_w = int(config["page_width"])
    page_h = int(config["page_height"])
    margin = int(config["page_margin"])
    gap = int(config["panel_gap"])

    for page in page_records:
        panels = page["panels"]
        canvas = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(canvas)
        page_no = int(page["page"])

        count = max(1, len(panels))
        usable_w = page_w - 2 * margin
        usable_h = page_h - 2 * margin
        layout_style = str(page.get("layout_style", "dialogue"))

        # Vary panel shapes by the beat structure while keeping the reading
        # order chronological (top-to-bottom, then left-to-right).
        if count == 1:
            boxes = [(margin, margin, usable_w, usable_h)]
        elif count == 3:
            half_w = (usable_w - gap) // 2
            if layout_style == "resolution":
                top_h = int((usable_h - gap) * 0.47)
                bottom_h = usable_h - gap - top_h
                boxes = [(margin, margin, half_w, top_h),
                         (margin + half_w + gap, margin, half_w, top_h),
                         (margin, margin + top_h + gap, usable_w, bottom_h)]
            else:
                top_h = int((usable_h - gap) * 0.48)
                bottom_h = usable_h - gap - top_h
                boxes = [(margin, margin, usable_w, top_h),
                         (margin, margin + top_h + gap, half_w, bottom_h),
                         (margin + half_w + gap, margin + top_h + gap, half_w, bottom_h)]
            if layout_style == "montage":
                row_h = (usable_h - 2 * gap) // 3
                boxes = [
                    (margin, margin + index * (row_h + gap), usable_w, row_h)
                    for index in range(3)
                ]
        elif count == 4 and layout_style in {"establishing", "emotional", "resolution"}:
            hero_h = int((usable_h - 2 * gap) * 0.36)
            middle_h = int((usable_h - 2 * gap) * 0.30)
            final_h = usable_h - hero_h - middle_h - 2 * gap
            half_w = (usable_w - gap) // 2
            boxes = [
                (margin, margin, usable_w, hero_h),
                (margin, margin + hero_h + gap, half_w, middle_h),
                (margin + half_w + gap, margin + hero_h + gap, half_w, middle_h),
                (margin, margin + hero_h + middle_h + 2 * gap, usable_w, final_h),
            ]
        elif count == 4:
            half_w = (usable_w - gap) // 2
            top_h = max(1, int(half_w / 1.6))
            lower_h = (usable_h - top_h - 2 * gap) // 2
            boxes = [
                (margin, margin, half_w, top_h),
                (margin + half_w + gap, margin, half_w, top_h),
                (margin, margin + top_h + gap, usable_w, lower_h),
                (margin, margin + top_h + lower_h + 2 * gap, usable_w, lower_h),
            ]
        elif count == 5:
            top_h = int((usable_h - gap) * 0.40)
            row_h = (usable_h - top_h - 2 * gap) // 2
            half_w = (usable_w - gap) // 2
            boxes = [(margin, margin, usable_w, top_h)]
            for row in range(2):
                y = margin + top_h + gap + row * (row_h + gap)
                boxes.extend([(margin, y, half_w, row_h),
                              (margin + half_w + gap, y, half_w, row_h)])
        elif count == 6 and layout_style == "establishing":
            top_h = int((usable_h - 3 * gap) * 0.38)
            remaining_h = usable_h - top_h - gap
            row_h = (remaining_h - 2 * gap) // 3
            half_w = (usable_w - gap) // 2
            boxes = [(margin, margin, usable_w, top_h)]
            for row in range(2):
                y = margin + top_h + gap + row * (row_h + gap)
                boxes.extend([(margin, y, half_w, row_h),
                              (margin + half_w + gap, y, half_w, row_h)])
            final_y = margin + top_h + gap + 2 * (row_h + gap)
            boxes.append((margin + (usable_w - half_w) // 2, final_y, half_w, row_h))
        else:
            cols = 2 if count >= 2 else 1
            rows = math.ceil(count / cols)
            cell_w = (usable_w - (cols - 1) * gap) // cols
            cell_h = (usable_h - (rows - 1) * gap) // rows
            boxes = [
                (margin + (index % cols) * (cell_w + gap),
                 margin + (index // cols) * (cell_h + gap), cell_w, cell_h)
                for index in range(count)
            ]

        for index, panel in enumerate(panels):
            image_path = Path(panel["image"])

            with Image.open(image_path) as src:
                panel_image = src.convert("RGB")

            x, y, cell_w, cell_h = boxes[index]

            # Panels are framed before text is drawn so the page fit never
            # slices a balloon, caption, tail, or character after lettering.
            fitted = ImageOps.contain(
                panel_image,
                (cell_w, cell_h),
                method=Image.Resampling.LANCZOS,
            )
            px = x + (cell_w - fitted.width) // 2
            py = y + (cell_h - fitted.height) // 2
            canvas.paste(fitted, (px, py))
            draw.rectangle(
                (
                    px,
                    py,
                    px + fitted.width - 1,
                    py + fitted.height - 1,
                ),
                outline="#111111",
                width=2,
            )

        page_path = output_dir / f"page_{page_no:02d}.png"
        canvas.save(page_path)
        page_paths.append(page_path)

    return page_paths


def write_html(
    page_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    chunks = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>CineComic Export</title><style>",
        "body{font-family:Arial,sans-serif;background:#eee;"
        "margin:0;padding:24px;color:#111}",
        ".page{max-width:900px;margin:0 auto 32px;background:white;"
        "padding:16px;box-shadow:0 2px 8px #bbb}",
        ".panel{margin:12px 0}",
        ".panel img{display:block;max-width:100%;height:auto;"
        "margin:auto;border:2px solid #111}",
        ".dialogue{font-size:14px;margin:6px auto;max-width:90%;"
        "padding:8px;border:1px solid #777;border-radius:12px;"
        "background:#fffdf7}",
        "</style></head><body><h1>CineComic</h1>",
    ]

    for page in page_records:
        chunks.append(
            f"<section class='page'><h2>Page {int(page['page'])}</h2>"
        )

        for panel in page["panels"]:
            chunks.append("<div class='panel'>")
            image_rel = Path(os.path.relpath(
                panel["image"],
                output_path.parent,
            )).as_posix()
            image_rel = html.escape(image_rel)
            panel_id = html.escape(str(panel["panel_id"]))

            chunks.append(
                f"<img src='{image_rel}' alt='Panel {panel_id}'>"
            )

            for bubble in panel.get("bubbles", []):
                dialogue = html.escape(str(bubble.get("text", "")))
                chunks.append(
                    f"<div class='dialogue'>{dialogue}</div>"
                )

            chunks.append("</div>")

        chunks.append("</section>")

    chunks.append("</body></html>")
    output_path.write_text("\n".join(chunks), encoding="utf-8")


def export_pdf(
    page_images: list[Path],
    output_path: Path,
) -> None:
    if not page_images:
        LOGGER.warning("No page images available for PDF export.")
        return

    images: list[Image.Image] = []

    for path in page_images:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())

    try:
        first_image = images[0]
        remaining_images = images[1:]

        first_image.save(
            output_path,
            save_all=True,
            append_images=remaining_images,
            resolution=150.0,
        )
    finally:
        for image in images:
            image.close()


def export_cbz(
    page_images: list[Path],
    output_path: Path,
) -> None:
    with zipfile.ZipFile(
        output_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for index, path in enumerate(page_images, 1):
            archive.write(path, arcname=f"{index:03d}.png")


def run_phase12(
    video_id: str,
    config_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)

    paths = {
        "transcript": artifact_path(
            config, "transcript_dir", video_id
        ),
        "keyframes": artifact_path(
            config, "keyframes_dir", video_id
        ),
        "layout": artifact_path(
            config, "layout_dir", video_id
        ),
        "speaker": artifact_path(
            config, "speaker_dir", video_id
        ),
        "emotion": artifact_path(
            config, "emotion_dir", video_id
        ),
        "alignment": resolve_path(
            Path("data/alignment") / f"{video_id}.json"
        ),
    }

    if not paths["layout"].exists():
        raise FileNotFoundError(
            f"Required Phase 12 layout artifact not found: {paths['layout']}"
        )

    transcript = (
        load_transcript(paths["transcript"])
        if paths["transcript"].exists()
        else {}
    )
    visual_only_mode = not transcript
    if visual_only_mode:
        LOGGER.info(
            "No dialogue transcript available; composing visual-only comic pages."
        )

    story_plan_path = resolve_path(Path("data/story") / f"{video_id}.json")
    story_plan = as_dict(read_json(story_plan_path))
    story_events = [
        event for event in as_list(story_plan.get("events"))
        if isinstance(event, dict)
    ]
    story_by_frame = {
        str(event.get("frame_id")): event
        for event in story_events
        if event.get("frame_id") is not None
    }
    story_by_beat = {
        str(event.get("beat_id")): event
        for event in story_events
        if event.get("beat_id") is not None
    }
    scene_first_panels = {
        str(
            scene.get("summary_anchor_panel_id")
            or scene["panels"][0].get("panel_id")
        )
        for scene in as_list(story_plan.get("scenes"))
        if isinstance(scene, dict)
        and as_list(scene.get("panels"))
        and isinstance(scene["panels"][0], dict)
    }

    shot_frames: dict[int, list[dict[str, Any]]] = {}
    shot_data = as_dict(read_json(resolve_path(Path("data/shots") / f"{video_id}.json")))
    for shot in as_list(shot_data.get("shots")):
        if not isinstance(shot, dict):
            continue
        try:
            shot_id = int(shot.get("shot_id"))
        except (TypeError, ValueError):
            continue
        shot_frames[shot_id] = [
            frame for frame in as_list(shot.get("frames"))
            if isinstance(frame, dict)
        ]

    panels = load_layout_panels(paths["layout"])
    if not panels:
        raise ValueError(
            f"No panels found in layout artifact: {paths['layout']}"
        )

    keyframes = load_keyframe_records(paths["keyframes"])
    panels, matched_frames = join_keyframes(panels, keyframes)

    emotions = load_emotions(paths["emotion"])
    alignment_data = as_dict(read_json(paths["alignment"]))
    visual_actions = as_dict(alignment_data.get("visual_actions"))
    segment_speakers, speaker_turns = load_speaker_artifact(
        paths["speaker"]
    )
    speaker_labels = speaker_display_labels(speaker_turns)
    editorial_path = resolve_path(
        Path(str(config.get("editorial_dir", "data/editorial")))
        / f"{video_id}.json"
    )
    editorial_data = as_dict(read_json(editorial_path))
    editorial_by_panel = {
        str(item.get("panel_id")): item
        for item in as_list(editorial_data.get("panels"))
        if isinstance(item, dict) and item.get("panel_id") is not None
    }
    panel_image_overrides = {
        str(item.get("panel_id")): item
        for item in as_list(editorial_data.get("panel_image_overrides"))
        if isinstance(item, dict) and item.get("panel_id") is not None
    }
    segment_speaker_overrides = {
        str(item.get("segment_id")): item
        for item in as_list(editorial_data.get("speaker_overrides"))
        if isinstance(item, dict) and item.get("segment_id") is not None
    }
    intentional_omissions = {
        str(segment_id)
        for segment_id in as_list(
            editorial_data.get("omitted_segment_ids")
        )
    }

    output_root = resolve_path(str(config["output_dir"]))
    output_dir = output_root / video_id
    panels_output_dir = output_dir / "panels"
    panels_output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[int, list[dict[str, Any]]] = {}

    for panel in panels:
        page_no = int(
            to_float(
                first_value(panel, ("page", "page_number")),
                1,
            ) or 1
        )
        grouped.setdefault(page_no, []).append(panel)

    for page_no in grouped:
        grouped[page_no].sort(
            key=lambda panel: int(
                to_float(
                    first_value(panel, ("panel_id", "id", "index")),
                    0,
                ) or 0
            )
        )

    page_records: list[dict[str, Any]] = []
    total_bubbles = 0
    unmatched_segments = 0
    segment_seen: set[str] = set()
    overflow_count = 0
    attribution_count = 0
    panels_without_segments = 0

    for page_no in sorted(grouped):
        panel_records: list[dict[str, Any]] = []

        for fallback_index, panel in enumerate(
            grouped[page_no], 1
        ):
            panel_id = first_value(
                panel,
                ("panel_id", "id", "index"),
                fallback_index,
            )

            image_override = panel_image_overrides.get(str(panel_id))
            if image_override:
                override_path = Path(str(image_override.get("image", "")))
                if not override_path.is_absolute():
                    override_path = PROJECT_ROOT / override_path
                if override_path.is_file():
                    panel["original_frame_id"] = panel.get("frame_id")
                    panel["frame_id"] = str(image_override.get("frame_id", panel.get("frame_id", "")))
                    panel["timestamp"] = to_float(image_override.get("timestamp"), panel.get("timestamp"))
                    panel["shot_id"] = image_override.get("shot_id", panel.get("shot_id"))
                    panel["shot_start"] = image_override.get("shot_start", panel.get("shot_start"))
                    panel["shot_end"] = image_override.get("shot_end", panel.get("shot_end"))
                    panel["source_image"] = str(override_path)
                    panel["panel_image"] = str(override_path)
                    # Do not transfer face matches from the replaced frame;
                    # that would point the speaker tail at the wrong person.
                    if bool(image_override.get("clear_face_rois", True)):
                        panel["rois"] = []
                        panel["rois_panel_coordinates"] = []

            image_path = find_panel_image(
                panel, video_id, config
            )

            # Phase 11 artwork may use a different crop from the source frame
            # that Phase 10 analyzed. Face coordinates then point at stale
            # locations in the styled image (for example, at a prop instead
            # of the tracked speaker). Keep the source crop whenever tracked
            # faces are available so bubble placement and artwork share the
            # same coordinate space and visible characters.
            tracked_faces = any(
                isinstance(roi, dict)
                and roi.get("kind") in ("face", "speaker_face")
                and isinstance(roi.get("box"), (list, tuple))
                and len(roi["box"]) >= 4
                for roi in as_list(panel.get("rois"))
            )
            source_panel_image = panel.get("panel_image")
            if tracked_faces and source_panel_image:
                source_path = Path(str(source_panel_image))
                if not source_path.is_absolute():
                    source_path = PROJECT_ROOT / source_path
                if source_path.is_file():
                    image_path = source_path

            with Image.open(image_path) as source:
                panel_image = source.convert("RGB").copy()

            panel_image, trim_top, trim_bottom = trim_letterbox_bars(panel_image)
            if trim_top or trim_bottom:
                panel["safe_zones_panel_coordinates"] = [
                    [
                        zone[0],
                        max(0, int(zone[1]) - trim_top),
                        zone[2],
                        min(panel_image.height, int(zone[3]) - trim_top),
                    ]
                    for zone in as_list(panel.get("safe_zones_panel_coordinates"))
                    if isinstance(zone, (list, tuple)) and len(zone) >= 4
                ]

            target_size = page_cell_size(config, len(grouped[page_no]))
            panel_image = fit_panel_to_page_cell(
                panel_image, panel, target_size, trim_top=trim_top
            )

            width, height = panel_image.size
            safe_zone = normalize_safe_zone(
                panel,
                width,
                height,
                int(config["bubble_margin"]),
            )

            segment_ids = panel_segment_ids(
                panel, transcript, config
            )
            editorial = editorial_by_panel.get(str(panel_id), {})
            has_editorial_dialogue = (
                bool(config.get("allow_editorial_dialogue_override", False))
                and "dialogue" in editorial
            )
            if has_editorial_dialogue:
                source_ids = [
                    str(value)
                    for value in as_list(editorial.get("source_segment_ids"))
                    if str(value) in transcript
                ]
                segment_ids = source_ids

            segment_ids = sorted(
                segment_ids,
                key=lambda sid: float(transcript[sid].get("start", 0.0)),
            )
            caption_face_boxes = [
                compact_speaker_face_box(roi, width, height)
                for roi in as_list(panel.get("rois_panel_coordinates"))
                if isinstance(roi, dict)
                and roi.get("kind") in ("face", "speaker_face")
                and isinstance(roi.get("box"), (list, tuple))
                and len(roi["box"]) >= 4
            ]
            caption_face_boxes = [
                box for box in caption_face_boxes if box is not None
            ]

            frame_id = first_value(panel, ("frame_id", "frameId", "keyframe_id"))
            visual_action = visual_actions.get(str(frame_id), {})
            action_label = panel.get("visual_action_label")
            if isinstance(visual_action, dict) and not action_label:
                action_label = visual_action.get("label")
            story_event = (
                story_by_frame.get(str(frame_id))
                or story_by_beat.get(str(panel.get("story_beat_id")))
                or {}
            )
            visual_description = str(
                story_event.get("visual_description")
                or supported_action_caption(
                    panel.get("visual_action_caption") or visual_action.get("caption"),
                    action_label,
                )
                or "The visible action is unclear in this frame."
            )
            panel_summary = str(
                story_event.get("panel_summary") or visual_description
            )
            if (
                str(story_event.get("frame_id")) != str(frame_id)
                and isinstance(visual_action, dict)
                and visual_action.get("status") == "ok"
            ):
                actual_frame_caption = supported_action_caption(
                    visual_action.get("caption"),
                    visual_action.get("label"),
                )
                if actual_frame_caption:
                    visual_description = actual_frame_caption
                    context = story_event.get("dialogue_context")
                    panel_summary = visual_description + (" " + str(context) if context else "")
            scene_summary = str(story_event.get("scene_summary") or "")
            scene_id = story_event.get("scene_id")
            caption_lines = []
            if str(story_event.get("beat_id")) in scene_first_panels and scene_summary:
                caption_lines.append(f"SCENE: {scene_summary}")
            caption_lines.append(f"PANEL: {panel_summary}")
            caption = "\n".join(caption_lines)
            caption_height = draw_caption(
                panel_image,
                caption,
                config,
                avoid_boxes=caption_face_boxes,
            ) if caption else 0
            safe_zone = (
                safe_zone[0],
                max(safe_zone[1], caption_height + 16),
                safe_zone[2],
                safe_zone[3],
            )

            if not segment_ids:
                panels_without_segments += 1

            bubbles: list[dict[str, Any]] = []

            bubble_specs: list[dict[str, Any]] = []
            if has_editorial_dialogue:
                corrected_text = editorial.get("dialogue")
                if isinstance(corrected_text, str) and corrected_text.strip():
                    bubble_specs.append({
                        "text": corrected_text.strip(),
                        "segment_ids": segment_ids,
                        "emotion": editorial.get("emotion"),
                        "speaker_label": editorial.get("speaker_label"),
                        "tail_target": editorial.get("tail_target"),
                        "tail_enabled": bool(editorial.get("tail_enabled", True)),
                    })
            else:
                # Join adjacent fragments from the same identified speaker
                # into a single natural balloon (ASR often splits one turn).
                for sid in segment_ids:
                    if sid not in transcript:
                        continue
                    segment = transcript[sid]
                    speaker = speaker_for_segment(
                        segment, segment_speakers, speaker_turns
                    )
                    speaker_override = segment_speaker_overrides.get(str(sid))
                    if speaker_override:
                        speaker = {
                            "speaker_id": speaker_override.get("speaker_id", f"MANUAL_{sid}"),
                            "character_id": speaker_override.get("character_id"),
                            "track_id": speaker_override.get("track_id"),
                            "confidence": 1.0,
                        }
                    speaker_id = speaker.get("speaker_id")
                    start = float(segment["start"])
                    previous = bubble_specs[-1] if bubble_specs else None
                    if (
                        speaker_id
                        and previous
                        and previous.get("_speaker_id") == speaker_id
                        and start - float(previous["_end"]) <= 1.5
                        and len(str(previous["text"])) + len(str(segment["text"])) + 1
                        <= int(config.get("max_merged_dialogue_chars", 110))
                    ):
                        previous["text"] = (
                            str(previous["text"]).rstrip()
                            + " " + str(segment["text"]).lstrip()
                        )
                        previous["segment_ids"].append(sid)
                        previous["_end"] = float(segment["end"])
                    else:
                        bubble_specs.append({
                            "text": segment["text"],
                            "segment_ids": [sid],
                            "_speaker_id": speaker_id,
                            "_speaker_info": speaker if speaker_override else None,
                            "_speaker_label_override": speaker_override.get("speaker_label") if speaker_override else None,
                            "_off_panel_speaker": bool(speaker_override.get("off_panel", False)) if speaker_override else False,
                            "_tail_target_override": speaker_override.get("tail_target") if speaker_override else None,
                            "_tail_target": tuple(speaker_override.get("tail_target")) if speaker_override and isinstance(speaker_override.get("tail_target"), (list, tuple)) else None,
                            "_tail_enabled_override": speaker_override.get("tail_enabled") if speaker_override else None,
                            "_bubble_width_ratio": speaker_override.get("bubble_max_width_ratio") if speaker_override else None,
                            "_speaker_side": speaker_override.get("speaker_side") if speaker_override else None,
                            "_bubble_region": speaker_override.get("bubble_region") if speaker_override else None,
                            "_end": float(segment["end"]),
                        })

            # Resolve the existing speaker mapping before slot layout. The
            # layout can then preserve dialogue order while keeping each
            # bubble close to its matched face rather than treating every
            # balloon as generic text.
            for spec in bubble_specs:
                spec_ids = [str(value) for value in spec.get("segment_ids", [])]
                first_segment = transcript.get(spec_ids[0]) if spec_ids else None
                if has_editorial_dialogue:
                    spec["_display_speaker_label"] = (
                        str(spec.get("speaker_label") or spec.get("_speaker_label_override") or "").strip()
                        or None
                    )
                    continue
                if first_segment is None:
                    spec["_display_speaker_label"] = "Unidentified voice"
                    continue
                speaker_info = spec.get("_speaker_info") or speaker_for_segment(
                    first_segment, segment_speakers, speaker_turns
                )
                matched_face = speaker_face_for_panel(
                    panel,
                    speaker_info.get("character_id"),
                    speaker_info.get("track_id"),
                )
                if matched_face:
                    matched_face = dict(matched_face)
                    compact_box = compact_speaker_face_box(matched_face, width, height)
                    if compact_box:
                        matched_face["box"] = compact_box
                spec["_speaker_info"] = speaker_info
                spec["_matched_face"] = matched_face
                speaker_id = speaker_info.get("speaker_id")
                if spec.get("_speaker_label_override"):
                    display_label = str(spec["_speaker_label_override"])
                elif speaker_id is not None and str(speaker_id).casefold() not in {"unknown", "none", ""}:
                    display_label = speaker_labels.get(str(speaker_id), "Speaker")
                else:
                    display_label = "Unidentified voice"
                spec["_display_speaker_label"] = display_label
                spec["_off_panel_speaker"] = bool(
                    spec.get("_off_panel_speaker")
                    or (
                        matched_face is None
                        and speaker_info.get("track_id")
                        and speaker_info.get("character_id")
                    )
                )
                if spec.get("_off_panel_speaker"):
                    # A manual speaker override may deliberately be off-panel.
                    spec["_matched_face"] = None
                if spec.get("_tail_target_override") is not None:
                    override_target = spec["_tail_target_override"]
                    if isinstance(override_target, (list, tuple)) and len(override_target) >= 2:
                        spec["_tail_target"] = (float(override_target[0]), float(override_target[1]))
                if matched_face:
                    mouth = matched_face.get("mouth_panel_coordinates")
                    if isinstance(mouth, (list, tuple)) and len(mouth) >= 2:
                        spec["_tail_target"] = (mouth[0] / width, mouth[1] / height)
                        # Place the bubble on the opposite side of the face so
                        # the balloon can sit beside the speaker and point in.
                        spec["_speaker_side"] = "right" if mouth[0] < width / 2 else "left"

            face_boxes = caption_face_boxes
            bubble_chunks = layout_bubble_chunks(
                panel_image,
                bubble_specs,
                caption_height,
                safe_zone,
                face_boxes,
                config,
            )

            for chunk_index, (chunk_specs, slots) in enumerate(bubble_chunks):
                chunk_frame_id = str(first_value(panel, ("frame_id", "frameId", "keyframe_id"), ""))
                chunk_timestamp = float(to_float(first_value(panel, ("timestamp", "time", "frame_time")), 0.0) or 0.0)
                chunk_source_image = str(image_path)
                chunk_image = panel_image.copy()
                if chunk_index:
                    alternate = continuation_artwork(
                        panel,
                        video_id,
                        chunk_specs,
                        transcript,
                        panel_image.size,
                        shot_frames,
                    )
                    if alternate is not None:
                        chunk_image, chunk_frame_id, chunk_timestamp, chunk_source_image = alternate
                        if caption:
                            draw_caption(
                                chunk_image,
                                caption,
                                config,
                                avoid_boxes=caption_face_boxes,
                            )
                bubbles: list[dict[str, Any]] = []
                chunk_segment_ids: list[str] = []

                for bubble_index, (spec, slot) in enumerate(zip(chunk_specs, slots)):
                    spec_segment_ids = [str(sid) for sid in spec.get("segment_ids", [])]
                    segment_id = spec_segment_ids[0] if spec_segment_ids else None
                    segment = transcript.get(segment_id) if segment_id else None
                    if not segment:
                        unmatched_segments += 1
                        LOGGER.warning("Panel %s has no transcript source for its dialogue bubble", panel_id)
                        continue

                    already_seen = any(sid in segment_seen for sid in spec_segment_ids)
                    if already_seen and not spec.get("_continuation") and not bool(config.get("allow_duplicate_segments", False)):
                        continue
                    segment_seen.update(spec_segment_ids)
                    chunk_segment_ids.extend(spec_segment_ids)

                    if has_editorial_dialogue:
                        speaker_info = {"speaker_id": None, "character_id": None, "confidence": 0.0}
                    else:
                        speaker_info = spec.get("_speaker_info") or speaker_for_segment(
                            segment, segment_speakers, speaker_turns
                        )
                    speaker_id = speaker_info.get("speaker_id")
                    track_id = speaker_info.get("track_id")
                    character_id = speaker_info.get("character_id")
                    speaker_label = (
                        str(spec.get("speaker_label"))
                        if has_editorial_dialogue and spec.get("speaker_label")
                        else str(spec.get("_speaker_label_override"))
                        if spec.get("_speaker_label_override")
                        else str(spec.get("_display_speaker_label") or "Unidentified voice")
                    )
                    emotion = emotions.get(str(segment_id), {"label": "neutral", "confidence": 0.0})
                    if spec.get("emotion"):
                        emotion = {
                            "label": str(spec["emotion"]),
                            "confidence": 1.0 if has_editorial_dialogue else emotion.get("confidence", 0.0),
                        }

                    tail_target = tuple(spec["tail_target"]) if spec.get("tail_target") else spec.get("_tail_target")
                    matched_face = spec.get("_matched_face")
                    if tail_target is None and (character_id is not None or track_id is not None):
                        matched_face = matched_face or speaker_face_for_panel(
                            panel, character_id, track_id
                        )
                        if matched_face:
                            mouth = matched_face.get("mouth_panel_coordinates")
                            if isinstance(mouth, (list, tuple)) and len(mouth) >= 2:
                                tail_target = (mouth[0] / width, mouth[1] / height)
                            else:
                                rx1, ry1, rx2, ry2 = matched_face["box"]
                                tail_target = ((rx1 + rx2) / (2 * width), (ry1 + ry2) / (2 * height))
                    off_panel_speaker = bool(spec.get("_off_panel_speaker"))

                    bubble_config = dict(config)
                    if matched_face and isinstance(matched_face.get("box"), (list, tuple)):
                        bubble_config["speaker_face_box"] = matched_face["box"]
                        bubble_config["bubble_max_width_ratio"] = min(
                            float(config["bubble_max_width_ratio"]), 0.56
                        )
                    if spec.get("_bubble_width_ratio") is not None:
                        bubble_config["bubble_max_width_ratio"] = min(
                            float(config["bubble_max_width_ratio"]),
                            float(spec["_bubble_width_ratio"]),
                        )
                    bubble_config["avoid_face_boxes"] = face_boxes
                    bubble_config["bubble_slot_align"] = (
                        spec.get("_speaker_side")
                        or ("left" if bubble_index % 2 == 0 else "right")
                    )
                    bubble_position = slot
                    if spec.get("_bubble_region") == "right_column":
                        # Keep a visible speaker's reply beside their face, away
                        # from the left column used for off-panel dialogue.
                        bubble_position = (
                            int(width * 0.60),
                            int(height * 0.36),
                            width - int(config["bubble_margin"]),
                            int(height * 0.82),
                        )
                    if spec.get("_off_panel_speaker") and spec.get("_bubble_region") in {"left_column", "lower_left"}:
                        off_panel_specs = [
                            candidate for candidate in chunk_specs
                            if candidate.get("_off_panel_speaker")
                            and candidate.get("_bubble_region") == spec.get("_bubble_region")
                        ]
                        region_top = (
                            int(height * 0.58)
                            if spec.get("_bubble_region") == "lower_left"
                            else int(config["bubble_margin"])
                        )
                        region_bottom = height - int(config["bubble_margin"])
                        row_height = max(1, (region_bottom - region_top) // max(1, len(off_panel_specs)))
                        region_index = off_panel_specs.index(spec)
                        bubble_position = (
                            int(config["bubble_margin"]),
                            region_top + region_index * row_height,
                            int(width * (0.52 if spec.get("_bubble_region") == "lower_left" else 0.45)),
                            region_top + (region_index + 1) * row_height,
                        )
                        if spec.get("_bubble_region") == "lower_left":
                            bubble_config["bubble_font_size"] = min(int(config["bubble_font_size"]), 15)
                        tail_target = (
                            0.01,
                            ((bubble_position[1] + bubble_position[3]) / 2) / height,
                        )
                    bubble = draw_bubble(
                        chunk_image,
                        str(spec.get("text", segment["text"])),
                        emotion["label"],
                        speaker_label,
                        slot,
                        bubble_config,
                        tail_target=tail_target,
                        tail_enabled=bool(
                            spec.get("_tail_enabled_override")
                            if spec.get("_tail_enabled_override") is not None
                            else spec.get("tail_enabled", True)
                        )
                        and (
                            matched_face is not None
                            or off_panel_speaker
                            or spec.get("tail_target") is not None
                            or spec.get("_tail_target_override") is not None
                        ),
                        tail_to_edge=off_panel_speaker,
                        position=bubble_position,
                    )
                    bubble.update({
                        "segment_id": str(segment_id),
                        "source_segment_ids": spec_segment_ids,
                        "start": segment["start"],
                        "end": max(transcript[sid]["end"] for sid in spec_segment_ids),
                        "emotion": emotion["label"],
                        "emotion_confidence": emotion["confidence"],
                        "speaker_id": speaker_id,
                        "speaker_track_id": track_id,
                        "character_id": character_id,
                        "speaker_face_visible": matched_face is not None,
                        "off_panel_speaker": off_panel_speaker,
                        "mouth_panel_coordinates": matched_face.get("mouth_panel_coordinates") if matched_face else None,
                        "speaker_confidence": speaker_info.get("confidence"),
                        "speaker_match_score": speaker_info.get("match_score"),
                        "speaker_attribution_decision": speaker_info.get("decision", "unknown"),
                        "alignment_method": panel.get("alignment_method", "explicit_phase8_segment"),
                        "alignment_confidence": to_float(panel.get("alignment_confidence")),
                        "transcription_confidence": to_float(first_value(
                            segment.get("raw", {}),
                            ("transcription_confidence", "confidence", "probability"),
                        )),
                        "editorial_correction": bool(has_editorial_dialogue),
                        "dialogue_fragment": bool(spec.get("_dialogue_fragment")),
                        "source_text": str(
                            spec.get("text", segment["text"])
                            if spec.get("_dialogue_fragment")
                            else spec.get("_source_text", spec.get("text", segment["text"]))
                        ),
                    })
                    bubbles.append(bubble)
                    total_bubbles += 1
                    overflow_count += int(bubble["text_overflow"])
                    attribution_count += int(bubble["attributed"])

                ImageDraw.Draw(chunk_image).rectangle(
                    (0, 0, width - 1, height - 1),
                    outline="#111111",
                    width=max(1, int(config["panel_border_width"])),
                )
                output_panel_id = panel_id if chunk_index == 0 else f"{panel_id}-continued-{chunk_index + 1}"
                panel_out = panels_output_dir / f"page_{page_no:02d}_panel_{output_panel_id}.png"
                chunk_image.save(panel_out)
                chunk_segment_ids = sorted(set(chunk_segment_ids), key=lambda sid: float(transcript[sid]["start"]))
                panel_records.append({
                    "panel_id": output_panel_id,
                    "frame_id": chunk_frame_id,
                    "timestamp": chunk_timestamp,
                    "panel_dimensions": [width, height],
                    "shot_id": panel.get("shot_id"),
                    "shot_start": panel.get("shot_start"),
                    "shot_end": panel.get("shot_end"),
                    "story_beat_id": panel.get("story_beat_id"),
                    "story_role": panel.get("story_role"),
                    "story_importance": panel.get("story_importance"),
                    "scene_id": scene_id,
                    "scene_summary": scene_summary or None,
                    "visual_description": visual_description,
                    "panel_summary": panel_summary,
                    "dialogue_context": story_event.get("dialogue_context"),
                    "visual_action_label": panel.get("visual_action_label"),
                    "visible_face_count": sum(
                        1 for roi in as_list(panel.get("rois_panel_coordinates"))
                        if isinstance(roi, dict)
                        and roi.get("kind") in ("face", "speaker_face")
                    ),
                    "segment_ids": chunk_segment_ids,
                    "caption": caption,
                    "source_image": chunk_source_image,
                    "image": str(panel_out),
                    "bubbles": bubbles,
                })

        page_records.append({
            "page": page_no,
            "panels": panel_records,
        })

    ordered_panels = [
        panel
        for page in sorted(page_records, key=lambda item: item["page"])
        for panel in page["panels"]
    ]
    dialogue_mapping: dict[str, dict[str, Any]] = {}
    reading_order = 0
    for panel in ordered_panels:
        for bubble in panel.get("bubbles", []):
            reading_order += 1
            source_ids = [
                str(value)
                for value in bubble.get("source_segment_ids", [])
            ] or [str(bubble.get("segment_id", ""))]
            for sid in source_ids:
                if sid not in transcript:
                    continue
                entry = dialogue_mapping.setdefault(sid, {
                    "dialogue_id": sid,
                    "original_transcript": transcript[sid].get("text", ""),
                    "start": transcript[sid].get("start"),
                    "end": transcript[sid].get("end"),
                    "audio_speaker_id": bubble.get("speaker_id"),
                    "visual_character_id": bubble.get("character_id") or "unknown",
                    "speaker_match_score": bubble.get("speaker_match_score"),
                    "speaker_match_score_type": "uncalibrated lower-face motion heuristic",
                    "speaker_confidence": bubble.get("speaker_confidence"),
                    "speaker_attribution_decision": bubble.get("speaker_attribution_decision"),
                    "speaker_track_id": bubble.get("speaker_track_id"),
                    "associated_panel_id": panel.get("panel_id"),
                    "panel_ids": [],
                    "reading_order": reading_order,
                    "conversation_id": panel.get("scene_id"),
                    "dialogue_fragment": bool(bubble.get("dialogue_fragment")),
                })
                if panel.get("panel_id") not in entry["panel_ids"]:
                    entry["panel_ids"].append(panel.get("panel_id"))
                entry["dialogue_fragment"] = bool(
                    entry["dialogue_fragment"] or bubble.get("dialogue_fragment")
                )

    dialogue_mapping_path = output_dir / "dialogue_mapping.json"
    write_json(dialogue_mapping_path, {
        "video_id": video_id,
        "schema_version": 1,
        "score_note": "speaker_match_score is an uncalibrated heuristic; speaker_confidence is null unless supplied by a calibrated source.",
        "dialogues": list(dialogue_mapping.values()),
    })
    scene_summaries_path = output_dir / "scene_summaries.json"
    write_json(scene_summaries_path, {
        "video_id": video_id,
        "summary_policy": story_plan.get("summary_policy"),
        "scenes": story_plan.get("scenes", []),
    })
    story_validation = validate_story_output(
        story_plan,
        transcript,
        ordered_panels,
        frame_tolerance_s=float(config.get("story_frame_tolerance_s", 2.0)),
        video_duration_s=to_float(
            first_value(
                as_dict(read_json(resolve_path(Path("data/metadata") / f"{video_id}.json"))),
                ("duration_s", "duration"),
            )
        ),
        bubble_coverage_limit=float(config.get("bubble_layout_max_coverage_ratio", 0.42)),
    )
    story_validation_path = output_dir / "story_validation.json"
    write_json(story_validation_path, story_validation)
    page_records = paginate_story_panels(ordered_panels, config)

    page_images = build_pages(
        page_records,
        output_dir,
        config,
    )

    dialogue_alignments = []
    for page in page_records:
        for panel in page["panels"]:
            for bubble in panel["bubbles"]:
                warnings = []
                if not bubble.get("speaker_id"):
                    warnings.append("speaker_unidentified")
                elif not bubble.get("speaker_face_visible") and not bubble.get("off_panel_speaker"):
                    warnings.append("speaker_visual_match_unavailable")
                if bubble.get("alignment_method") == "temporal_scene_fallback":
                    warnings.append("scene_level_temporal_fallback")
                if (
                    bubble.get("alignment_confidence") is not None
                    and bubble["alignment_confidence"] < 0.5
                ):
                    warnings.append("low_alignment_confidence")
                if (
                    bubble.get("transcription_confidence") is not None
                    and bubble["transcription_confidence"] < 0.5
                ):
                    warnings.append("low_transcription_confidence")
                if (
                    bubble.get("transcription_confidence") is None
                    and len(str(bubble["text"]).split()) == 1
                ):
                    warnings.append("short_transcript_needs_review")

                character_id = bubble.get("character_id")
                dialogue_alignments.append({
                    "panel_id": panel["panel_id"],
                    "scene_id": first_value(
                        panel, ("scene_id", "shot_id")
                    ),
                    "timestamp": {
                        "start": bubble["start"],
                        "end": bubble["end"],
                    },
                    "source_frame": panel.get("frame_id"),
                    "characters": (
                        [{"character_id": character_id}]
                        if character_id
                        else []
                    ),
                    "dialogue": {
                        "segment_id": bubble["segment_id"],
                        "text": bubble["text"],
                        "speaker_id": bubble.get("speaker_id"),
                        "character_id": character_id,
                        "transcription_confidence": bubble.get(
                            "transcription_confidence"
                        ),
                    },
                    "alignment_method": bubble.get("alignment_method"),
                    "alignment_confidence": bubble.get(
                        "alignment_confidence"
                    ),
                    "warnings": warnings,
                })

    unplaced_segment_ids = sorted(
        set(transcript) - segment_seen - intentional_omissions
    )
    if unplaced_segment_ids:
        LOGGER.warning(
            "Phase 12 did not place transcript segments: %s",
            ", ".join(unplaced_segment_ids),
        )

    html_path = output_dir / "page.html"
    pdf_path = output_dir / "comic.pdf"
    cbz_path = output_dir / "comic.cbz"
    alignment_path = output_dir / "dialogue_frame_alignment.json"
    write_json(alignment_path, {
        "video_id": video_id,
        "phase": 12,
        "alignments": dialogue_alignments,
        "unplaced_segment_ids": unplaced_segment_ids,
    })

    if bool(config.get("export_html", True)):
        write_html(page_records, html_path)

    if bool(config.get("export_pdf", True)):
        export_pdf(page_images, pdf_path)

    if bool(config.get("export_cbz", True)):
        export_cbz(page_images, cbz_path)

    metrics = {
        "page_count": len(page_records),
        "panel_count": sum(
            len(page["panels"]) for page in page_records
        ),
        "visual_only_mode": visual_only_mode,
        "bubble_count": total_bubbles,
        "transcript_segment_count": len(transcript),
        "keyframe_record_count": len(keyframes),
        "keyframe_layout_matches": matched_frames,
        "segments_missing_from_transcript": unmatched_segments,
        "segments_placed": len(segment_seen),
        "segments_intentionally_omitted": sorted(intentional_omissions),
        "segments_not_placed": unplaced_segment_ids,
        "dialogue_segment_coverage": (
            round(
                len(segment_seen) / max(1, len(transcript) - len(intentional_omissions)),
                4,
            )
            if transcript
            else 1.0
        ),
        "panels_without_segments": panels_without_segments,
        "text_overflow_count": overflow_count,
        "attribution_count": attribution_count,
        "speaker_turn_count": len(speaker_turns),
        "story_validation_status": story_validation["status"],
        "story_validation_issue_count": len(story_validation["issues"]),
        "story_plan_available": bool(story_plan),
        "scene_summary_count": len(as_list(story_plan.get("scenes"))),
        "dialogue_mapping_count": len(dialogue_mapping),
        "emotion_segment_count": len(emotions),
        "note": (
            "Speaker attribution depends on Phase 7 turn matches. "
            "Review unknown or low-confidence speakers manually."
        ),
    }

    page_json = {
        "video_id": video_id,
        "phase": 12,
        "stage": "dialogue_aware_page_assembly_export",
        "pages": page_records,
        "metrics": metrics,
        "story_validation": story_validation,
        "source_artifacts": {
            key: str(value) for key, value in paths.items()
        },
        "exports": {
            "dialogue_frame_alignment": str(alignment_path),
            "story_validation": str(story_validation_path),
            "dialogue_mapping": str(dialogue_mapping_path),
            "scene_summaries": str(scene_summaries_path),
            "html": (
                str(html_path)
                if bool(config.get("export_html", True))
                else None
            ),
            "pdf": (
                str(pdf_path)
                if bool(config.get("export_pdf", True))
                else None
            ),
            "cbz": (
                str(cbz_path)
                if bool(config.get("export_cbz", True))
                else None
            ),
            "page_images": [str(path) for path in page_images],
        },
    }

    write_json(output_dir / "page.json", page_json)

    LOGGER.info(
        "Phase 12 keyframe join: %d layout panels, %d matched frame IDs",
        len(panels),
        matched_frames,
    )
    LOGGER.info(
        "Phase 12 completed: pages=%d panels=%d bubbles=%d "
        "panels_without_segments=%d",
        metrics["page_count"],
        metrics["panel_count"],
        total_bubbles,
        panels_without_segments,
    )

    return {
        "metrics": metrics,
        "exports": page_json["exports"],
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 12: dialogue-aware comic assembly and export"
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument(
        "--config",
        default="configs/phase12.yaml",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(
            logging, args.log_level.upper(), logging.INFO
        ),
        format=(
            "%(asctime)s | %(levelname)-7s | "
            "%(name)s | %(message)s"
        ),
    )

    config_path = resolve_path(args.config)
    result = run_phase12(args.video_id, config_path)
    metrics = result["metrics"]

    print(
        f"Phase 12 completed. Pages: {metrics['page_count']}; "
        f"panels: {metrics['panel_count']}; "
        f"bubbles: {metrics['bubble_count']}"
    )
    print(f"Output directory: {result['output_dir']}")

    for name, path in result["exports"].items():
        if path:
            print(f"{name.upper()}: {path}")


if __name__ == "__main__":
    main()
