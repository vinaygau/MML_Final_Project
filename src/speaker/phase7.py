"""
CineComic_MML - Phase 7
Speaker-to-character matching using:
    1. Diarization time intervals
    2. Face-track temporal overlap
    3. Face-track confidence
    4. Heuristic mouth-region motion
    5. Conservative speaker-to-character consistency

Run from the project root:
    python -m src.speaker.phase7 --config configs/phase7.yaml \
        --video-id a13eb5bedd6d

Important:
Mouth-region motion is only a heuristic. It is not lip-reading,
and match_rate is coverage, not verified identification accuracy.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


LOGGER = logging.getLogger("phase7")


# =========================================================
# 1. JSON AND CONFIGURATION UTILITIES
# =========================================================

def read_json(path: Path) -> Any:
    """Read a JSON file and return its decoded content."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json(path: Path, data: dict) -> None:
    """Write JSON safely by writing to a temporary file first."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    temporary_path.replace(path)


def expand_template(template: str, video_id: str) -> Path:
    """Expand a path template containing {video_id}."""
    return Path(str(template).format(video_id=video_id))


def first_value(
    item: dict,
    names: tuple[str, ...],
    default: Any = None,
) -> Any:
    """Return the first present, non-None value from a dictionary."""
    if not isinstance(item, dict):
        return default

    for name in names:
        if name in item and item[name] is not None:
            return item[name]

    return default


def as_float(value: Any, default: Any = None) -> Any:
    """Convert a value to a finite float."""
    try:
        number = float(value)
        if np.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass

    return default


def normalize_label(value: Any) -> str:
    """Normalize a label for case-insensitive comparisons."""
    if value is None:
        return ""

    return str(value).strip().lower()


def is_unknown(value: Any, unknown_label: str = "unknown") -> bool:
    """Check whether a value represents an unknown identity."""
    return normalize_label(value) in {
        "",
        "unknown",
        "none",
        "null",
        "n/a",
        "na",
        normalize_label(unknown_label),
    }


def valid_character_id(
    value: Any,
    unknown_label: str = "unknown",
) -> bool:
    """Return True if the value represents a usable character ID."""
    return not is_unknown(value, unknown_label)


def derive_character_id(track_id: Any) -> str:
    """Generate a stable fallback character ID from a track ID."""
    safe_track = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        str(track_id),
    )

    return f"CHAR_{safe_track}"


# =========================================================
# 2. DIARIZATION PARSING
# =========================================================

def normalize_speaker(value: Any) -> str:
    """Normalize missing or ambiguous speaker labels."""
    if value is None:
        return "unknown"

    text = str(value).strip()

    if text.lower() in {
        "",
        "none",
        "unknown",
        "null",
        "mixed",
        "overlap",
    }:
        return "unknown"

    return text


def parse_turns(data: dict) -> list[dict]:
    """
    Parse speaker turns from common diarization JSON structures.

    Supported top-level collections:
        turns
        speaker_turns
        diarization
        segments

    Also supports a separate segment_speakers collection as fallback.
    """
    raw_turns = first_value(
        data,
        ("turns", "speaker_turns", "diarization", "segments"),
        [],
    )

    if not isinstance(raw_turns, list):
        raw_turns = []

    turns = []

    for index, item in enumerate(raw_turns):
        if not isinstance(item, dict):
            continue

        start = as_float(
            first_value(item, ("start", "start_s", "start_time"))
        )
        end = as_float(
            first_value(item, ("end", "end_s", "end_time"))
        )

        speaker = normalize_speaker(
            first_value(
                item,
                (
                    "speaker",
                    "speaker_id",
                    "label",
                    "speaker_label",
                ),
            )
        )

        if start is None or end is None or end <= start:
            continue

        turns.append({
            "turn_id": str(
                first_value(
                    item,
                    ("turn_id", "segment_id", "id"),
                    index,
                )
            ),
            "start": float(start),
            "end": float(end),
            "speaker": speaker,
            "text": str(item.get("text", "")),
        })

    # Fallback for files that store timestamps in segments and speaker
    # labels in a separate segment_speakers array.
    if not turns:
        segments = data.get("segments", [])
        segment_speakers = data.get("segment_speakers", [])

        if isinstance(segments, list) and isinstance(
            segment_speakers, list
        ):
            speakers_by_id = {}

            for item in segment_speakers:
                if not isinstance(item, dict):
                    continue

                key = str(
                    first_value(item, ("segment_id", "id"), "")
                )

                speakers_by_id[key] = normalize_speaker(
                    first_value(
                        item,
                        ("speaker", "speaker_id", "label"),
                    )
                )

            for index, item in enumerate(segments):
                if not isinstance(item, dict):
                    continue

                start = as_float(
                    first_value(item, ("start", "start_s", "start_time"))
                )
                end = as_float(
                    first_value(item, ("end", "end_s", "end_time"))
                )

                if start is None or end is None or end <= start:
                    continue

                key = str(
                    first_value(
                        item,
                        ("segment_id", "id"),
                        index,
                    )
                )

                turns.append({
                    "turn_id": key,
                    "start": float(start),
                    "end": float(end),
                    "speaker": speakers_by_id.get(key, "unknown"),
                    "text": str(item.get("text", "")),
                })

    turns.sort(key=lambda turn: turn["start"])

    return turns


# =========================================================
# 3. FACE-TRACK PARSING
# =========================================================

def get_bbox(item: dict) -> list[float] | None:
    """
    Extract a bounding box.

    Supported forms:
        bbox: [x1, y1, x2, y2]
        bbox: {x1, y1, x2, y2}
        bbox: {x, y, w, h}
        x1/y1/x2/y2
        x/y/w/h
    """
    bbox = first_value(item, ("bbox", "box", "face_bbox"))

    if isinstance(bbox, dict):
        x1 = as_float(bbox.get("x1"))
        y1 = as_float(bbox.get("y1"))
        x2 = as_float(bbox.get("x2"))
        y2 = as_float(bbox.get("y2"))

        if None not in (x1, y1, x2, y2):
            return [x1, y1, x2, y2]

        x = as_float(bbox.get("x"))
        y = as_float(bbox.get("y"))
        w = as_float(first_value(bbox, ("w", "width")))
        h = as_float(first_value(bbox, ("h", "height")))

        if None not in (x, y, w, h):
            return [x, y, x + w, y + h]

    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        values = [as_float(value) for value in bbox[:4]]

        if all(value is not None for value in values):
            return values

    x1 = as_float(item.get("x1"))
    y1 = as_float(item.get("y1"))
    x2 = as_float(item.get("x2"))
    y2 = as_float(item.get("y2"))

    if None not in (x1, y1, x2, y2):
        return [x1, y1, x2, y2]

    x = as_float(item.get("x"))
    y = as_float(item.get("y"))
    w = as_float(first_value(item, ("w", "width")))
    h = as_float(first_value(item, ("h", "height")))

    if None not in (x, y, w, h):
        return [x, y, x + w, y + h]

    return None


def extract_track_objects(data: dict) -> list[dict]:
    """
    Convert face-track JSON into a normalized representation.

    Supported collections:
        {"tracks": [...]}
        {"tracks": {"track-id": {...}}}
        {"track_summaries": [...]}

    Each normalized track has:
        track_id, start, end, identity, track_confidence, detections
    """
    raw_tracks = data.get(
        "tracks",
        data.get("track_summaries", []),
    )

    if isinstance(raw_tracks, dict):
        entries = []

        for key, value in raw_tracks.items():
            if not isinstance(value, dict):
                continue

            track = dict(value)
            track.setdefault("track_id", key)
            entries.append(track)

        raw_tracks = entries

    if not isinstance(raw_tracks, list):
        raise ValueError(
            "Face-track JSON has no supported 'tracks' collection."
        )

    normalized_tracks = []

    for index, track in enumerate(raw_tracks):
        if not isinstance(track, dict):
            continue

        track_id = str(
            first_value(
                track,
                ("track_id", "id", "character_track_id"),
                f"T{index + 1:04d}",
            )
        )

        raw_detections = first_value(
            track,
            ("detections", "observations", "frames", "samples"),
            [],
        )

        if isinstance(raw_detections, dict):
            # Some pipelines store detections as a frame-keyed object.
            expanded = []

            for frame_key, value in raw_detections.items():
                if not isinstance(value, dict):
                    continue

                detection = dict(value)

                if "frame_index" not in detection:
                    try:
                        detection["frame_index"] = int(frame_key)
                    except (TypeError, ValueError):
                        pass

                expanded.append(detection)

            raw_detections = expanded

        if not isinstance(raw_detections, list):
            raw_detections = []

        # Support a track object that itself contains a single bbox.
        if not raw_detections and get_bbox(track) is not None:
            raw_detections = [track]

        detections = []

        for detection in raw_detections:
            if not isinstance(detection, dict):
                continue

            bbox = get_bbox(detection)

            if bbox is None:
                continue

            timestamp = as_float(
                first_value(
                    detection,
                    ("t", "time", "timestamp", "time_s", "t_sec"),
                )
            )

            frame_index = as_float(
                first_value(
                    detection,
                    ("frame_index", "frame_idx", "frame"),
                )
            )

            if timestamp is None:
                timestamp = as_float(
                    first_value(
                        track,
                        ("t", "time", "timestamp", "time_s"),
                    )
                )

            confidence = as_float(
                first_value(
                    detection,
                    ("det_score", "score", "confidence", "quality"),
                ),
                0.5,
            )

            detections.append({
                "t": timestamp,
                "frame_index": frame_index,
                "bbox": bbox,
                "landmarks": first_value(
                    detection,
                    ("landmarks", "landmark", "keypoints"),
                ),
                "confidence": max(0.0, min(1.0, confidence)),
            })

        track_start = as_float(
            first_value(track, ("start", "start_s", "start_time"))
        )
        track_end = as_float(
            first_value(track, ("end", "end_s", "end_time"))
        )

        # Infer temporal bounds from detection timestamps where possible.
        detection_times = [
            detection["t"]
            for detection in detections
            if detection["t"] is not None
        ]

        if track_start is None and detection_times:
            track_start = min(detection_times)

        if track_end is None and detection_times:
            track_end = max(detection_times)

        # Support track-level frame bounds as an additional fallback.
        if track_start is None:
            track_start = as_float(
                first_value(track, ("start_frame", "first_frame"))
            )

        if track_end is None:
            track_end = as_float(
                first_value(track, ("end_frame", "last_frame"))
            )

        track_confidence = as_float(
            first_value(
                track,
                (
                    "track_confidence",
                    "mean_quality",
                    "confidence",
                    "score",
                ),
            ),
            0.5,
        )

        identity = first_value(
            track,
            (
                "character_id",
                "identity_id",
                "identity",
                "person_id",
            ),
        )

        normalized_tracks.append({
            "track_id": track_id,
            "start": track_start,
            "end": track_end,
            "identity": identity,
            "track_confidence": max(
                0.0,
                min(1.0, float(track_confidence)),
            ),
            "detections": detections,
        })

    return normalized_tracks


# =========================================================
# 4. TEMPORAL OVERLAP
# =========================================================

def overlap_seconds(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> float:
    """Calculate intersection duration between two intervals."""
    return max(
        0.0,
        min(a_end, b_end) - max(a_start, b_start),
    )


def track_time_bounds(track: dict, fps: float) -> tuple[float, float] | None:
    """Get a track's time bounds, deriving them from detections if needed."""
    start = as_float(track.get("start"))
    end = as_float(track.get("end"))

    if start is not None and end is not None and end >= start:
        # If bounds appear to be frame indices rather than seconds,
        # convert only when there is supporting frame-index evidence.
        if (
            fps > 0
            and track.get("start_frame") is not None
        ):
            start /= fps
            end /= fps

        return float(start), float(end)

    times = []

    for detection in track.get("detections", []):
        timestamp = detection.get("t")

        if timestamp is None:
            frame_index = detection.get("frame_index")

            if frame_index is not None and fps > 0:
                timestamp = frame_index / fps

        if timestamp is not None:
            times.append(float(timestamp))

    if not times:
        return None

    return min(times), max(times)


def overlap_ratio(
    turn: dict,
    track: dict,
    fps: float = 0.0,
) -> float:
    """
    Measure overlap relative to the shorter interval.

    This is a temporal compatibility score, not a probability.
    """
    bounds = track_time_bounds(track, fps)

    if bounds is None:
        return 0.0

    track_start, track_end = bounds

    intersection = overlap_seconds(
        turn["start"],
        turn["end"],
        track_start,
        track_end,
    )

    turn_duration = max(1e-6, turn["end"] - turn["start"])
    track_duration = max(1e-6, track_end - track_start)

    return min(
        1.0,
        intersection / min(turn_duration, track_duration),
    )


# =========================================================
# 5. MOUTH-REGION MOTION
# =========================================================

def sample_times(
    start: float,
    end: float,
    sample_fps: float,
) -> list[float]:
    """Generate sample times inside a speaker turn."""
    duration = end - start

    if duration <= 0 or sample_fps <= 0:
        return []

    count = max(2, int(math.ceil(duration * sample_fps)) + 1)

    return np.linspace(start, end, count).tolist()


def _detection_time(detection: dict, fps: float) -> float | None:
    value = detection.get("t", detection.get("timestamp"))
    if value is None:
        frame_index = detection.get("frame_index")
        if frame_index is not None and fps > 0:
            value = float(frame_index) / fps
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def interpolate_track_bbox(
    detections: list[dict],
    timestamp: float,
    fps: float,
    max_gap_s: float = 1.5,
    nearest_tolerance_s: float = 0.12,
) -> list[float] | None:
    """Interpolate a tracked face box at a speech sample time.

    Phase 5 intentionally samples frames sparsely. Phase 7 can still inspect
    the mouth between those observations by interpolating the tracked box and
    reading the corresponding source-video frame. It never extrapolates past
    the track's observed bounds.
    """
    points = []
    for detection in detections:
        time_s = _detection_time(detection, fps)
        box = detection.get("bbox", detection.get("box"))
        if time_s is None or not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        try:
            points.append((time_s, [float(value) for value in box[:4]]))
        except (TypeError, ValueError):
            continue
    if not points:
        return None
    points.sort(key=lambda item: item[0])

    for time_s, box in points:
        if abs(time_s - timestamp) <= 1e-6:
            return box

    before = next((item for item in reversed(points) if item[0] < timestamp), None)
    after = next((item for item in points if item[0] > timestamp), None)
    if before and after:
        span = after[0] - before[0]
        if span <= 0 or span > max_gap_s:
            return None
        ratio = (timestamp - before[0]) / span
        return [
            left + (right - left) * ratio
            for left, right in zip(before[1], after[1])
        ]

    nearest = min(points, key=lambda item: abs(item[0] - timestamp))
    if abs(nearest[0] - timestamp) <= nearest_tolerance_s:
        return nearest[1]
    return None


def interpolate_track_landmarks(
    detections: list[dict],
    timestamp: float,
    fps: float,
    max_gap_s: float = 1.5,
    nearest_tolerance_s: float = 0.12,
) -> list[list[float]] | None:
    """Interpolate facial landmarks only between nearby tracked observations."""
    points = []
    for detection in detections:
        time_s = _detection_time(detection, fps)
        landmarks = detection.get("landmarks")
        if time_s is None or not isinstance(landmarks, (list, tuple)) or len(landmarks) < 5:
            continue
        try:
            coords = [[float(value) for value in point[:2]] for point in landmarks[:5]]
        except (TypeError, ValueError):
            continue
        if any(len(point) != 2 for point in coords):
            continue
        points.append((time_s, coords))
    if not points:
        return None
    points.sort(key=lambda item: item[0])

    for time_s, coords in points:
        if abs(time_s - timestamp) <= 1e-6:
            return coords
    before = next((item for item in reversed(points) if item[0] < timestamp), None)
    after = next((item for item in points if item[0] > timestamp), None)
    if before and after:
        span = after[0] - before[0]
        if span <= 0 or span > max_gap_s:
            return None
        ratio = (timestamp - before[0]) / span
        return [
            [
                left[axis] + (right[axis] - left[axis]) * ratio
                for axis in range(2)
            ]
            for left, right in zip(before[1], after[1])
        ]
    nearest = min(points, key=lambda item: abs(item[0] - timestamp))
    if abs(nearest[0] - timestamp) <= nearest_tolerance_s:
        return nearest[1]
    return None


def crop_mouth(
    frame: np.ndarray,
    bbox: list[float],
    mouth_fraction: float,
    landmarks: list[list[float]] | None = None,
) -> np.ndarray | None:
    """
    Crop the approximate lower-face region.

    This is a geometric approximation, not facial-landmark detection.
    """
    height, width = frame.shape[:2]

    x1, y1, x2, y2 = [
        int(round(value)) for value in bbox
    ]

    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))

    face_width = x2 - x1
    face_height = y2 - y1

    if face_width < 12 or face_height < 12:
        return None

    if landmarks is not None and len(landmarks) >= 5:
        mouth_corners = landmarks[3:5]
        mouth_x1 = min(point[0] for point in mouth_corners)
        mouth_x2 = max(point[0] for point in mouth_corners)
        mouth_y = sum(point[1] for point in mouth_corners) / 2.0
        pad_x = max(2, int(face_width * 0.12))
        half_height = max(3, int(face_height * 0.17))
        crop_x1 = max(0, min(width - 1, int(mouth_x1) - pad_x))
        crop_x2 = max(crop_x1 + 1, min(width, int(mouth_x2) + pad_x + 1))
        crop_y1 = max(0, min(height - 1, int(mouth_y) - half_height))
        crop_y2 = max(crop_y1 + 1, min(height, int(mouth_y) + half_height + 1))
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        # Keep this fallback useful for display, but the active-speaker phase
        # requires landmark-backed mouth crops before it can accept a match.
        fraction = max(0.1, min(0.9, mouth_fraction))
        mouth_y1 = y1 + int(face_height * (1.0 - fraction))
        crop = frame[mouth_y1:y2, x1:x2]

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(
        gray,
        (64, 32),
        interpolation=cv2.INTER_AREA,
    )

    return gray


def crop_upper_face_control(
    frame: np.ndarray,
    bbox: list[float],
) -> np.ndarray | None:
    """Crop an upper-face control region to discount head/camera movement."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    face_width, face_height = x2 - x1, y2 - y1
    if face_width < 12 or face_height < 16:
        return None
    control = frame[
        y1 + int(face_height * 0.22):y1 + int(face_height * 0.60),
        x1:x2,
    ]
    if control.size == 0:
        return None
    gray = cv2.cvtColor(control, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 32), interpolation=cv2.INTER_AREA)


def compute_mouth_motion(
    video_path: Path,
    track: dict,
    turn: dict,
    fps: float,
    sample_fps: float,
    minimum_samples: int,
    mouth_fraction: float,
    max_interpolation_gap_s: float = 1.5,
) -> dict:
    """
    Estimate lower-face visual change during a speaker turn.

    Higher values mean more frame-to-frame image change. They do not
    prove that a person is speaking.
    """
    detections = track.get("detections", [])

    if not detections:
        return {
            "motion_score": 0.0,
            "num_samples": 0,
            "available": False,
        }

    timed_detections = [
        (time_s, detection)
        for detection in detections
        if (time_s := _detection_time(detection, fps)) is not None
    ]
    if not timed_detections:
        return {
            "motion_score": 0.0,
            "control_motion_score": 0.0,
            "motion_contrast": 0.0,
            "num_samples": 0,
            "available": False,
        }

    first_time = min(time_s for time_s, _ in timed_detections)
    last_time = max(time_s for time_s, _ in timed_detections)
    sample_start = max(float(turn["start"]), first_time)
    sample_end = min(float(turn["end"]), last_time)
    if sample_end <= sample_start:
        return {
            "motion_score": 0.0,
            "control_motion_score": 0.0,
            "motion_contrast": 0.0,
            "num_samples": 0,
            "available": False,
        }

    sample_points = []
    for target_time in sample_times(sample_start, sample_end, sample_fps):
        bbox = interpolate_track_bbox(
            detections,
            target_time,
            fps,
            max_gap_s=max_interpolation_gap_s,
        )
        landmarks = interpolate_track_landmarks(
            detections,
            target_time,
            fps,
            max_gap_s=max_interpolation_gap_s,
        )
        if bbox is not None and landmarks is not None:
            sample_points.append((target_time, bbox, landmarks))
    if len(sample_points) < minimum_samples:
        return {
            "motion_score": 0.0,
            "control_motion_score": 0.0,
            "motion_contrast": 0.0,
            "num_samples": len(sample_points),
            "available": False,
        }

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    mouth_crops = []
    control_crops = []

    try:
        for timestamp, bbox, landmarks in sample_points:
            frame_number = max(
                0,
                int(round(timestamp * fps)),
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()

            if not ok or frame is None:
                continue

            crop = crop_mouth(
                frame,
                bbox,
                mouth_fraction,
                landmarks=landmarks,
            )
            control = crop_upper_face_control(frame, bbox)
            if crop is not None and control is not None:
                mouth_crops.append(crop)
                control_crops.append(control)
    finally:
        cap.release()

    if len(mouth_crops) < minimum_samples:
        return {
            "motion_score": 0.0,
            "control_motion_score": 0.0,
            "motion_contrast": 0.0,
            "num_samples": len(mouth_crops),
            "available": False,
        }

    mouth_differences = []
    control_differences = []

    for previous, current in zip(mouth_crops[:-1], mouth_crops[1:]):
        mouth_differences.append(float(np.mean(
            cv2.absdiff(previous, current)
        ) / 255.0))
    for previous, current in zip(control_crops[:-1], control_crops[1:]):
        control_differences.append(float(np.mean(
            cv2.absdiff(previous, current)
        ) / 255.0))

    if not mouth_differences or not control_differences:
        return {
            "motion_score": 0.0,
            "control_motion_score": 0.0,
            "motion_contrast": 0.0,
            "num_samples": len(mouth_crops),
            "available": False,
        }

    mouth_score = float(np.median(mouth_differences))
    control_score = float(np.median(control_differences))
    return {
        "motion_score": mouth_score,
        "control_motion_score": control_score,
        "motion_contrast": mouth_score - control_score,
        "num_samples": len(mouth_crops),
        "available": True,
    }


# =========================================================
# 6. SCORING AND EVIDENCE HELPERS
# =========================================================

def minmax_normalize(values: list[float]) -> list[float]:
    """Min-max normalize values to the range [0, 1]."""
    if not values:
        return []

    low = min(values)
    high = max(values)

    if high - low < 1e-8:
        return [0.0 for _ in values]

    return [
        (value - low) / (high - low)
        for value in values
    ]


def rank_active_speaker_candidates(
    candidates: list[dict],
    minimum_active_contrast: float,
    strong_active_contrast: float,
    minimum_match_score: float,
    minimum_candidate_margin: float,
) -> dict:
    """Rank face tracks only when mouth motion exceeds its face-motion control."""
    strong = max(minimum_active_contrast + 1e-6, strong_active_contrast)
    for candidate in candidates:
        contrast = float(candidate.get("motion_contrast", 0.0))
        candidate["active_speaker_evidence"] = bool(
            candidate.get("motion_available")
            and contrast >= minimum_active_contrast
        )
        activity = (
            max(0.0, min(1.0, contrast / strong))
            if candidate.get("motion_available")
            else 0.0
        )
        candidate["match_score"] = round(
            0.30 * float(candidate.get("temporal_overlap", 0.0))
            + 0.15 * float(candidate.get("track_confidence", 0.0))
            + 0.55 * activity,
            4,
        )

    candidates.sort(key=lambda item: item["match_score"], reverse=True)
    if not candidates:
        return {
            "candidates": [],
            "best": None,
            "score_margin": 0.0,
            "accepted": False,
            "decision": "no_candidate_track",
        }

    best = candidates[0]
    second_score = candidates[1]["match_score"] if len(candidates) > 1 else 0.0
    margin = float(best["match_score"] - second_score)
    active_candidates = [item for item in candidates if item["active_speaker_evidence"]]
    if not any(item.get("motion_available") for item in candidates):
        decision = "mouth_motion_unavailable"
    elif not active_candidates:
        decision = "no_active_speaker_evidence"
    elif (
        not best["active_speaker_evidence"]
        or best["match_score"] < minimum_match_score
        or (len(candidates) > 1 and margin < minimum_candidate_margin)
    ):
        decision = "low_confidence_or_ambiguous"
    else:
        decision = "matched"

    return {
        "candidates": candidates,
        "best": best,
        "score_margin": round(margin, 4),
        "accepted": decision == "matched",
        "decision": decision,
    }


def reliable_match(
    match: dict,
    min_score: float,
    unknown_label: str = "unknown",
) -> bool:
    """Check whether a match is suitable for speaker-level voting."""
    if match.get("decision") != "matched":
        return False

    character_id = match.get(
        "visual_character_id",
        match.get("character_id"),
    )

    if not valid_character_id(character_id, unknown_label):
        return False

    try:
        score = float(match.get("match_score", 0.0))
    except (TypeError, ValueError):
        return False

    return score >= min_score


def build_speaker_mapping(
    matches: list[dict],
    min_score: float,
    unknown_label: str,
    minimum_evidence_turns: int,
    minimum_dominance: float,
    minimum_relative_margin: float,
) -> tuple[dict, list[dict], dict]:
    """
    Build a conservative speaker-to-character mapping.

    Only accepted original visual matches contribute evidence.
    Corrections are applied later, so they cannot feed back into votes.
    """
    speaker_votes = defaultdict(lambda: defaultdict(float))
    speaker_turn_counts = defaultdict(int)

    for match in matches:
        speaker = str(match["speaker"])
        speaker_turn_counts[speaker] += 1

        if not reliable_match(match, min_score, unknown_label):
            continue

        character_id = str(match["visual_character_id"])
        score = float(match["match_score"])

        speaker_votes[speaker][character_id] += score

    accepted_mapping = {}
    mapping_records = []

    for speaker, votes in sorted(speaker_votes.items()):
        ranked_votes = sorted(
            votes.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        if not ranked_votes:
            continue

        best_character, best_weight = ranked_votes[0]

        second_weight = (
            ranked_votes[1][1]
            if len(ranked_votes) > 1
            else 0.0
        )

        total_weight = sum(votes.values())

        dominance = best_weight / max(total_weight, 1e-9)

        relative_margin = (
            (best_weight - second_weight)
            / max(best_weight, 1e-9)
        )

        evidence_turns = sum(
            1
            for match in matches
            if (
                match["speaker"] == speaker
                and str(match.get("visual_character_id", ""))
                == best_character
                and reliable_match(
                    match,
                    min_score,
                    unknown_label,
                )
            )
        )

        accepted = (
            evidence_turns >= minimum_evidence_turns
            and dominance >= minimum_dominance
            and relative_margin >= minimum_relative_margin
        )

        if accepted:
            accepted_mapping[speaker] = best_character

        mapping_records.append({
            "speaker_id": speaker,
            "character_id": best_character,
            "evidence_weight": round(float(best_weight), 4),
            "dominance": round(float(dominance), 4),
            "relative_margin": round(float(relative_margin), 4),
            "reliable_evidence_turns": evidence_turns,
            "mapping_accepted": accepted,
            "matched_turns": sum(
                1
                for match in matches
                if (
                    match["speaker"] == speaker
                    and str(match.get("visual_character_id", ""))
                    == best_character
                )
            ),
            "total_turns": speaker_turn_counts[speaker],
        })

    return accepted_mapping, mapping_records, speaker_turn_counts


# =========================================================
# 7. MAIN PHASE 7 MATCHING ENGINE
# =========================================================

def run_phase7(config_path: Path, video_id: str) -> dict:
    started = time.perf_counter()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8-sig") as file:
        config_data = yaml.safe_load(file) or {}

    if not isinstance(config_data, dict):
        raise ValueError("Phase 7 YAML must contain a mapping/object.")

    config = config_data.get("phase7", config_data)

    if not isinstance(config, dict):
        raise ValueError(
            "The 'phase7' configuration section must be a mapping."
        )

    required_templates = (
        "video_template",
        "diarization_template",
        "tracks_template",
        "output_template",
        "metrics_template",
    )

    missing_keys = [
        key for key in required_templates if key not in config
    ]

    if missing_keys:
        raise KeyError(
            "Missing Phase 7 configuration keys: "
            + ", ".join(missing_keys)
        )

    video_path = expand_template(
        config["video_template"],
        video_id,
    )
    diarization_path = expand_template(
        config["diarization_template"],
        video_id,
    )
    tracks_path = expand_template(
        config["tracks_template"],
        video_id,
    )
    output_path = expand_template(
        config["output_template"],
        video_id,
    )
    metrics_path = expand_template(
        config["metrics_template"],
        video_id,
    )

    metadata_template = config.get("metadata_template")
    metadata_path = (
        expand_template(metadata_template, video_id)
        if metadata_template
        else None
    )

    for path in (video_path, diarization_path, tracks_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input missing: {path}"
            )

    if output_path.exists() and not config.get("overwrite", True):
        LOGGER.info(
            "Output exists and overwrite=false: %s",
            output_path,
        )
        return read_json(output_path)

    diarization_data = read_json(diarization_path)
    tracks_data = read_json(tracks_path)

    metadata = {}

    if metadata_path is not None and metadata_path.exists():
        metadata = read_json(metadata_path)

    if not isinstance(metadata, dict):
        metadata = {}

    fps = as_float(
        first_value(metadata, ("fps", "frame_rate", "video_fps")),
        0.0,
    )

    if fps <= 0:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video to read FPS: {video_path}"
            )

        fps = as_float(
            cap.get(cv2.CAP_PROP_FPS),
            0.0,
        )

        cap.release()

    if fps <= 0:
        raise ValueError(
            "Could not determine video FPS. Add fps to the metadata "
            "JSON or check that the video can be opened."
        )

    turns = parse_turns(diarization_data)
    tracks = extract_track_objects(tracks_data)

    LOGGER.info("Video ID: %s", video_id)
    LOGGER.info("Speaker turns: %d", len(turns))
    LOGGER.info("Face tracks: %d", len(tracks))
    LOGGER.info("Video FPS: %.3f", fps)

    # -----------------------------------------------------
    # Matching configuration
    # -----------------------------------------------------

    min_overlap = float(
        config.get("minimum_turn_overlap", 0.30)
    )
    min_track_conf = float(
        config.get("minimum_track_confidence", 0.30)
    )
    min_score = float(
        config.get("minimum_match_score", 0.45)
    )
    sample_fps = float(
        config.get("sample_fps", 3.0)
    )
    minimum_samples = int(
        config.get("minimum_samples", 3)
    )
    mouth_fraction = float(
        config.get("mouth_region_fraction", 0.45)
    )
    max_interpolation_gap_s = float(
        config.get("max_track_interpolation_gap_s", 1.5)
    )
    minimum_active_contrast = float(
        config.get("minimum_active_speaker_contrast", 0.012)
    )
    strong_active_contrast = max(
        minimum_active_contrast + 1e-6,
        float(config.get("strong_active_speaker_contrast", 0.055)),
    )
    unknown_label = str(
        config.get("unknown_label", "unknown")
    )

    # Speaker mapping thresholds.
    minimum_evidence_turns = int(
        config.get("minimum_mapping_evidence_turns", 2)
    )
    minimum_dominance = float(
        config.get("minimum_mapping_dominance", 0.70)
    )
    minimum_relative_margin = float(
        config.get("minimum_mapping_relative_margin", 0.20)
    )
    strong_conflict_score = float(
        config.get("strong_conflict_score", 0.85)
    )
    minimum_candidate_margin = float(
        config.get("minimum_candidate_margin", 0.03)
    )

    matches = []
    motion_cache = {}

    # -----------------------------------------------------
    # Match each diarization turn to visual tracks
    # -----------------------------------------------------

    for turn_index, turn in enumerate(turns, start=1):
        LOGGER.info(
            "Matching turn %d/%d: %s (%.2f-%.2f s)",
            turn_index,
            len(turns),
            turn["speaker"],
            turn["start"],
            turn["end"],
        )

        if is_unknown(turn["speaker"], unknown_label):
            matches.append({
                **turn,
                "character_id": unknown_label,
                "visual_character_id": unknown_label,
                "final_character_id": unknown_label,
                "matched_track_id": None,
                "match_score": 0.0,
                "score_margin": 0.0,
                "decision": "unknown_speaker",
                "candidates": [],
            })
            continue

        candidates = []

        for track in tracks:
            track_conf = max(
                0.0,
                min(1.0, float(track["track_confidence"])),
            )

            if track_conf < min_track_conf:
                continue

            temporal = overlap_ratio(
                turn,
                track,
                fps,
            )

            if temporal < min_overlap:
                continue

            cache_key = (
                turn["turn_id"],
                track["track_id"],
            )

            if cache_key not in motion_cache:
                motion_cache[cache_key] = compute_mouth_motion(
                    video_path=video_path,
                    track=track,
                    turn=turn,
                    fps=fps,
                    sample_fps=sample_fps,
                    minimum_samples=minimum_samples,
                    mouth_fraction=mouth_fraction,
                    max_interpolation_gap_s=max_interpolation_gap_s,
                )

            motion_result = motion_cache[cache_key]

            candidates.append({
                "track_id": track["track_id"],
                "existing_character_id": track["identity"],
                "temporal_overlap": round(float(temporal), 4),
                "track_confidence": round(float(track_conf), 4),
                "motion_score": round(
                    float(motion_result["motion_score"]),
                    6,
                ),
                "control_motion_score": round(
                    float(motion_result.get("control_motion_score", 0.0)),
                    6,
                ),
                "motion_contrast": round(
                    float(motion_result.get("motion_contrast", 0.0)),
                    6,
                ),
                "motion_available": bool(
                    motion_result["available"]
                ),
                "motion_samples": int(
                    motion_result["num_samples"]
                ),
            })

        if not candidates:
            matches.append({
                **turn,
                "character_id": unknown_label,
                "visual_character_id": unknown_label,
                "final_character_id": unknown_label,
                "matched_track_id": None,
                "match_score": 0.0,
                "score_margin": 0.0,
                "decision": "no_candidate_track",
                "candidates": [],
            })
            continue

        # A visible face is not sufficient evidence for a speaking turn.
        ranking = rank_active_speaker_candidates(
            candidates,
            minimum_active_contrast=minimum_active_contrast,
            strong_active_contrast=strong_active_contrast,
            minimum_match_score=min_score,
            minimum_candidate_margin=minimum_candidate_margin,
        )
        candidates = ranking["candidates"]
        best = ranking["best"]
        margin = ranking["score_margin"]
        accepted = ranking["accepted"]
        decision = ranking["decision"]

        if accepted:
            visual_character = best["existing_character_id"]

            if not valid_character_id(
                visual_character,
                unknown_label,
            ):
                visual_character = derive_character_id(
                    best["track_id"]
                )

            matched_track_id = best["track_id"]
        else:
            visual_character = unknown_label
            matched_track_id = None

        matches.append({
            **turn,
            "character_id": visual_character,
            "visual_character_id": visual_character,
            "final_character_id": visual_character,
            "matched_track_id": matched_track_id,
            "match_score": best["match_score"],
            "score_margin": round(float(margin), 4),
            "decision": decision,
            "candidates": candidates,
        })

    # =====================================================
    # 8. SPEAKER-TO-CHARACTER CONSISTENCY
    # =====================================================

    (
        accepted_speaker_mapping,
        speaker_mapping,
        speaker_turn_counts,
    ) = build_speaker_mapping(
        matches=matches,
        min_score=min_score,
        unknown_label=unknown_label,
        minimum_evidence_turns=minimum_evidence_turns,
        minimum_dominance=minimum_dominance,
        minimum_relative_margin=minimum_relative_margin,
    )

    # Apply only accepted mappings. Keep the original visual assignment
    # in visual_character_id so the result remains auditable.
    for match in matches:
        speaker = str(match["speaker"])
        mapped_character = accepted_speaker_mapping.get(speaker)

        if mapped_character is None:
            continue

        visual_character = str(
            match.get("visual_character_id", unknown_label)
        )

        if visual_character == mapped_character:
            match["final_character_id"] = mapped_character
            match["character_id"] = mapped_character
            continue

        current_score = float(
            match.get("match_score", 0.0)
        )

        # Preserve strong contradictory visual evidence for review.
        if (
            valid_character_id(visual_character, unknown_label)
            and current_score >= strong_conflict_score
        ):
            match["identity_consistency_flag"] = (
                "strong_conflict_review"
            )
            match["suggested_character_id"] = mapped_character
            match["final_character_id"] = visual_character
            match["character_id"] = visual_character
            continue

        # Record the old assignment before applying the speaker mapping.
        match["raw_character_id"] = visual_character
        match["raw_matched_track_id"] = match.get("matched_track_id")

        match["character_id"] = mapped_character
        match["final_character_id"] = mapped_character
        match["decision"] = "speaker_mapping_corrected"
        match["identity_consistency_flag"] = (
            "corrected_from_repeated_evidence"
        )
        match["suggested_character_id"] = mapped_character

    # =====================================================
    # 9. METRICS
    # =====================================================

    total_turns = len(matches)

    matched_turns = sum(
        1
        for match in matches
        if valid_character_id(
            match.get("final_character_id"),
            unknown_label,
        )
    )

    corrected_turns = sum(
        1
        for match in matches
        if match.get("decision") == "speaker_mapping_corrected"
    )

    conflict_review_turns = sum(
        1
        for match in matches
        if (
            match.get("identity_consistency_flag")
            == "strong_conflict_review"
        )
    )

    accepted_mapping_count = sum(
        1
        for item in speaker_mapping
        if item.get("mapping_accepted")
    )

    unknown_turns = total_turns - matched_turns

    match_rate = (
        matched_turns / total_turns
        if total_turns
        else 0.0
    )

    runtime = time.perf_counter() - started

    # =====================================================
    # 10. OUTPUT JSON
    # =====================================================

    output = {
        "video_id": video_id,
        "phase": 7,
        "method": (
            "Temporal overlap + face-track confidence + "
            "interpolated facial-landmark mouth motion contrasted with upper-face control + "
            "conservative speaker consistency"
        ),
        "match_score_interpretation": "uncalibrated heuristic score; not a probability",
        "speaker_confidence_calibrated": False,
        "video_fps": round(float(fps), 4),
        "num_speaker_turns": total_turns,
        "num_face_tracks": len(tracks),
        "num_matched_turns": matched_turns,
        "num_unknown_turns": unknown_turns,
        "match_rate": round(match_rate, 4),
        "accepted_speaker_mappings": accepted_mapping_count,
        "corrected_turns": corrected_turns,
        "strong_conflict_review_turns": conflict_review_turns,
        "speaker_mapping": speaker_mapping,
        "turn_matches": matches,
        "runtime_seconds": round(runtime, 3),
        "ground_truth_evaluated": False,
        "limitations": [
            (
                "Mouth-region motion is a heuristic, "
                "not lip-reading."
            ),
            (
                "Head movement, camera movement, and lighting "
                "can affect motion scores."
            ),
            (
                "Speaker diarization errors can propagate "
                "into Phase 7."
            ),
            (
                "Manual ground-truth labels are required to "
                "measure actual identification accuracy."
            ),
            (
                "Speaker-level consistency may be wrong for "
                "dubbing, voice-over, or diarization errors."
            ),
            (
                "visual_character_id preserves the original "
                "visual-track assignment."
            ),
            (
                "final_character_id and character_id represent "
                "the final speaker-level assignment."
            ),
            (
                "raw_character_id and raw_matched_track_id preserve "
                "the original assignment when a correction is made."
            ),
            (
                "A corrected speaker identity does not mean the "
                "visual track itself has been relabeled."
            ),
        ],
    }

    # =====================================================
    # 11. METRICS JSON
    # =====================================================

    metrics = {
        "video_id": video_id,
        "phase": 7,
        "num_speaker_turns": total_turns,
        "num_face_tracks": len(tracks),
        "num_matched_turns": matched_turns,
        "num_unknown_turns": unknown_turns,
        "match_rate": round(match_rate, 4),
        "accepted_speaker_mappings": accepted_mapping_count,
        "corrected_turns": corrected_turns,
        "strong_conflict_review_turns": conflict_review_turns,
        "runtime_seconds": round(runtime, 3),
        "ground_truth_evaluated": False,
    }

    # =====================================================
    # 12. SAVE RESULTS
    # =====================================================

    write_json(output_path, output)
    write_json(metrics_path, metrics)

    LOGGER.info("Phase 7 complete.")
    LOGGER.info(
        "Matched turns: %d/%d",
        matched_turns,
        total_turns,
    )
    LOGGER.info(
        "Unknown turns: %d",
        unknown_turns,
    )
    LOGGER.info(
        "Match coverage: %.2f%%",
        match_rate * 100,
    )
    LOGGER.info(
        "Accepted speaker mappings: %d",
        accepted_mapping_count,
    )
    LOGGER.info(
        "Corrected turns: %d",
        corrected_turns,
    )
    LOGGER.info(
        "Conflicts requiring review: %d",
        conflict_review_turns,
    )
    LOGGER.info("Output: %s", output_path)
    LOGGER.info("Metrics: %s", metrics_path)
    LOGGER.info("Runtime: %.2f seconds", runtime)

    return output


# =========================================================
# 13. COMMAND-LINE ENTRY POINT
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 7: Active speaker to face-track matching"
        )
    )

    parser.add_argument(
        "--config",
        default="configs/phase7.yaml",
        help="Path to Phase 7 YAML configuration",
    )

    parser.add_argument(
        "--video-id",
        required=True,
        help="Video ID, e.g. a13eb5bedd6d",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        run_phase7(
            Path(args.config),
            args.video_id,
        )
    except Exception:
        LOGGER.exception("Phase 7 failed.")
        raise



if __name__ == "__main__":
    main()
