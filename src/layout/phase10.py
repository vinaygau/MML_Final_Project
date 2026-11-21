
"""
CineComic-MML — Phase 10
ROI Estimation, Page Layout, and Panel Cropping

Inputs:
    data/keyframes/{video_id}.json
    data/tracks/{video_id}.json
    data/shots/{video_id}.json
    data/frames/**/*.jpg  (or frame paths in the JSON artifacts)

Output:
    data/layout/{video_id}.json
    data/layout/{video_id}/panels/*.jpg
    data/layout/{video_id}/contact_sheet.jpg

The implementation:
    1. Loads selected keyframes and face tracks.
    2. Estimates regions of interest (ROIs).
    3. Selects panel aspect ratios and page templates.
    4. Searches for crops that retain important ROIs.
    5. Estimates bubble-safe zones.
    6. Saves crop offsets, scale factors, metrics, and panel images.

Coordinates in this module are pixel coordinates in the source frame.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_logger(video_id: str) -> logging.Logger:
    """Create a console and file logger."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"phase10_{video_id}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        file_handler = logging.FileHandler(
            log_dir / f"{video_id}_phase10.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    """Load JSON with a clear error message."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Save JSON, creating the output directory if necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def first_existing(paths: list[Path]) -> Path:
    """Return the first existing file from a list of candidates."""
    for path in paths:
        if path.is_file():
            return path

    candidates = "\n".join(str(path) for path in paths)
    raise FileNotFoundError(
        f"Could not locate an input JSON file. Checked:\n{candidates}"
    )


def find_frame_path(
    item: dict[str, Any],
    video_id: str,
    frame_id: str,
) -> Path | None:
    """Find a frame using a path stored in the keyframe record or its ID."""
    path_fields = (
        "image_path",
        "frame_path",
        "path",
        "file_path",
        "image",
    )

    for field in path_fields:
        value = item.get(field)

        if isinstance(value, str) and value.strip():
            candidate = Path(value)

            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate

            if candidate.is_file():
                return candidate

    search_roots = [
        PROJECT_ROOT / "data" / "frames",
        PROJECT_ROOT / "data" / "keyframes",
        PROJECT_ROOT / "data" / "shots",
    ]

    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    # Exact frame-ID matches are preferred.
    for root in search_roots:
        if not root.exists():
            continue

        for candidate in root.rglob("*"):
            if (
                candidate.is_file()
                and candidate.suffix.lower() in supported_extensions
                and candidate.stem == frame_id
            ):
                return candidate

    # Some pipelines save keyframes using a numeric filename.
    frame_number = item.get("frame_number", item.get("frame_index"))

    if frame_number is not None:
        for root in search_roots:
            if not root.exists():
                continue

            for candidate in root.rglob("*"):
                if (
                    candidate.is_file()
                    and candidate.suffix.lower() in supported_extensions
                    and candidate.stem == str(frame_number)
                ):
                    return candidate

    return None


def get_panels(data: Any) -> list[dict[str, Any]]:
    """Extract panel/keyframe records from common JSON structures."""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if not isinstance(data, dict):
        raise ValueError("Keyframe JSON must contain an object or list.")

    if isinstance(data.get("panels"), list):
        return data["panels"]

    # Support page-based layouts from an earlier pipeline.
    pages = data.get("pages", [])

    if isinstance(pages, list):
        panels = []

        for page in pages:
            if isinstance(page, dict):
                panels.extend(
                    x for x in page.get("panels", [])
                    if isinstance(x, dict)
                )

        if panels:
            return panels

    for key in ("keyframes", "selected_frames", "frames", "results"):
        value = data.get(key)

        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]

    raise ValueError(
        "No panels/keyframes list found in the keyframe JSON. "
        "Expected 'panels', 'keyframes', 'selected_frames', or 'frames'."
    )


def get_box(value: Any) -> tuple[float, float, float, float] | None:
    """
    Convert a box into (x1, y1, x2, y2).

    Supports:
        [x1, y1, x2, y2]
        [x, y, width, height]
        {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
        {"x": ..., "y": ..., "w": ..., "h": ...}
        {"bbox": [...]}
    """
    if isinstance(value, dict):
        if "bbox" in value:
            return get_box(value["bbox"])

        if all(k in value for k in ("x1", "y1", "x2", "y2")):
            return (
                float(value["x1"]),
                float(value["y1"]),
                float(value["x2"]),
                float(value["y2"]),
            )

        if all(k in value for k in ("x", "y", "w", "h")):
            x, y, w, h = (
                float(value["x"]),
                float(value["y"]),
                float(value["w"]),
                float(value["h"]),
            )
            return x, y, x + w, y + h

        return None

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None

    x1, y1, a, b = map(float, value)

    # Explicit format labels remove ambiguity.
    return x1, y1, a, b


def box_to_xyxy(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    fmt: str = "xyxy",
) -> tuple[float, float, float, float] | None:
    """Normalize a box to clipped xyxy coordinates."""
    x1, y1, a, b = box

    if fmt == "xywh":
        x2, y2 = x1 + a, y1 + b
    else:
        x2, y2 = a, b

    x1 = float(np.clip(x1, 0, width))
    y1 = float(np.clip(y1, 0, height))
    x2 = float(np.clip(x2, 0, width))
    y2 = float(np.clip(y2, 0, height))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def parse_box_field(
    value: Any,
    width: int,
    height: int,
    fmt: str = "xyxy",
) -> tuple[float, float, float, float] | None:
    box = get_box(value)

    if box is None:
        return None

    return box_to_xyxy(box, width, height, fmt)


def extract_track_records(data: Any) -> list[dict[str, Any]]:
    """Extract face-track records from Phase 5 output."""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if not isinstance(data, dict):
        return []

    for key in ("tracks", "faces", "detections"):
        records = data.get(key)

        if isinstance(records, list):
            return [x for x in records if isinstance(x, dict)]

    return []


def extract_faces_for_frame(
    tracks_data: Any,
    frame_id: str,
    timestamp: float | None,
    width: int,
    height: int,
    time_tolerance: float = 0.75,
) -> list[dict[str, Any]]:
    """
    Extract face boxes from track artifacts.

    Supports tracks containing frame records, and individual face detections.
    """
    records = extract_track_records(tracks_data)
    found: list[dict[str, Any]] = []

    for record in records:
        character_id = record.get(
            "character_id",
            record.get("character", "UNKNOWN"),
        )
        track_id = record.get("track_id")

        # Phase 5's track artifact stores sampled face observations under
        # `tracks[].detections`; handle those explicitly so Phase 10 can keep
        # lettering away from faces instead of falling back to saliency only.
        detections = record.get("detections")
        if isinstance(detections, list):
            for detection in detections:
                if not isinstance(detection, dict):
                    continue
                detection_time = detection.get(
                    "t", detection.get("timestamp", detection.get("time"))
                )
                if timestamp is None or not isinstance(detection_time, (int, float)):
                    continue
                if abs(float(detection_time) - timestamp) > time_tolerance:
                    continue
                box = parse_box_field(
                    detection.get("box", detection.get("bbox", detection.get("face_box"))),
                    width,
                    height,
                    str(detection.get("box_format", "xyxy")).lower(),
                )
                if box is not None:
                    mouth = detection.get("mouth_center")
                    if not (isinstance(mouth, (list, tuple)) and len(mouth) >= 2):
                        landmarks = detection.get("landmarks")
                        if isinstance(landmarks, (list, tuple)) and len(landmarks) >= 5:
                            mouth = [
                                (float(landmarks[3][0]) + float(landmarks[4][0])) / 2,
                                (float(landmarks[3][1]) + float(landmarks[4][1])) / 2,
                            ]
                    orientation = detection.get("face_orientation", detection.get("orientation"))
                    landmarks = detection.get("landmarks")
                    if orientation is None and isinstance(landmarks, (list, tuple)) and len(landmarks) >= 3:
                        try:
                            eye_mid_x = (float(landmarks[0][0]) + float(landmarks[1][0])) / 2
                            nose_offset = (float(landmarks[2][0]) - eye_mid_x) / max(1.0, box[2] - box[0])
                            orientation = "frontal" if abs(nose_offset) < 0.10 else (
                                "turned_right" if nose_offset > 0 else "turned_left"
                            )
                        except (TypeError, ValueError, IndexError):
                            orientation = None
                    found.append({
                        "box": box,
                        "character_id": detection.get("character_id", character_id),
                        "track_id": detection.get("track_id", track_id),
                        "mouth_center": [float(mouth[0]), float(mouth[1])]
                        if isinstance(mouth, (list, tuple)) and len(mouth) >= 2 else None,
                        "det_score": detection.get("det_score", detection.get("confidence")),
                        "face_orientation": orientation,
                        "source": "track_detection",
                    })
            continue

        frames = record.get("frames")

        if isinstance(frames, list):
            for frame in frames:
                if not isinstance(frame, dict):
                    continue

                same_frame = frame.get("frame_id") == frame_id

                frame_time = frame.get(
                    "t",
                    frame.get("timestamp", frame.get("time")),
                )

                same_time = (
                    timestamp is not None
                    and isinstance(frame_time, (int, float))
                    and abs(float(frame_time) - timestamp) <= time_tolerance
                )

                if not same_frame and not same_time:
                    continue

                raw_box = frame.get(
                    "box",
                    frame.get("bbox", frame.get("face_box")),
                )

                box = parse_box_field(
                    raw_box, width, height,
                    str(frame.get("box_format", "xyxy")).lower(),
                )

                if box is not None:
                    found.append({
                        "box": box,
                        "character_id": character_id,
                        "track_id": track_id,
                        "source": "track",
                    })

        else:
            frame_matches = (
                record.get("frame_id") == frame_id
                or (
                    timestamp is not None
                    and isinstance(record.get("t"), (int, float))
                    and abs(float(record["t"]) - timestamp)
                    <= time_tolerance
                )
            )

            if frame_matches:
                raw_box = record.get(
                    "box",
                    record.get("bbox", record.get("face_box")),
                )

                box = parse_box_field(
                    raw_box, width, height,
                    str(record.get("box_format", "xyxy")).lower(),
                )

                if box is not None:
                    found.append({
                        "box": box,
                        "character_id": character_id,
                        "track_id": track_id,
                        "source": "detection",
                    })

    return found


def extract_inline_faces(
    item: dict[str, Any],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Read optional face boxes directly from a keyframe record."""
    result = []

    values = item.get("faces", item.get("face_boxes", []))

    if isinstance(values, dict):
        values = [values]

    if not isinstance(values, list):
        return result

    for face in values:
        if isinstance(face, dict):
            raw_box = face.get(
                "box",
                face.get("bbox", face.get("face_box")),
            )
            character = face.get("character_id", "UNKNOWN")
            track_id = face.get("track_id")
            fmt = str(face.get("box_format", "xyxy")).lower()
        else:
            raw_box = face
            character = "UNKNOWN"
            track_id = None
            fmt = "xyxy"

        box = parse_box_field(raw_box, width, height, fmt)

        if box is not None:
            result.append({
                "box": box,
                "character_id": character,
                "track_id": track_id,
                "source": "keyframe",
            })

    return result


def get_saliency_map(frame: np.ndarray) -> np.ndarray:
    """
    Approximate visual saliency with OpenCV spectral-residual saliency.

    If the contrib saliency module is unavailable, use a gradient-based
    fallback. The result is normalized to [0, 1].
    """
    height, width = frame.shape[:2]

    try:
        if hasattr(cv2, "saliency"):
            detector = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency = detector.computeSaliency(frame)

            if success and saliency is not None:
                saliency = np.asarray(saliency, dtype=np.float32)
                return cv2.normalize(
                    saliency, None, 0.0, 1.0, cv2.NORM_MINMAX
                )
    except Exception:
        pass

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    # Local contrast gives a lightweight fallback saliency estimate.
    local_mean = cv2.blur(gray.astype(np.float32), (21, 21))
    contrast = np.abs(gray.astype(np.float32) - local_mean)

    saliency = 0.65 * gradient + 0.35 * contrast
    saliency = cv2.GaussianBlur(saliency, (0, 0), 3)

    return cv2.normalize(
        saliency, None, 0.0, 1.0, cv2.NORM_MINMAX
    )


def normalize_roi(
    face_records: list[dict[str, Any]],
    saliency: np.ndarray,
    width: int,
    height: int,
    speaker_character_ids: set[str] | None,
    speaker_track_ids: set[str] | None,
) -> list[dict[str, Any]]:
    """
    Combine face boxes and salient regions.

    Face boxes receive higher priority when Phase 7 identifies them as a
    speaking character in the selected panel.
    """
    rois = []

    for face in face_records:
        x1, y1, x2, y2 = face["box"]
        character = face.get("character_id", "UNKNOWN")

        character_key = str(character)
        track_key = str(face.get("track_id"))
        is_speaker = (
            character_key in (speaker_character_ids or set())
            or track_key in (speaker_track_ids or set())
        )

        rois.append({
            "box": [x1, y1, x2, y2],
            "weight": 6.0 if is_speaker else 3.0,
            "kind": "speaker_face" if is_speaker else "face",
            "character_id": character,
            "track_id": face.get("track_id"),
            "mouth_center": face.get("mouth_center"),
            "det_score": face.get("det_score"),
            "face_orientation": face.get("face_orientation"),
        })

    # Divide the image into cells and retain cells with strong saliency.
    grid_rows, grid_cols = 6, 8
    cell_h = max(1, height // grid_rows)
    cell_w = max(1, width // grid_cols)

    cell_scores = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = min(width, x1 + cell_w)
            y2 = min(height, y1 + cell_h)

            region = saliency[y1:y2, x1:x2]

            if region.size:
                cell_scores.append((
                    float(np.mean(region)),
                    [x1, y1, x2, y2],
                ))

    cell_scores.sort(key=lambda x: x[0], reverse=True)

    # Use a few high-saliency regions rather than letting all cells dominate.
    for score, box in cell_scores[:3]:
        if score > 0.15:
            rois.append({
                "box": box,
                "weight": max(0.25, score),
                "kind": "salient_region",
                "character_id": None,
                "track_id": None,
            })

    # A frame can have no detected face and very weak saliency.
    # Keep a low-weight center ROI as a fallback.
    if not rois:
        rois.append({
            "box": [
                width * 0.25,
                height * 0.20,
                width * 0.75,
                height * 0.80,
            ],
            "weight": 1.0,
            "kind": "fallback_center",
            "character_id": None,
            "track_id": None,
        })

    return rois


def crop_roi_retention(
    crop: tuple[int, int, int, int],
    rois: list[dict[str, Any]],
) -> float:
    """Weighted fraction of ROI area retained by a crop."""
    x1, y1, x2, y2 = crop
    total_weighted_area = 0.0
    retained_weighted_area = 0.0

    for roi in rois:
        rx1, ry1, rx2, ry2 = roi["box"]
        weight = max(0.0, float(roi.get("weight", 1.0)))

        area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)

        if area <= 0:
            continue

        ix1 = max(x1, rx1)
        iy1 = max(y1, ry1)
        ix2 = min(x2, rx2)
        iy2 = min(y2, ry2)

        intersection = (
            max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        )

        total_weighted_area += weight * area
        retained_weighted_area += weight * intersection

    if total_weighted_area <= 0:
        return 1.0

    return float(
        np.clip(
            retained_weighted_area / total_weighted_area,
            0.0,
            1.0,
        )
    )


def roi_center_penalty(
    crop: tuple[int, int, int, int],
    rois: list[dict[str, Any]],
) -> float:
    """Penalize crops that exclude ROI centers."""
    x1, y1, x2, y2 = crop
    penalty = 0.0
    total_weight = 0.0

    for roi in rois:
        rx1, ry1, rx2, ry2 = roi["box"]
        cx = (rx1 + rx2) / 2
        cy = (ry1 + ry2) / 2
        weight = max(0.0, float(roi.get("weight", 1.0)))

        total_weight += weight

        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            penalty += weight

    if total_weight <= 0:
        return 0.0

    return penalty / total_weight


def adaptive_speaker_aspect_ratio(
    speaker_faces: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    default_ratio: float,
) -> tuple[float, str, list[float] | None]:
    """Choose a crop aspect that can contain the full active speaker group."""
    if not speaker_faces:
        return default_ratio, "visual_fallback", None
    if len(speaker_faces) == 1:
        return default_ratio, "single_speaker", list(speaker_faces[0]["box"])

    x1 = min(float(roi["box"][0]) for roi in speaker_faces)
    y1 = min(float(roi["box"][1]) for roi in speaker_faces)
    x2 = max(float(roi["box"][2]) for roi in speaker_faces)
    y2 = max(float(roi["box"][3]) for roi in speaker_faces)
    group_w, group_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    # Room around the group keeps heads/mouths inside the art and leaves
    # safe space for bubbles. It is based on face-group geometry, not image
    # resizing, so unrelated close-ups keep their original framing.
    # Conversation panels reserve lateral room for a balloon beside a face.
    # This prevents a right or left speaker from being pinned to a panel edge
    # where the renderer would otherwise need a long leader tail.
    required_w = group_w + max(group_w * 0.48, group_h * 0.90)
    required_h = group_h + max(group_h * 0.28, group_w * 0.12)
    lower_ratio = required_w / max(1.0, frame_height)
    upper_ratio = frame_width / max(1.0, required_h)
    if lower_ratio <= upper_ratio:
        ratio = min(upper_ratio, max(default_ratio, lower_ratio))
        mode = "two_speaker_medium" if len(speaker_faces) == 2 else "group_wide"
    else:
        # The desired breathing room exceeds the source image. Use the whole
        # original frame as a safe fallback: no speaking face is cropped.
        ratio = frame_width / max(1, frame_height)
        mode = "group_full_frame_fallback"
    return ratio, mode, [x1, y1, x2, y2]


def find_best_crop(
    frame: np.ndarray,
    rois: list[dict[str, Any]],
    aspect_ratio: float,
    saliency: np.ndarray,
    grid_steps: int = 12,
    required_rois: list[dict[str, Any]] | None = None,
    minimum_required_retention: float = 0.995,
) -> dict[str, Any]:
    """
    Slide a crop window over the image and select the highest-scoring crop.

    aspect_ratio = crop_width / crop_height
    """
    height, width = frame.shape[:2]

    if aspect_ratio <= 0:
        raise ValueError("Panel aspect ratio must be positive.")

    # Largest crop with the requested aspect ratio that fits the frame.
    if width / height >= aspect_ratio:
        crop_h = height
        crop_w = min(width, int(round(height * aspect_ratio)))
    else:
        crop_w = width
        crop_h = min(height, int(round(width / aspect_ratio)))

    crop_w = max(1, crop_w)
    crop_h = max(1, crop_h)

    max_x = width - crop_w
    max_y = height - crop_h

    xs = sorted(set(
        int(round(x))
        for x in np.linspace(0, max_x, max(1, grid_steps))
    ))
    ys = sorted(set(
        int(round(y))
        for y in np.linspace(0, max_y, max(1, grid_steps))
    ))

    # Include centered and ROI-centered candidates.
    xs.append(max_x // 2)
    ys.append(max_y // 2)

    for roi in rois:
        rx1, ry1, rx2, ry2 = roi["box"]
        cx = (rx1 + rx2) / 2
        cy = (ry1 + ry2) / 2

        xs.append(int(np.clip(round(cx - crop_w / 2), 0, max_x)))
        ys.append(int(np.clip(round(cy - crop_h / 2), 0, max_y)))

    if required_rois:
        gx1 = min(float(roi["box"][0]) for roi in required_rois)
        gy1 = min(float(roi["box"][1]) for roi in required_rois)
        gx2 = max(float(roi["box"][2]) for roi in required_rois)
        gy2 = max(float(roi["box"][3]) for roi in required_rois)
        xs.append(int(np.clip(round((gx1 + gx2 - crop_w) / 2), 0, max_x)))
        ys.append(int(np.clip(round((gy1 + gy2 - crop_h) / 2), 0, max_y)))
        if len(required_rois) == 1:
            # Bias a one-speaker close-up slightly to either side. This
            # preserves the face while providing an empty half-panel for a
            # readable balloon and a short, natural tail.
            rx1, ry1, rx2, ry2 = required_rois[0]["box"]
            face_center_x = (float(rx1) + float(rx2)) / 2
            for panel_fraction in (0.30, 0.70):
                xs.append(int(np.clip(
                    round(face_center_x - crop_w * panel_fraction), 0, max_x
                )))

    face_rois = [roi for roi in rois if roi.get("kind") in ("face", "speaker_face")]
    if len(face_rois) > 1:
        # Add a crop centered on the full conversation group so separated
        # speakers are considered together, rather than one face at a time.
        fx1 = min(float(roi["box"][0]) for roi in face_rois)
        fy1 = min(float(roi["box"][1]) for roi in face_rois)
        fx2 = max(float(roi["box"][2]) for roi in face_rois)
        fy2 = max(float(roi["box"][3]) for roi in face_rois)
        xs.append(int(np.clip(round((fx1 + fx2 - crop_w) / 2), 0, max_x)))
        ys.append(int(np.clip(round((fy1 + fy2 - crop_h) / 2), 0, max_y)))

    xs = sorted(set(xs))
    ys = sorted(set(ys))

    best = None
    best_score = -float("inf")

    for x in xs:
        for y in ys:
            crop = (x, y, x + crop_w, y + crop_h)

            retention = crop_roi_retention(crop, rois)
            speaker_retention = crop_roi_retention(crop, required_rois or [])
            center_penalty = roi_center_penalty(crop, rois)

            saliency_region = saliency[
                y:y + crop_h, x:x + crop_w
            ]

            saliency_score = (
                float(np.mean(saliency_region))
                if saliency_region.size
                else 0.0
            )

            # Favor ROI retention first; saliency breaks close ties.
            required_bonus = (
                100.0 if speaker_retention >= minimum_required_retention else
                -100.0 * (minimum_required_retention - speaker_retention)
            ) if required_rois else 0.0
            bubble_clearance = 0.0
            if required_rois and len(required_rois) == 1:
                rx1, _, rx2, _ = required_rois[0]["box"]
                bubble_clearance = max(
                    max(0.0, float(rx1) - x),
                    max(0.0, x + crop_w - float(rx2)),
                ) / max(1.0, crop_w)
            score = (
                required_bonus
                + 25.0 * speaker_retention
                + 0.80 * retention
                + 0.15 * saliency_score
                + 0.05 * (1.0 - center_penalty)
                + 0.45 * bubble_clearance
            )

            if score > best_score:
                best_score = score
                best = {
                    "crop": list(crop),
                    "roi_retention": retention,
                    "speaker_roi_retention": speaker_retention,
                    "saliency_score": saliency_score,
                    "center_penalty": center_penalty,
                    "crop_score": score,
                    "bubble_clearance": bubble_clearance,
                }

    if best is None:
        raise RuntimeError("Could not find a valid panel crop.")

    return best


def find_bubble_safe_zone(
    crop: tuple[int, int, int, int],
    rois: list[dict[str, Any]],
    saliency: np.ndarray,
    grid_rows: int = 4,
    grid_cols: int = 4,
) -> list[int]:
    """
    Select a low-ROI, low-saliency rectangle inside the crop.

    Returns [x1, y1, x2, y2] in ORIGINAL frame coordinates.
    """
    x1, y1, x2, y2 = crop
    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w <= 0 or crop_h <= 0:
        return [x1, y1, x2, y2]

    # Prefer a zone roughly 35% of the panel width and 22% of its height.
    zone_w = max(1, int(crop_w * 0.35))
    zone_h = max(1, int(crop_h * 0.22))

    best_zone = None
    best_score = float("inf")

    x_positions = np.linspace(
        x1, max(x1, x2 - zone_w), max(2, grid_cols)
    )
    y_positions = np.linspace(
        y1, max(y1, y2 - zone_h), max(2, grid_rows)
    )

    for zx in x_positions:
        for zy in y_positions:
            bx1 = int(round(zx))
            by1 = int(round(zy))
            bx2 = min(x2, bx1 + zone_w)
            by2 = min(y2, by1 + zone_h)

            saliency_region = saliency[by1:by2, bx1:bx2]
            saliency_score = (
                float(np.mean(saliency_region))
                if saliency_region.size
                else 1.0
            )

            overlap = 0.0

            for roi in rois:
                rx1, ry1, rx2, ry2 = roi["box"]

                ix1 = max(bx1, rx1)
                iy1 = max(by1, ry1)
                ix2 = min(bx2, rx2)
                iy2 = min(by2, ry2)

                intersection = (
                    max(0, ix2 - ix1) * max(0, iy2 - iy1)
                )

                zone_area = max(1, (bx2 - bx1) * (by2 - by1))
                overlap += (
                    intersection / zone_area
                ) * float(roi.get("weight", 1.0))

            score = saliency_score + 0.5 * overlap

            if score < best_score:
                best_score = score
                best_zone = [bx1, by1, bx2, by2]

    return best_zone or [x1, y1, x2, y2]


def choose_template(panel_count: int) -> tuple[str, list[float]]:
    """
    Choose a simple page template.

    Returns the template name and panel aspect ratios (width / height).
    These are layout templates, not learned predictions.
    """
    if panel_count <= 1:
        return "T1", [1.0]

    if panel_count == 2:
        return "T2", [1.35, 1.35]

    if panel_count == 3:
        return "T3A", [1.7, 1.05, 1.05]

    if panel_count == 4:
        return "T4", [1.35, 1.35, 1.35, 1.35]

    if panel_count == 5:
        return "T5", [1.6, 1.05, 1.05, 1.05, 1.05]

    return "T6", [1.35] * panel_count


def load_config(config_path: Path | None) -> dict[str, Any]:
    """Load an optional YAML config; defaults are used if PyYAML is absent."""
    defaults = {
        "data_dir": "data",
        "output_dir": "data/layout",
        "panels_per_page": 4,
        "panel_aspect_ratio": 1.35,
        "grid_steps": 12,
        "speaker_weight": 5.0,
        "speaker_group_min_retention": 0.995,
        "time_tolerance_s": 0.75,
        "target_page_width": 1200,
        "target_panel_height": 360,
        "save_panel_images": True,
        "save_contact_sheet": True,
    }

    if config_path is None:
        return defaults

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read a YAML config. "
            "Install it with: python -m pip install pyyaml"
        ) from exc

    with config_path.open("r", encoding="utf-8") as file:
        user_config = yaml.safe_load(file) or {}

    if not isinstance(user_config, dict):
        raise ValueError("Phase 10 YAML must contain a mapping.")

    # Permit either a phase10: {...} section or root-level settings.
    if isinstance(user_config.get("phase10"), dict):
        user_config = {**user_config, **user_config["phase10"]}

    return {**defaults, **user_config}


def load_input_artifacts(
    video_id: str,
    data_dir: Path,
) -> tuple[Any, Any, Path]:
    """Locate the keyframe and track artifacts."""
    keyframes_path = first_existing([
        data_dir / "keyframes" / f"{video_id}.json",
        data_dir / "keyframes" / f"keyframes_{video_id}.json",
        data_dir / "keyframes" / video_id / "keyframes.json",
    ])

    tracks_candidates = [
        data_dir / "tracks" / f"{video_id}.json",
        data_dir / "tracks" / f"tracks_{video_id}.json",
        data_dir / "faces" / f"{video_id}.json",
    ]

    tracks_path = next(
        (p for p in tracks_candidates if p.is_file()),
        None,
    )

    keyframes_data = load_json(keyframes_path)
    tracks_data = load_json(tracks_path) if tracks_path else {}

    return keyframes_data, tracks_data, keyframes_path


def load_dialogue_speaker_context(
    video_id: str,
    data_dir: Path,
) -> tuple[dict[str, tuple[float, float]], list[dict[str, Any]]]:
    """Load the existing transcript and Phase 7 turn matches when present."""
    segment_windows: dict[str, tuple[float, float]] = {}
    subtitle_path = data_dir / "subtitles" / f"{video_id}.json"
    speaker_path = data_dir / "speaker" / f"{video_id}.json"

    if subtitle_path.is_file():
        subtitles = load_json(subtitle_path)
        rows = subtitles.get("segments", []) if isinstance(subtitles, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                segment_windows[str(row.get("segment_id"))] = (
                    float(row["start"]), float(row["end"])
                )
            except (KeyError, TypeError, ValueError):
                continue

    turns: list[dict[str, Any]] = []
    if speaker_path.is_file():
        speaker_data = load_json(speaker_path)
        rows = speaker_data.get("turn_matches", []) if isinstance(speaker_data, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                start, end = float(row["start"]), float(row["end"])
            except (KeyError, TypeError, ValueError):
                continue
            character_id = row.get(
                "final_character_id",
                row.get("visual_character_id", row.get("character_id")),
            )
            if str(character_id).casefold() in {"", "none", "unknown"}:
                character_id = None
            track_id = row.get("matched_track_id", row.get("track_id"))
            turns.append({
                "start": start,
                "end": end,
                "speaker_id": row.get("speaker", row.get("speaker_id")),
                "character_id": character_id,
                "track_id": track_id,
            })
    return segment_windows, turns


def panel_speaker_targets(
    panel: dict[str, Any],
    segment_windows: dict[str, tuple[float, float]],
    speaker_turns: list[dict[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    """Return Phase 7 speaker identities active in a Phase 8 panel."""
    segment_ids = panel.get("segment_ids", panel.get("segments", []))
    if not isinstance(segment_ids, list):
        segment_ids = [segment_ids]
    windows = [segment_windows[str(segment_id)] for segment_id in segment_ids
               if str(segment_id) in segment_windows]
    character_ids: set[str] = set()
    track_ids: set[str] = set()
    speaker_ids: set[str] = set()
    if windows:
        start = min(window[0] for window in windows)
        end = max(window[1] for window in windows)
        for turn in speaker_turns:
            overlap = max(0.0, min(end, float(turn["end"])) - max(start, float(turn["start"])))
            if overlap <= 0:
                continue
            if turn.get("character_id") is not None:
                character_ids.add(str(turn["character_id"]))
            if turn.get("track_id") is not None:
                track_ids.add(str(turn["track_id"]))
            if turn.get("speaker_id") is not None:
                speaker_ids.add(str(turn["speaker_id"]))

    # Preserve the keyframe's direct speaker evidence for standalone or older
    # Phase 8 artifacts that did not store transcript segment IDs.
    direct_character = panel.get("character_id", panel.get("speaker_character_id"))
    if direct_character not in (None, "", "unknown", "UNKNOWN"):
        character_ids.add(str(direct_character))
    for speaker_id in panel.get("speaker_ids", []):
        if speaker_id is not None:
            speaker_ids.add(str(speaker_id))
    return character_ids, track_ids, speaker_ids


def load_shot_frame_index(video_id: str, data_dir: Path) -> dict[int, list[dict[str, Any]]]:
    """Index existing Phase 4 sampled frames by shot for safe re-selection."""
    path = data_dir / "shots" / f"{video_id}.json"
    if not path.is_file():
        return {}
    data = load_json(path)
    shots = data.get("shots", []) if isinstance(data, dict) else []
    indexed: dict[int, list[dict[str, Any]]] = {}
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        try:
            shot_id = int(shot["shot_id"])
        except (KeyError, TypeError, ValueError):
            continue
        indexed[shot_id] = [frame for frame in shot.get("frames", []) if isinstance(frame, dict)]
    return indexed


def choose_conversation_frame(
    item: dict[str, Any],
    video_id: str,
    tracks_data: Any,
    shot_frames: dict[int, list[dict[str, Any]]],
    speaker_character_ids: set[str],
    speaker_track_ids: set[str],
    time_tolerance: float,
    segment_windows: dict[str, tuple[float, float]] | None = None,
    max_frame_time_shift_s: float = 2.0,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    """Use a wider same-shot frame only when it improves face visibility."""
    if not item.get("segment_ids") or item.get("shot_id") is None:
        return item, None
    try:
        candidates = shot_frames.get(int(item["shot_id"]), [])
    except (TypeError, ValueError):
        candidates = []
    if not candidates:
        return item, None

    original_id = str(item.get("frame_id", ""))
    try:
        original_time = float(item.get("t", item.get("timestamp")))
    except (TypeError, ValueError):
        original_time = None
    best_item = item
    best_faces: list[dict[str, Any]] | None = None
    best_score = -float("inf")

    for candidate in candidates:
        frame_id = str(candidate.get("frame_id", ""))
        path = find_frame_path(candidate, video_id, frame_id)
        if path is None:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        timestamp = candidate.get("t", candidate.get("timestamp"))
        try:
            timestamp = float(timestamp) if timestamp is not None else None
        except (TypeError, ValueError):
            timestamp = None
        if (
            original_time is not None
            and timestamp is not None
            and abs(timestamp - original_time) > max(0.0, max_frame_time_shift_s)
        ):
            continue
        utterance_midpoints = [
            (window[0] + window[1]) / 2.0
            for sid in item.get("segment_ids", [])
            if (window := (segment_windows or {}).get(str(sid))) is not None
        ]
        if timestamp is not None and utterance_midpoints and not all(
            abs(timestamp - midpoint) <= max(0.0, max_frame_time_shift_s)
            for midpoint in utterance_midpoints
        ):
            continue
        faces = extract_faces_for_frame(
            tracks_data, frame_id, timestamp, width, height, time_tolerance
        )
        if not faces:
            faces = extract_inline_faces(candidate, width, height)
        matched = sum(
            str(face.get("character_id")) in speaker_character_ids
            or str(face.get("track_id")) in speaker_track_ids
            for face in faces
        )
        # Favor actual speaker faces; when upstream attribution is uncertain,
        # favor a scene frame with more visible faces. Sharpness only breaks ties.
        score = matched * 100.0 + len(faces) * 10.0 + float(candidate.get("sharp", 0.0)) * 0.001
        if score > best_score:
            best_score = score
            best_faces = faces
            best_item = {**item, **candidate, "source_frame_id": original_id}

    if best_item is item:
        return item, None
    # Do not replace a keyframe on a tie: the Phase 8 choice remains the
    # narrative default unless there is a measurable visibility improvement.
    original_path = find_frame_path(item, video_id, original_id)
    original_faces: list[dict[str, Any]] = []
    if original_path is not None:
        original_image = cv2.imread(str(original_path))
        if original_image is not None:
            height, width = original_image.shape[:2]
            original_time = item.get("t", item.get("timestamp"))
            try:
                original_time = float(original_time) if original_time is not None else None
            except (TypeError, ValueError):
                original_time = None
            original_faces = extract_faces_for_frame(
                tracks_data, original_id, original_time, width, height, time_tolerance
            )
    def speaker_hits(rows: list[dict[str, Any]]) -> int:
        return sum(
            str(face.get("character_id")) in speaker_character_ids
            or str(face.get("track_id")) in speaker_track_ids
            for face in rows
        )
    if speaker_hits(best_faces or []) > speaker_hits(original_faces) or len(best_faces or []) > len(original_faces):
        return best_item, best_faces
    return item, None


def run_phase10(
    video_id: str,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Execute Phase 10 and save the layout JSON plus panel images."""
    logger = setup_logger(video_id)
    config = load_config(config_path)

    data_dir = PROJECT_ROOT / str(config["data_dir"])
    output_root = PROJECT_ROOT / str(config["output_dir"])
    output_json = output_root / f"{video_id}.json"
    panel_dir = output_root / video_id / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    keyframes_data, tracks_data, keyframes_path = load_input_artifacts(
        video_id, data_dir
    )
    segment_windows, speaker_turns = load_dialogue_speaker_context(
        video_id, data_dir
    )
    shot_frames = load_shot_frame_index(video_id, data_dir)

    keyframes = get_panels(keyframes_data)

    if not keyframes:
        raise ValueError("The keyframe artifact contains no selected panels.")

    logger.info("Loaded %d selected keyframes", len(keyframes))
    logger.info("Keyframe source: %s", keyframes_path)

    panels_per_page = max(1, int(config["panels_per_page"]))
    page_groups = [
        keyframes[i:i + panels_per_page]
        for i in range(0, len(keyframes), panels_per_page)
    ]

    pages = []
    panel_metrics = []
    global_panel_order = 0

    for page_number, group in enumerate(page_groups, start=1):
        template_name, template_ratios = choose_template(len(group))

        if len(group) == panels_per_page:
            template_name, template_ratios = choose_template(
                panels_per_page
            )

        page_panels = []

        for local_index, item in enumerate(group):
            global_panel_order += 1

            speaker_character_ids, speaker_track_ids, active_speaker_ids = panel_speaker_targets(
                item, segment_windows, speaker_turns
            )
            item, replacement_faces = choose_conversation_frame(
                item,
                video_id,
                tracks_data,
                shot_frames,
                speaker_character_ids,
                speaker_track_ids,
                float(config["time_tolerance_s"]),
                segment_windows,
                float(config.get("conversation_frame_max_shift_s", 2.0)),
            )

            frame_id = str(
                item.get(
                    "frame_id",
                    item.get(
                        "id",
                        item.get("keyframe_id", f"frame_{global_panel_order:04d}"),
                    ),
                )
            )

            timestamp = item.get("t", item.get("timestamp", item.get("time")))

            if timestamp is not None:
                try:
                    timestamp = float(timestamp)
                except (TypeError, ValueError):
                    timestamp = None

            frame_path = find_frame_path(item, video_id, frame_id)

            if frame_path is None:
                logger.warning(
                    "Skipping panel %s: image file was not found.", frame_id
                )
                continue

            frame = cv2.imread(str(frame_path))

            if frame is None:
                logger.warning(
                    "Skipping panel %s: OpenCV could not read %s",
                    frame_id, frame_path,
                )
                continue

            height, width = frame.shape[:2]

            faces = replacement_faces or extract_faces_for_frame(
                tracks_data=tracks_data,
                frame_id=frame_id,
                timestamp=timestamp,
                width=width,
                height=height,
                time_tolerance=float(config["time_tolerance_s"]),
            )

            # Prefer inline face boxes only if tracks did not provide any.
            if not faces:
                faces = extract_inline_faces(item, width, height)

            saliency = get_saliency_map(frame)

            rois = normalize_roi(
                faces,
                saliency,
                width,
                height,
                speaker_character_ids,
                speaker_track_ids,
            )

            aspect_ratio = float(
                item.get(
                    "panel_aspect_ratio",
                    template_ratios[
                        min(local_index, len(template_ratios) - 1)
                    ] if template_ratios else config["panel_aspect_ratio"],
                )
            )

            if aspect_ratio <= 0:
                aspect_ratio = float(config["panel_aspect_ratio"])

            all_face_rois = [roi for roi in rois if roi.get("kind") in ("face", "speaker_face")]
            speaker_faces = [roi for roi in rois if roi.get("kind") == "speaker_face"]
            # If Phase 7 cannot identify a speaker, preserve a visible
            # conversation group rather than zooming toward a single face.
            has_dialogue = bool(item.get("segment_ids", item.get("segments", [])))
            # A dialogue panel with two visible faces is a conversation shot.
            # Keep the listener together with the active speaker even when
            # this individual transcript segment has only one speaker.
            framing_faces = (
                all_face_rois if len(all_face_rois) >= 2 and (has_dialogue or len(speaker_faces) >= 2) else
                speaker_faces or all_face_rois
            )
            aspect_ratio, composition_mode, speaker_group_box = adaptive_speaker_aspect_ratio(
                framing_faces, width, height, aspect_ratio
            )

            crop_result = find_best_crop(
                frame,
                rois,
                aspect_ratio,
                saliency,
                grid_steps=int(config["grid_steps"]),
                required_rois=framing_faces if len(framing_faces) > 1 else speaker_faces,
                minimum_required_retention=float(config["speaker_group_min_retention"]),
            )

            x1, y1, x2, y2 = map(int, crop_result["crop"])
            crop_image = frame[y1:y2, x1:x2].copy()

            if crop_image.size == 0:
                logger.warning(
                    "Skipping panel %s: crop has no pixels.", frame_id
                )
                continue

            safe_zone = find_bubble_safe_zone(
                tuple(crop_result["crop"]),
                rois,
                saliency,
            )

            # The scale maps original crop pixels to the saved panel image.
            target_height = max(1, int(config["target_panel_height"]))
            target_width = max(
                1,
                int(round(target_height * crop_image.shape[1] /
                          crop_image.shape[0])),
            )

            scale_x = target_width / crop_image.shape[1]
            scale_y = target_height / crop_image.shape[0]

            resized = cv2.resize(
                crop_image,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA
                if target_width < crop_image.shape[1]
                else cv2.INTER_CUBIC,
            )

            panel_image_path = (
                panel_dir / f"panel_{global_panel_order:04d}.jpg"
            )

            if bool(config["save_panel_images"]):
                if not cv2.imwrite(str(panel_image_path), resized):
                    raise IOError(
                        f"Failed to write panel image: {panel_image_path}"
                    )

            # Convert safe-zone coordinates to crop-local coordinates.
            safe_crop_local = [
                int(safe_zone[0] - x1),
                int(safe_zone[1] - y1),
                int(safe_zone[2] - x1),
                int(safe_zone[3] - y1),
            ]

            safe_panel_coords = [
                int(round(safe_crop_local[0] * scale_x)),
                int(round(safe_crop_local[1] * scale_y)),
                int(round(safe_crop_local[2] * scale_x)),
                int(round(safe_crop_local[3] * scale_y)),
            ]

            roi_retention = float(crop_result["roi_retention"])

            panel_record = {
                "panel_id": global_panel_order,
                "page": page_number,
                "frame_id": frame_id,
                "original_frame_id": item.get("source_frame_id", frame_id),
                "timestamp": timestamp,
                "shot_id": item.get("shot_id"),
                "shot_start": item.get("shot_start"),
                "shot_end": item.get("shot_end"),
                "segment_ids": list(item.get("segment_ids", [])),
                "story_beat_id": item.get("story_beat_id"),
                "story_role": item.get("story_role"),
                "story_importance": item.get("story_importance"),
                "visual_action_caption": item.get("visual_action_caption"),
                "visual_action_label": item.get("visual_action_label"),
                "source_image": str(frame_path.relative_to(PROJECT_ROOT))
                if frame_path.is_relative_to(PROJECT_ROOT)
                else str(frame_path),
                "panel_image": str(
                    panel_image_path.relative_to(PROJECT_ROOT)
                ) if bool(config["save_panel_images"]) else None,
                # Retain the historical singular field for consumers that
                # expect it, while `speaking_character_ids` holds the full
                # multi-speaker association used by adaptive framing.
                "character_id": next(iter(sorted(speaker_character_ids)), None),
                "crop": [x1, y1, x2, y2],
                "crop_format": "xyxy",
                "crop_coordinate_space": "original_frame_pixels",
                "panel_size": [target_width, target_height],
                "scale": float(scale_y),
                "scale_x": float(scale_x),
                "scale_y": float(scale_y),
                "crop_transform": {
                    "offset_x": x1,
                    "offset_y": y1,
                    "scale_x": float(scale_x),
                    "scale_y": float(scale_y),
                    "source_width": int(width),
                    "source_height": int(height),
                    "panel_width": int(target_width),
                    "panel_height": int(target_height),
                    "formula": (
                        "panel_x=(original_x-offset_x)*scale_x; "
                        "panel_y=(original_y-offset_y)*scale_y"
                    ),
                },
                "rois": [
                    {
                        "box": [
                            float(v) for v in roi["box"]
                        ],
                        "kind": roi["kind"],
                        "weight": float(roi["weight"]),
                        "character_id": roi.get("character_id"),
                        "track_id": roi.get("track_id"),
                        "mouth_center": roi.get("mouth_center"),
                        "det_score": roi.get("det_score"),
                        "face_orientation": roi.get("face_orientation"),
                    }
                    for roi in rois
                ],
                "safe_zones": [safe_zone],
                "safe_zones_crop_local": [safe_crop_local],
                "safe_zones_panel_coordinates": [safe_panel_coords],
                "roi_retention": roi_retention,
                "saliency_score": float(crop_result["saliency_score"]),
                "crop_score": float(crop_result["crop_score"]),
                "speaker_roi_retention": float(crop_result["speaker_roi_retention"]),
                "face_count": len(faces),
                "speaking_character_ids": sorted(speaker_character_ids),
                "speaking_track_ids": sorted(speaker_track_ids),
                "active_speaker_ids": sorted(active_speaker_ids),
                "speaker_face_count": len(speaker_faces),
                "framing_face_count": len(framing_faces),
                "speaker_group_box": speaker_group_box,
                "composition_mode": composition_mode,
                "template": template_name,
            }

            page_panels.append(panel_record)
            panel_metrics.append(panel_record)

        pages.append({
            "page": page_number,
            "template": template_name,
            "panels": page_panels,
        })

    if not panel_metrics:
        raise RuntimeError(
            "Phase 10 did not produce any panels. Check frame paths and "
            "the Phase 8 keyframe JSON structure."
        )

    # Automatic metrics. These are proxies, not manually verified accuracy.
    face_cut_count = 0
    face_roi_count = 0
    speaker_face_cut_count = 0
    speaker_face_roi_count = 0

    for panel in panel_metrics:
        x1, y1, x2, y2 = panel["crop"]

        for roi in panel["rois"]:
            if roi["kind"] not in ("face", "speaker_face"):
                continue

            face_roi_count += 1
            rx1, ry1, rx2, ry2 = roi["box"]

            center_x = (rx1 + rx2) / 2
            center_y = (ry1 + ry2) / 2

            # Count a face as cut when its box extends outside the crop.
            is_cut = (
                rx1 < x1
                or ry1 < y1
                or rx2 > x2
                or ry2 > y2
                or not (x1 <= center_x <= x2 and y1 <= center_y <= y2)
            )
            if is_cut:
                face_cut_count += 1
            if roi["kind"] == "speaker_face":
                speaker_face_roi_count += 1
                if is_cut:
                    speaker_face_cut_count += 1

    retentions = [p["roi_retention"] for p in panel_metrics]

    mean_retention = float(np.mean(retentions))
    min_retention = float(np.min(retentions))

    # Panel crop aspect error should be near zero by construction.
    aspect_errors = []

    for panel in panel_metrics:
        x1, y1, x2, y2 = panel["crop"]
        crop_ratio = (x2 - x1) / max(1, y2 - y1)
        target_ratio = panel["panel_size"][0] / max(
            1, panel["panel_size"][1]
        )
        aspect_errors.append(abs(crop_ratio - target_ratio))

    metrics = {
        "panel_count": len(panel_metrics),
        "page_count": len(pages),
        "mean_roi_retention": mean_retention,
        "minimum_roi_retention": min_retention,
        "face_roi_count": face_roi_count,
        "face_cut_count": face_cut_count,
        "face_cut_rate": (
            face_cut_count / face_roi_count
            if face_roi_count
            else None
        ),
        "speaker_face_roi_count": speaker_face_roi_count,
        "speaker_face_cut_count": speaker_face_cut_count,
        "speaker_face_cut_rate": (
            speaker_face_cut_count / speaker_face_roi_count
            if speaker_face_roi_count else None
        ),
        "mean_aspect_error": float(np.mean(aspect_errors)),
        "pass_roi_retention_095": mean_retention >= 0.95,
        "pass_face_cut_rate_zero": face_cut_count == 0,
        "pass_speaker_face_cut_rate_zero": speaker_face_cut_count == 0,
        "note": (
            "Automatic ROI and face-cut estimates depend on upstream face "
            "annotations. Visually inspect panels before treating these "
            "metrics as validated."
        ),
    }

    output = {
        "video_id": video_id,
        "phase": 10,
        "stage": "roi_estimation_and_comic_layout",
        "coordinate_convention": (
            "All crop and ROI boxes use [x1, y1, x2, y2] in original "
            "frame pixel coordinates. Safe zones are also recorded in "
            "original, crop-local, and resized-panel coordinates."
        ),
        "source_keyframes": str(keyframes_path),
        "pages": pages,
        "metrics": metrics,
        "config": config,
    }

    save_json(output_json, output)

    if bool(config["save_contact_sheet"]):
        create_contact_sheet(
            panel_metrics,
            output_root / video_id / "contact_sheet.jpg",
        )

    logger.info("Phase 10 completed.")
    logger.info("Panels: %d", len(panel_metrics))
    logger.info("Pages: %d", len(pages))
    logger.info("Mean ROI retention: %.4f", mean_retention)
    logger.info("Minimum ROI retention: %.4f", min_retention)
    logger.info("Face-cut count: %d", face_cut_count)
    logger.info("Saved layout: %s", output_json)

    return output


def create_contact_sheet(
    panels: list[dict[str, Any]],
    output_path: Path,
    columns: int = 3,
    thumbnail_width: int = 320,
) -> None:
    """Create a contact sheet to visually inspect the panel crops."""
    thumbnails = []

    for panel in panels:
        image_path_value = panel.get("panel_image")

        if not image_path_value:
            continue

        image_path = Path(image_path_value)

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]
        thumb_height = max(
            1, int(round(height * thumbnail_width / max(1, width)))
        )

        thumb = cv2.resize(
            image,
            (thumbnail_width, thumb_height),
            interpolation=cv2.INTER_AREA,
        )

        label = f"Panel {panel['panel_id']} | {panel['frame_id']}"
        cv2.rectangle(thumb, (0, 0), (thumbnail_width, 28), (0, 0, 0), -1)
        cv2.putText(
            thumb,
            label[:45],
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        thumbnails.append(thumb)

    if not thumbnails:
        return

    rows = math.ceil(len(thumbnails) / columns)
    max_thumb_height = max(thumb.shape[0] for thumb in thumbnails)
    sheet = np.full(
        (rows * max_thumb_height, columns * thumbnail_width, 3),
        245,
        dtype=np.uint8,
    )

    for index, thumb in enumerate(thumbnails):
        row = index // columns
        col = index % columns

        y = row * max_thumb_height
        x = col * thumbnail_width

        sheet[
            y:y + thumb.shape[0],
            x:x + thumb.shape[1],
        ] = thumb

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), sheet):
        raise IOError(f"Failed to save contact sheet: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 10: ROI estimation, crop optimization, "
            "page layout, and panel cropping."
        )
    )

    parser.add_argument(
        "--video-id",
        required=True,
        help="Video ID generated by Phase 1.",
    )

    parser.add_argument(
        "--config",
        default="configs/phase10.yaml",
        help="Path to the Phase 10 YAML configuration.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    try:
        result = run_phase10(args.video_id, config_path)
        print("\nPhase 10 completed successfully.")
        print(f"Pages: {result['metrics']['page_count']}")
        print(f"Panels: {result['metrics']['panel_count']}")
        print(
            "Mean ROI retention: "
            f"{result['metrics']['mean_roi_retention']:.4f}"
        )
        print(f"Output: data/layout/{args.video_id}.json")

    except Exception as exc:
        logger = setup_logger(args.video_id)
        logger.exception("Phase 10 failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
