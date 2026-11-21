from src.speaker.phase7 import (
    interpolate_track_bbox,
    rank_active_speaker_candidates,
)


def candidate(
    track_id: str,
    *,
    motion_available: bool,
    motion_contrast: float = 0.0,
    temporal_overlap: float = 0.9,
    track_confidence: float = 0.9,
):
    return {
        "track_id": track_id,
        "motion_available": motion_available,
        "motion_contrast": motion_contrast,
        "temporal_overlap": temporal_overlap,
        "track_confidence": track_confidence,
    }


def rank(candidates):
    return rank_active_speaker_candidates(
        candidates,
        minimum_active_contrast=0.012,
        strong_active_contrast=0.055,
        minimum_match_score=0.45,
        minimum_candidate_margin=0.03,
    )


def test_track_box_interpolates_only_inside_observed_track():
    detections = [
        {"t": 1.0, "bbox": [0, 0, 20, 20]},
        {"t": 2.0, "bbox": [10, 20, 30, 40]},
    ]

    assert interpolate_track_bbox(detections, 1.5, fps=24) == [5.0, 10.0, 25.0, 30.0]
    assert interpolate_track_bbox(detections, 0.5, fps=24) is None
    assert interpolate_track_bbox(detections, 1.5, fps=24, max_gap_s=0.5) is None


def test_visible_face_without_mouth_motion_is_not_attributed():
    result = rank([candidate("track-1", motion_available=True, motion_contrast=0.005)])

    assert result["accepted"] is False
    assert result["decision"] == "no_active_speaker_evidence"
    assert result["best"]["active_speaker_evidence"] is False


def test_visibility_cannot_substitute_for_unavailable_motion():
    result = rank([candidate("track-1", motion_available=False, temporal_overlap=1.0)])

    assert result["accepted"] is False
    assert result["decision"] == "mouth_motion_unavailable"


def test_unique_active_mouth_evidence_can_match():
    result = rank([candidate("track-1", motion_available=True, motion_contrast=0.07)])

    assert result["accepted"] is True
    assert result["decision"] == "matched"
    assert result["best"]["track_id"] == "track-1"


def test_two_equally_active_visible_faces_remain_ambiguous():
    result = rank([
        candidate("track-1", motion_available=True, motion_contrast=0.07),
        candidate("track-2", motion_available=True, motion_contrast=0.07),
    ])

    assert result["accepted"] is False
    assert result["decision"] == "low_confidence_or_ambiguous"
