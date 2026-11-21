"""Build evidence-linked comic story plans without generating story content."""

from __future__ import annotations

import math
import re
from typing import Any


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _segment_id(segment: dict[str, Any], index: int) -> str:
    for key in ("segment_id", "segmentId", "id", "uid"):
        if segment.get(key) is not None:
            return str(segment[key])
    return str(index)


_OBSERVED_ACTION_LABELS = {
    "talking", "walking", "running", "fighting", "pointing", "sitting",
    "standing", "holding", "working", "reacting", "laughing", "crying",
    "looking", "driving", "entering", "group", "landscape", "object",
}


def supported_action_caption(caption: Any, label: Any) -> str | None:
    """Accept only captions tied to the existing visual cue vocabulary."""
    text = _clean_text(caption)
    action = str(label or "").strip().casefold()
    if text and action in _OBSERVED_ACTION_LABELS:
        return text
    return None


def _dialogue_context(dialogue: list[dict[str, Any]]) -> str | None:
    """Describe turn structure without copying transcript wording."""
    if not dialogue:
        return None
    speakers = [
        str(line["speaker_id"])
        for line in dialogue
        if line.get("speaker_id")
    ]
    distinct = list(dict.fromkeys(speakers))
    has_question = any("?" in str(line.get("text", "")) for line in dialogue)
    if len(distinct) > 1 and has_question:
        return "A question is followed by another speaker's response."
    if len(distinct) > 1:
        return "Two or more voices alternate in conversation."
    # A single-speaker caption adds no useful context and becomes repetitive
    # when the same turn spans several consecutive panels.
    return None


def _speaker_for_segment(
    segment: dict[str, Any],
    speaker_turns: list[dict[str, Any]],
) -> dict[str, Any]:
    start = _number(segment.get("start"))
    end = _number(segment.get("end"), start)
    best: dict[str, Any] | None = None
    best_overlap = 0.0
    for turn in speaker_turns:
        turn_start = _number(turn.get("start"), float("inf"))
        turn_end = _number(turn.get("end"), float("-inf"))
        overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
        if overlap > best_overlap:
            best, best_overlap = turn, overlap
    if best is None:
        return {}
    return {
        "speaker_id": best.get("speaker_id"),
        "character_id": best.get("character_id"),
        "track_id": best.get("track_id"),
        "match_score": best.get("confidence"),
        "decision": best.get("decision"),
    }


def _best_scene_caption(events: list[dict[str, Any]]) -> str | None:
    caption_counts: dict[str, int] = {}
    for event in events:
        caption = event.get("visual_evidence", {}).get("action_caption")
        if caption:
            caption_counts[str(caption)] = caption_counts.get(str(caption), 0) + 1
    if not caption_counts:
        return None
    # Use the most repeated accepted cue in this local scene interval. This
    # avoids concatenating unrelated frame labels into a video-wide summary.
    return max(
        caption_counts,
        key=lambda caption: (caption_counts[caption], -len(caption)),
    )


def _scene_summary(events: list[dict[str, Any]]) -> str:
    best_caption = _best_scene_caption(events)
    if best_caption:
        summary = f"Across these panels: {best_caption}"
    else:
        summary = "The selected frames do not provide a clear visual action summary."

    dialogue = [
        line
        for event in events
        for line in event.get("dialogue", [])
    ]
    context = _dialogue_context(dialogue)
    if context:
        summary += " " + context
    return summary


_SCENE_TOPIC = {
    "landscape": "place",
    "walking": "movement",
    "running": "movement",
    "entering": "movement",
    "driving": "movement",
    "fighting": "conflict",
    "talking": "interaction",
    "standing": "interaction",
    "pointing": "interaction",
    "reacting": "interaction",
    "laughing": "interaction",
    "crying": "interaction",
    "group": "interaction",
    "sitting": "activity",
    "holding": "activity",
    "working": "activity",
    "looking": "activity",
    "object": "activity",
}


def _visual_topic(event: dict[str, Any]) -> str | None:
    label = str(event.get("visual_evidence", {}).get("action_label") or "").casefold()
    return _SCENE_TOPIC.get(label)


def _sustained_visual_transition(
    previous_events: list[dict[str, Any]],
    upcoming_events: list[dict[str, Any]],
) -> bool:
    """Detect a persistent coarse action shift, not a single camera cut."""
    previous_topics = [
        topic for item in previous_events[-3:]
        if (topic := _visual_topic(item)) is not None
    ]
    if len(previous_topics) < 2:
        return False
    previous_topic = max(set(previous_topics), key=previous_topics.count)
    if previous_topics.count(previous_topic) < 2:
        return False

    next_topics = [
        topic for item in upcoming_events[:3]
        if (topic := _visual_topic(item)) is not None
    ]
    if len(next_topics) < 2:
        return False
    next_topic = max(set(next_topics), key=next_topics.count)
    return (
        next_topic != previous_topic
        and next_topics.count(next_topic) >= 2
        and _visual_topic(upcoming_events[0]) == next_topic
    )


def build_story_plan(
    video_id: str,
    selected_panels: list[dict[str, Any]],
    transcript_segments: list[dict[str, Any]],
    visual_actions: dict[str, Any] | None = None,
    speaker_turns: list[dict[str, Any]] | None = None,
    max_scene_span_s: float = 50.0,
) -> dict[str, Any]:
    """Structure selected video evidence into chronological, non-fictional beats."""
    transcript = {
        _segment_id(segment, index): segment
        for index, segment in enumerate(transcript_segments)
    }
    actions = visual_actions or {}
    turns = speaker_turns or []
    ordered = sorted(
        selected_panels,
        key=lambda item: (
            item.get("t") is None,
            _number(item.get("t"), float("inf")),
            str(item.get("frame_id", "")),
        ),
    )

    events: list[dict[str, Any]] = []
    prior_shot = None
    for index, panel in enumerate(ordered):
        frame_id = str(panel.get("frame_id", ""))
        action = actions.get(frame_id, {})
        action = action if isinstance(action, dict) else {}
        action_label = action.get("label")
        caption = (
            supported_action_caption(action.get("caption"), action_label)
            if action.get("status") == "ok" else None
        )
        segment_ids = sorted(
            {
                str(value)
                for value in panel.get("segment_ids", [])
                if str(value) in transcript
            },
            key=lambda sid: (
                _number(transcript[sid].get("start")),
                sid,
            ),
        )
        dialogue = []
        speakers = {str(value) for value in (panel.get("speaker_ids") or []) if value is not None}
        for sid in segment_ids:
            source = transcript[sid]
            speaker = _speaker_for_segment(source, turns)
            speaker_id = source.get("speaker_id") or speaker.get("speaker_id")
            if speaker_id:
                speakers.add(str(speaker_id))
            dialogue.append({
                "segment_id": sid,
                "start": _number(source.get("start")),
                "end": _number(source.get("end"), _number(source.get("start"))),
                "text": _clean_text(source.get("text")),
                "speaker_id": str(speaker_id) if speaker_id else None,
                "character_id": source.get("character_id") or speaker.get("character_id"),
                "track_id": source.get("track_id") or speaker.get("track_id"),
                "speaker_match_score": source.get("speaker_match_score") or speaker.get("match_score"),
                "speaker_decision": source.get("speaker_decision") or speaker.get("decision"),
            })

        shot_id = panel.get("shot_id")
        is_transition = index > 0 and shot_id is not None and prior_shot is not None and str(shot_id) != str(prior_shot)
        scores = panel.get("scores", {}) if isinstance(panel.get("scores"), dict) else {}
        action_score = _number(scores.get("action"))
        importance = max(
            0.0,
            min(1.0, 0.45 * action_score
                + 0.20 * _number(scores.get("sem"))
                + 0.15 * _number(scores.get("spk"))
                + 0.10 * _number(scores.get("sal"))
                + 0.10 * _number(scores.get("q"))),
        )
        if index == 0:
            role = "opening"
        elif index == len(ordered) - 1:
            role = "conclusion"
        elif is_transition:
            role = "scene_transition"
        elif action_score >= 0.45 or str(action_label or "").casefold() in _OBSERVED_ACTION_LABELS:
            role = "important_action"
        elif dialogue:
            role = "dialogue_exchange"
        else:
            role = "visual_development"

        event_id = f"beat_{index + 1:03d}"
        events.append({
            "beat_id": event_id,
            "sequence": index + 1,
            "role": role,
            "start": _number(panel.get("t")),
            "end": _number(panel.get("shot_end"), _number(panel.get("t"))),
            "frame_id": frame_id,
            "shot_id": shot_id,
            "shot_start": panel.get("shot_start"),
            "shot_end": panel.get("shot_end"),
            "segment_ids": segment_ids,
            "dialogue": dialogue,
            "speaker_ids": sorted(speakers),
            "character_ids": sorted({
                str(value)
                for value in [panel.get("character_id"), *(panel.get("character_ids") or [])]
                if value not in (None, "", "unknown", "UNKNOWN")
            }),
            "visual_evidence": {
                "action_caption": caption or None,
                "action_label": action_label,
                "action_confidence": _number(action.get("confidence")) if action else None,
                "action_status": action.get("status") if action else "unavailable",
                "action_score": round(action_score, 4),
                "semantic_score": round(_number(scores.get("sem")), 4),
                "speaker_score": round(_number(scores.get("spk")), 4),
                "frame_quality": round(_number(scores.get("q")), 4),
                "alignment_method": panel.get("alignment_method"),
            },
            "importance": round(importance, 4),
            "scene_transition": bool(is_transition),
        })
        panel["story_beat_id"] = event_id
        panel["story_role"] = role
        panel["story_importance"] = round(importance, 4)
        panel["visual_action_caption"] = caption
        panel["visual_action_label"] = action_label
        prior_shot = shot_id

    # A Phase 4 shot is a cut, not necessarily a new scene. Keep adjacent cuts
    # together while dialogue and time remain continuous; split on a sustained
    # dialogue gap or a clearly separate time interval.
    scene_groups: list[list[dict[str, Any]]] = []
    for event_index, event in enumerate(events):
        if not scene_groups:
            scene_groups.append([event])
            continue
        prior_group = scene_groups[-1]
        prior = prior_group[-1]
        prior_dialogue_end = max(
            (_number(line.get("end")) for item in prior_group for line in item.get("dialogue", [])),
            default=_number(prior.get("end"), _number(prior.get("start"))),
        )
        current_dialogue_start = min(
            (_number(line.get("start")) for line in event.get("dialogue", [])),
            default=_number(event.get("start")),
        )
        dialogue_gap = current_dialogue_start - prior_dialogue_end
        time_gap = _number(event.get("start")) - _number(prior.get("start"))
        scene_span = _number(event.get("start")) - _number(prior_group[0].get("start"))
        sustained_visual_transition = (
            scene_span >= 12.0
            and _sustained_visual_transition(prior_group, events[event_index:])
        )
        interval_limit = (
            max_scene_span_s > 0.0
            and scene_span >= max_scene_span_s
        )
        if dialogue_gap >= 12.0 or time_gap >= 18.0 or sustained_visual_transition or interval_limit:
            scene_groups.append([event])
        else:
            prior_group.append(event)

    scenes: list[dict[str, Any]] = []
    for scene_index, group in enumerate(scene_groups, start=1):
        scene_id = f"SCENE_{scene_index:02d}"
        summary = _scene_summary(group)
        best_caption = _best_scene_caption(group)
        summary_anchor = next(
            (
                event for event in group
                if best_caption
                and event["visual_evidence"].get("action_caption") == best_caption
            ),
            group[0],
        )
        scene_start = min(_number(event.get("start")) for event in group)
        scene_end = max(_number(event.get("end")) for event in group)
        if scene_index < len(scene_groups):
            next_scene_start = min(
                _number(event.get("start")) for event in scene_groups[scene_index]
            )
            scene_end = min(scene_end, next_scene_start)
        scene_dialogue = [line for event in group for line in event.get("dialogue", [])]
        scene_characters = sorted({
            str(value)
            for event in group
            for value in event.get("character_ids", [])
        })
        seen_visual_descriptions: set[str] = set()
        repeated_caption_counts: dict[str, int] = {}
        uncertain_frame_count = 0
        for event in group:
            caption = event["visual_evidence"].get("action_caption")
            if caption:
                if caption in seen_visual_descriptions:
                    repeat_options = (
                        "The same visible activity continues in this shot.",
                        "The view changes while that activity continues.",
                        "The activity remains the focus of this view.",
                        "Another shot holds on the same activity.",
                    )
                    repeat_index = repeated_caption_counts.get(caption, 0)
                    visual_description = repeat_options[repeat_index % len(repeat_options)]
                    repeated_caption_counts[caption] = repeat_index + 1
                else:
                    visual_description = caption
                    seen_visual_descriptions.add(caption)
            else:
                uncertain_options = (
                    "The visible action is unclear in this frame.",
                    "This shot continues, but the action is unclear.",
                    "The view changes while the action remains uncertain.",
                )
                visual_description = uncertain_options[
                    uncertain_frame_count % len(uncertain_options)
                ]
                uncertain_frame_count += 1
            context = _dialogue_context(event.get("dialogue", []))
            panel_summary = visual_description
            if context:
                panel_summary += " " + context
            event.update({
                "scene_id": scene_id,
                "scene_summary": summary,
                "visual_description": visual_description,
                "dialogue_context": context,
                "panel_summary": panel_summary,
            })
        scenes.append({
            "scene_id": scene_id,
            "scene_start": scene_start,
            "scene_end": max(scene_start, scene_end),
            "scene_summary": summary,
            "summary_anchor_panel_id": summary_anchor["beat_id"],
            "characters_present": scene_characters,
            "dialogue_context": _dialogue_context(scene_dialogue),
            "panels": [{
                "panel_id": event["beat_id"],
                "visual_description": event["visual_description"],
                "actions": [event["visual_evidence"]["action_caption"]]
                    if event["visual_evidence"].get("action_caption") else [],
                "dialogue_context": event["dialogue_context"],
                "panel_summary": event["panel_summary"],
            } for event in group],
            "summary_basis": "timestamped Phase 6 visual cues and diarized turn structure",
        })

    important_events = [
        event["beat_id"]
        for event in sorted(
            (event for event in events if event["visual_evidence"]["action_score"] >= 0.45),
            key=lambda event: event["importance"],
            reverse=True,
        )
    ]
    return {
        "schema_version": 1,
        "video_id": video_id,
        "source_of_truth": "Phase 1 video, timestamped transcript, shot boundaries, and existing model evidence",
        "generation_policy": "No dialogue or events are generated; all dialogue text is copied from transcript segment IDs.",
        "summary_policy": "Visual claims use only accepted existing visual cues; dialogue summaries describe turn structure without quoting transcript text.",
        "chronological": True,
        "scenes": scenes,
        "beginning": events[0]["beat_id"] if events else None,
        "important_events": important_events,
        "conclusion": events[-1]["beat_id"] if events else None,
        "events": events,
    }


def validate_story_output(
    plan: dict[str, Any],
    transcript: dict[str, dict[str, Any]],
    rendered_panels: list[dict[str, Any]],
    frame_tolerance_s: float = 2.0,
    video_duration_s: float | None = None,
    bubble_coverage_limit: float = 0.42,
) -> dict[str, Any]:
    """Check transcript integrity, order, duplicates, and frame/scene grounding."""
    issues: list[dict[str, Any]] = []
    occurrences: dict[str, list[dict[str, Any]]] = {}
    chronological = True
    prior_start = -float("inf")
    text_mismatches = 0
    frame_mismatches = 0
    plan_text_mismatches = 0
    repeated_frames: dict[str, int] = {}
    missing_panel_summaries = 0
    missing_scene_summaries = 0
    speaker_visual_unmatched = 0
    missing_speaker_ids = 0
    panels_missing_listener = 0
    bubble_overlaps = 0
    bubbles_outside_panel = 0
    missing_speech_tails = 0
    incorrect_speech_tails = 0
    dense_panels = 0
    fragment_text: dict[tuple[str, ...], list[tuple[float, str, Any]]] = {}

    if not plan.get("events"):
        issues.append({"type": "story_plan_missing"})
    if not plan.get("scenes"):
        issues.append({"type": "scene_summaries_missing"})
    for event in plan.get("events", []):
        if not _clean_text(event.get("panel_summary")):
            missing_panel_summaries += 1
            issues.append({"type": "panel_summary_missing", "beat_id": event.get("beat_id")})
        if not _clean_text(event.get("scene_summary")):
            missing_scene_summaries += 1
            issues.append({"type": "scene_summary_missing", "beat_id": event.get("beat_id")})
        for line in event.get("dialogue", []):
            sid = str(line.get("segment_id", ""))
            source = transcript.get(sid)
            if source is None or _clean_text(line.get("text")) != _clean_text(source.get("text")):
                plan_text_mismatches += 1
                issues.append({
                    "type": "story_plan_transcript_mismatch",
                    "beat_id": event.get("beat_id"),
                    "segment_id": sid,
                })

    for panel in rendered_panels:
        frame_id = str(panel.get("frame_id", ""))
        if frame_id:
            repeated_frames[frame_id] = repeated_frames.get(frame_id, 0) + 1
        if not _clean_text(panel.get("panel_summary")):
            missing_panel_summaries += 1
            issues.append({"type": "panel_summary_missing", "panel_id": panel.get("panel_id")})
        if not _clean_text(panel.get("scene_summary")):
            missing_scene_summaries += 1
            issues.append({"type": "scene_summary_missing", "panel_id": panel.get("panel_id")})
        dimensions = panel.get("panel_dimensions")
        panel_width, panel_height = (
            (int(dimensions[0]), int(dimensions[1]))
            if isinstance(dimensions, (list, tuple)) and len(dimensions) >= 2
            else (0, 0)
        )
        boxes: list[tuple[int, int, int, int]] = []
        for bubble in panel.get("bubbles", []):
            if not bubble.get("speaker_id"):
                missing_speaker_ids += 1
            elif not bubble.get("speaker_face_visible") and not bubble.get("off_panel_speaker"):
                speaker_visual_unmatched += 1
            source_ids = [str(value) for value in bubble.get("source_segment_ids", [])]
            if not source_ids and bubble.get("segment_id") is not None:
                source_ids = [str(bubble["segment_id"])]
            for sid in source_ids:
                occurrences.setdefault(sid, []).append({
                    "panel_id": panel.get("panel_id"),
                    "fragment": bool(bubble.get("dialogue_fragment")),
                })
            if bubble.get("dialogue_fragment") and source_ids:
                fragment_text.setdefault(tuple(source_ids), []).append((
                    _number(bubble.get("start")),
                    _clean_text(bubble.get("text")),
                    panel.get("panel_id"),
                ))

            bbox = bubble.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                box = tuple(int(value) for value in bbox[:4])
                x1, y1, x2, y2 = box
                if panel_width and panel_height and (
                    x1 < 0 or y1 < 0 or x2 > panel_width or y2 > panel_height
                    or x2 <= x1 or y2 <= y1
                ):
                    bubbles_outside_panel += 1
                    issues.append({
                        "type": "bubble_outside_panel",
                        "panel_id": panel.get("panel_id"),
                        "bbox": list(box),
                    })
                for prior_box in boxes:
                    overlap_w = max(0, min(x2, prior_box[2]) - max(x1, prior_box[0]))
                    overlap_h = max(0, min(y2, prior_box[3]) - max(y1, prior_box[1]))
                    if overlap_w * overlap_h > 9:
                        bubble_overlaps += 1
                        issues.append({"type": "bubble_overlap", "panel_id": panel.get("panel_id")})
                        break
                boxes.append(box)
            if bubble.get("speaker_face_visible"):
                target = bubble.get("mouth_panel_coordinates")
                tip = bubble.get("tail_tip")
                if isinstance(target, (list, tuple)) and len(target) >= 2:
                    if not isinstance(tip, (list, tuple)) or len(tip) < 2:
                        missing_speech_tails += 1
                        issues.append({"type": "speaker_tail_missing", "panel_id": panel.get("panel_id")})
                    elif math.hypot(
                        _number(tip[0]) - _number(target[0]),
                        _number(tip[1]) - _number(target[1]),
                    ) > 16.0:
                        incorrect_speech_tails += 1
                        issues.append({"type": "speaker_tail_misdirected", "panel_id": panel.get("panel_id")})

            start = _number(bubble.get("start"), prior_start)
            if start + 1e-6 < prior_start:
                chronological = False
                issues.append({
                    "type": "dialogue_order",
                    "panel_id": panel.get("panel_id"),
                    "segment_id": bubble.get("segment_id"),
                })
            prior_start = max(prior_start, start)

            if source_ids and not bubble.get("dialogue_fragment"):
                expected = " ".join(
                    _clean_text(transcript[sid].get("text"))
                    for sid in source_ids if sid in transcript
                )
                actual = _clean_text(bubble.get("source_text", bubble.get("text", "")))
                if _clean_text(expected) != actual:
                    text_mismatches += 1
                    issues.append({
                        "type": "transcript_text_mismatch",
                        "panel_id": panel.get("panel_id"),
                        "segment_ids": source_ids,
                    })
                frame_time = _number(panel.get("timestamp"), float("inf"))
                shot_start = panel.get("shot_start")
                shot_end = panel.get("shot_end")
                for sid in source_ids:
                    source = transcript.get(sid)
                    if not source:
                        continue
                    start = _number(source.get("start"))
                    end = _number(source.get("end"), start)
                    temporal_distance = max(start - frame_time, 0.0, frame_time - end)
                    if temporal_distance > frame_tolerance_s:
                        frame_mismatches += 1
                        issues.append({
                            "type": "dialogue_frame_mismatch",
                            "panel_id": panel.get("panel_id"),
                            "segment_id": sid,
                            "frame_id": panel.get("frame_id"),
                        })

        panel_area = max(1, panel_width * panel_height)
        occupied_area = sum(
            max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            for box in boxes
        )
        if len(boxes) > 1 and occupied_area / panel_area > bubble_coverage_limit:
            dense_panels += 1
            issues.append({
                "type": "dialogue_density_review",
                "panel_id": panel.get("panel_id"),
                "bubble_coverage_ratio": round(occupied_area / panel_area, 4),
            })

        panel_speakers = {
            str(bubble.get("speaker_id"))
            for bubble in panel.get("bubbles", [])
            if bubble.get("speaker_id")
        }
        if len(panel_speakers) >= 2 and int(panel.get("visible_face_count", 0) or 0) < 2:
            panels_missing_listener += 1

    missing = sorted(set(transcript) - set(occurrences))
    duplicates = sorted(
        sid for sid, rows in occurrences.items()
        if len({str(row.get("panel_id")) for row in rows}) > 1
        and not all(row.get("fragment") for row in rows)
    )
    for source_ids, fragments in fragment_text.items():
        expected = " ".join(
            _clean_text(transcript[sid].get("text"))
            for sid in source_ids if sid in transcript
        )
        actual = " ".join(row[1] for row in fragments)
        if _clean_text(expected) != _clean_text(actual):
            text_mismatches += 1
            issues.append({
                "type": "dialogue_fragment_text_mismatch",
                "segment_ids": list(source_ids),
            })
    for sid in missing:
        issues.append({"type": "missing_dialogue", "segment_id": sid})
    for sid in duplicates:
        issues.append({"type": "duplicated_dialogue", "segment_id": sid})

    repeated_frame_ids = sorted(frame_id for frame_id, count in repeated_frames.items() if count > 1)
    for frame_id in repeated_frame_ids:
        issues.append({"type": "repeated_frame", "frame_id": frame_id})
    if speaker_visual_unmatched or missing_speaker_ids:
        issues.append({
            "type": "speaker_visual_attribution_needs_review",
            "unmatched_visual_speakers": speaker_visual_unmatched,
            "missing_speaker_ids": missing_speaker_ids,
        })
    if panels_missing_listener:
        issues.append({"type": "conversation_listener_not_visible", "panel_count": panels_missing_listener})

    timestamped_panels = [
        _number(panel.get("timestamp"), -1.0)
        for panel in rendered_panels
        if _number(panel.get("timestamp"), -1.0) >= 0.0
    ]
    ending_time = timestamped_panels[-1] if timestamped_panels else None
    last_dialogue_end = max(
        (_number(segment.get("end"), _number(segment.get("start"))) for segment in transcript.values()),
        default=0.0,
    )
    ending_video_gap = (
        max(0.0, video_duration_s - ending_time)
        if video_duration_s is not None and ending_time is not None
        else None
    )
    ending_preserved = (
        ending_time is not None
        and ending_time + frame_tolerance_s >= last_dialogue_end
        and (
            ending_video_gap is None
            or ending_video_gap <= max(5.0, frame_tolerance_s)
        )
    )
    if not ending_preserved:
        issues.append({"type": "original_ending_not_represented", "last_dialogue_end": last_dialogue_end, "ending_frame_time": ending_time})

    return {
        "status": "pass" if not issues else "review_required",
        "chronological": chronological,
        "transcript_segment_count": len(transcript),
        "segments_placed": len(set(occurrences) & set(transcript)),
        "segments_missing": missing,
        "segments_duplicated": duplicates,
        "dialogue_text_mismatches": text_mismatches,
        "story_plan_text_mismatches": plan_text_mismatches,
        "dialogue_frame_mismatches": frame_mismatches,
        "repeated_frame_ids": repeated_frame_ids,
        "panel_summaries_missing": missing_panel_summaries,
        "scene_summaries_missing": missing_scene_summaries,
        "speaker_visual_unmatched_count": speaker_visual_unmatched,
        "missing_speaker_id_count": missing_speaker_ids,
        "panels_missing_listener": panels_missing_listener,
        "bubble_overlaps": bubble_overlaps,
        "bubbles_outside_panel": bubbles_outside_panel,
        "speaker_tails_missing": missing_speech_tails,
        "speaker_tails_misdirected": incorrect_speech_tails,
        "dialogue_density_review_panels": dense_panels,
        "ending_preserved": ending_preserved,
        "ending_frame_time": ending_time,
        "last_dialogue_end": last_dialogue_end,
        "video_duration_s": video_duration_s,
        "ending_video_gap_s": ending_video_gap,
        "issues": issues,
        "story_beat_count": len(plan.get("events", [])),
        "frame_tolerance_s": frame_tolerance_s,
    }

