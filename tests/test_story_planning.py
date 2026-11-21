from src.story.planning import build_story_plan, validate_story_output


def test_scene_and_panel_summaries_use_existing_visual_and_turn_evidence():
    transcript = [
        {"segment_id": "1", "start": 1.0, "end": 2.0, "text": "What happened?"},
        {"segment_id": "2", "start": 2.1, "end": 3.0, "text": "I left the door open."},
    ]
    panels = [
        {"frame_id": "f1", "t": 1.5, "shot_id": 1, "shot_end": 8.0, "segment_ids": ["1"]},
        {"frame_id": "f2", "t": 2.5, "shot_id": 1, "shot_end": 8.0, "segment_ids": ["2"]},
    ]
    visual_actions = {
        "f1": {"label": "talking", "caption": "Two people talk face to face.", "status": "ok"},
        "f2": {"label": "working", "caption": "A person works with a device.", "status": "ok"},
    }
    speaker_turns = [
        {"speaker_id": "SPEAKER_00", "start": 1.0, "end": 2.0, "confidence": 0.8},
        {"speaker_id": "SPEAKER_01", "start": 2.1, "end": 3.0, "confidence": 0.7},
    ]

    plan = build_story_plan(
        "video",
        panels,
        transcript,
        visual_actions,
        speaker_turns=speaker_turns,
    )

    assert len(plan["scenes"]) == 1
    assert all(event["panel_summary"] for event in plan["events"])
    assert all(event["scene_summary"] for event in plan["events"])
    assert plan["events"][0]["dialogue"][0]["speaker_id"] == "SPEAKER_00"
    assert "question is followed" in plan["scenes"][0]["scene_summary"].lower()
    assert len({event["panel_summary"] for event in plan["events"]}) == 2


def test_dialogue_frame_validation_does_not_accept_any_frame_in_same_shot():
    transcript = {
        "1": {"segment_id": "1", "start": 1.0, "end": 2.0, "text": "Hello."}
    }
    plan = build_story_plan(
        "video",
        [{"frame_id": "f1", "t": 1.5, "shot_id": 1, "shot_end": 20.0, "segment_ids": ["1"]}],
        list(transcript.values()),
        {"f1": {"label": "standing", "caption": "A person is standing.", "status": "ok"}},
    )
    event = plan["events"][0]
    panel = {
        "panel_id": 1,
        "frame_id": "f1",
        "timestamp": 10.0,
        "shot_id": 1,
        "shot_start": 0.0,
        "shot_end": 20.0,
        "scene_id": event["scene_id"],
        "scene_summary": event["scene_summary"],
        "panel_summary": event["panel_summary"],
        "bubbles": [{
            "segment_id": "1",
            "source_segment_ids": ["1"],
            "start": 1.0,
            "end": 2.0,
            "source_text": "Hello.",
            "text": "Hello.",
            "speaker_id": "SPEAKER_00",
            "speaker_face_visible": False,
            "off_panel_speaker": False,
        }],
    }

    validation = validate_story_output(plan, transcript, [panel], frame_tolerance_s=2.0)

    assert validation["dialogue_frame_mismatches"] == 1
    assert any(issue["type"] == "dialogue_frame_mismatch" for issue in validation["issues"])


def test_scene_planner_uses_persistent_visual_changes_and_bounds_summary_spans():
    panels = [
        {"frame_id": f"f{i}", "t": float(i * 10), "shot_id": i, "shot_end": float(i * 10 + 2)}
        for i in range(12)
    ]
    visual_actions = {
        **{
            f"f{i}": {"label": "working", "caption": "A person works with a device.", "status": "ok"}
            for i in range(8)
        },
        **{
            f"f{i}": {"label": "talking", "caption": "Two people talk face to face.", "status": "ok"}
            for i in range(8, 12)
        },
    }

    plan = build_story_plan("video", panels, [], visual_actions, max_scene_span_s=50.0)

    assert len(plan["scenes"]) == 3
    assert [scene["scene_start"] for scene in plan["scenes"]] == [0.0, 50.0, 80.0]
    assert all(scene["scene_summary"] for scene in plan["scenes"])
    assert all(event["panel_summary"] for event in plan["events"])


def test_single_speaker_context_is_not_repeated_in_panel_summaries():
    transcript = [
        {"segment_id": str(i), "start": float(i), "end": float(i + 0.5), "text": "Hello."}
        for i in range(2)
    ]
    panels = [
        {"frame_id": f"f{i}", "t": float(i), "segment_ids": [str(i)]}
        for i in range(2)
    ]
    turns = [{"speaker_id": "SPEAKER_00", "start": 0.0, "end": 2.0}]
    actions = {
        "f0": {"label": "standing", "caption": "A person is standing.", "status": "ok"},
        "f1": {"label": "walking", "caption": "A person is walking.", "status": "ok"},
    }

    plan = build_story_plan("video", panels, transcript, actions, speaker_turns=turns)

    assert all("speaker continues speaking" not in event["panel_summary"].lower() for event in plan["events"])
