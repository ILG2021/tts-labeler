from tts_labeler.models import SubtitleCue
from tts_labeler.srt import dumps, loads, validate_timeline


def test_srt_round_trip_multilingual() -> None:
    cues = [
        SubtitleCue(0, 1.234, 3.456, "你好，世界！"),
        SubtitleCue(1, 4.0, 5.25, "Hello world!\nمرحبا"),
    ]
    restored = loads(dumps(cues))
    assert [(cue.start, cue.end, cue.text) for cue in restored] == [
        (1.234, 3.456, "你好，世界！"),
        (4.0, 5.25, "Hello world!\nمرحبا"),
    ]


def test_srt_rejects_reversed_timing() -> None:
    try:
        loads("1\n00:00:02,000 --> 00:00:01,000\nbad\n")
    except ValueError as error:
        assert "end must be after start" in str(error)
    else:
        raise AssertionError("Expected invalid timing to fail")


def test_srt_timeline_rejects_excess_overlap_and_out_of_bounds() -> None:
    overlapping = [
        SubtitleCue(0, 0.0, 2.0, "one"),
        SubtitleCue(1, 1.0, 3.0, "two"),
    ]
    try:
        validate_timeline(overlapping, 4.0, max_overlap=0.1)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("Expected overlapping cues to fail")
    try:
        validate_timeline([SubtitleCue(0, 0.0, 5.0, "bad")], 4.0)
    except ValueError as error:
        assert "outside audio" in str(error)
    else:
        raise AssertionError("Expected out-of-bounds cue to fail")
