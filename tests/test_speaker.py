from tts_labeler.speaker import SpeakerTurn, speaker_metrics


def test_speaker_metrics_detect_short_foreign_turn() -> None:
    turns = [
        SpeakerTurn(0.0, 4.0, "main"),
        SpeakerTurn(4.0, 4.4, "other"),
        SpeakerTurn(4.4, 8.0, "main"),
    ]
    metrics = speaker_metrics(turns, "main", 0.0, 8.0)
    assert metrics["speech_seconds"] == 8.0
    assert metrics["foreign_speech_seconds"] == 0.4
    assert metrics["foreign_speech_ratio"] == 0.05


def test_speaker_metrics_count_overlap_as_foreign_speech() -> None:
    turns = [
        SpeakerTurn(0.0, 5.0, "main"),
        SpeakerTurn(2.0, 2.5, "other"),
    ]
    metrics = speaker_metrics(turns, "main", 0.0, 5.0)
    assert metrics["speech_seconds"] == 5.0
    assert metrics["foreign_speech_seconds"] == 0.5
    assert metrics["foreign_speech_ratio"] == 0.1
    assert metrics["overlap_speech_seconds"] == 0.5
