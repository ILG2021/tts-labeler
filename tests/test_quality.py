from tts_labeler.quality import transcript_quality


def test_transcript_quality_detects_repetition() -> None:
    metrics = transcript_quality("谢谢谢谢谢谢谢谢谢谢谢谢", 2.0)
    assert metrics["repetition_ratio"] > 0.6


def test_transcript_quality_reports_character_rate() -> None:
    metrics = transcript_quality("hello world", 2.0)
    assert metrics["normalized_characters"] == 10
    assert metrics["characters_per_second"] == 5.0
