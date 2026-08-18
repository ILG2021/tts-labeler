from tts_labeler.models import PipelineConfig


def test_config_rejects_invalid_duration_order() -> None:
    try:
        PipelineConfig(min_duration=10.0, target_duration=5.0)
    except ValueError as error:
        assert "min_duration" in str(error)
    else:
        raise AssertionError("Expected invalid duration order to fail")


def test_config_rejects_invalid_vad_threshold() -> None:
    try:
        PipelineConfig(vad_threshold=1.5)
    except ValueError as error:
        assert "vad_threshold" in str(error)
    else:
        raise AssertionError("Expected invalid VAD threshold to fail")


def test_config_rejects_invalid_speaker_ratio() -> None:
    try:
        PipelineConfig(max_foreign_speech_ratio=1.5)
    except ValueError as error:
        assert "max_foreign_speech_ratio" in str(error)
    else:
        raise AssertionError("Expected invalid speaker ratio to fail")
