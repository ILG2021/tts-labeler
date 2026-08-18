from tts_labeler.models import SubtitleCue
from tts_labeler.subtitle_alignment import align_subtitles_to_document


def test_document_text_replaces_asr_without_changing_timing() -> None:
    raw = [
        SubtitleCue(0, 0.0, 1.2, "Hello word"),
        SubtitleCue(1, 1.5, 2.8, "Good bye"),
    ]
    aligned = align_subtitles_to_document("Hello world! Goodbye.", raw)
    assert [cue.text for cue in aligned] == ["Hello world!", "Goodbye."]
    assert [(cue.start, cue.end) for cue in aligned] == [(0.0, 1.2), (1.5, 2.8)]
    assert [cue.asr_text for cue in aligned] == ["Hello word", "Good bye"]


def test_multilingual_document_alignment() -> None:
    raw = [
        SubtitleCue(0, 0.0, 1.0, "مرحبا العالم"),
        SubtitleCue(1, 1.2, 2.0, "नमस्ते दुनिया"),
    ]
    aligned = align_subtitles_to_document("مرحباً بالعالم؟ नमस्ते दुनिया।", raw)
    assert aligned[0].text.endswith("؟")
    assert aligned[1].text.endswith("।")


def test_unique_anchors_constrain_repeated_text() -> None:
    document = (
        "Chapter one unique opening. The repeated refrain appears here. "
        "Chapter two unique marker. The repeated refrain appears here again."
    )
    raw = [
        SubtitleCue(0, 0.0, 2.0, "chapter one unique opening repeated refrain"),
        SubtitleCue(1, 2.2, 4.5, "chapter two unique marker repeated refrain again"),
    ]
    aligned = align_subtitles_to_document(document, raw)
    assert "Chapter one" in aligned[0].text
    assert "Chapter two" in aligned[1].text
