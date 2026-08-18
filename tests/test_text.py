from tts_labeler.text import normalize_for_alignment, split_document


def test_normalize_for_alignment() -> None:
    assert normalize_for_alignment("你好，World １２３！") == "你好world123"


def test_normalize_for_alignment_is_multilingual() -> None:
    assert normalize_for_alignment("Привет, мир!") == "приветмир"
    assert normalize_for_alignment("مَرْحَبًا بالعالم؟") == "مَرْحَبًابالعالم"
    assert normalize_for_alignment("नमस्ते दुनिया।") == "नमस्तेदुनिया"


def test_split_document_strong_boundaries() -> None:
    units = split_document("第一句。第二句！\n第三句没有句号")
    assert [unit.text for unit in units] == ["第一句。", "第二句！", "第三句没有句号"]
    assert [unit.normalized for unit in units] == ["第一句", "第二句", "第三句没有句号"]


def test_long_document_unit_splits_at_weak_punctuation() -> None:
    units = split_document("很长的第一部分，后面还有很长的第二部分，最后结束。", max_chars=12)
    assert len(units) >= 2
    assert "".join(unit.text for unit in units) == "很长的第一部分，后面还有很长的第二部分，最后结束。"


def test_multilingual_sentence_boundaries() -> None:
    units = split_document("Hello world! كيف حالك؟ नमस्ते दुनिया।")
    assert [unit.text for unit in units] == ["Hello world!", "كيف حالك؟", "नमस्ते दुनिया।"]
