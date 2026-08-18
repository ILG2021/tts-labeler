from __future__ import annotations

import unicodedata

from .models import DocumentUnit


_STRONG_END = set("。！？!?；;؟؛۔।॥៕៘։\n")
_WEAK_END = set("，,、：:،՝")


def normalize_for_alignment(text: str) -> str:
    """Normalize any Unicode language to letters, marks, and numbers.

    This covers Latin, Cyrillic, Arabic, Indic, CJK, Hangul, Kana and other
    scripts without maintaining a fragile language-specific allow-list.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in text if unicodedata.category(char)[0] in {"L", "M", "N"})


def clean_display_text(text: str) -> str:
    # Preserve the source document exactly enough for training labels. Alignment
    # normalization is intentionally separate and may fold full-width forms.
    text = text.replace("\ufeff", "")
    return " ".join(text.split())


def _raw_units(document: str) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    buffer: list[str] = []
    for char in document.replace("\r\n", "\n").replace("\r", "\n"):
        buffer.append(char)
        if char in _STRONG_END:
            text = clean_display_text("".join(buffer))
            if text:
                units.append((text, "strong"))
            buffer = []
    tail = clean_display_text("".join(buffer))
    if tail:
        units.append((tail, "strong"))
    return units


def _split_long_unit(text: str, max_chars: int) -> list[tuple[str, str]]:
    if len(normalize_for_alignment(text)) <= max_chars:
        return [(text, "strong")]
    pieces: list[tuple[str, str]] = []
    buffer: list[str] = []
    for char in text:
        buffer.append(char)
        if char in _WEAK_END and len(normalize_for_alignment("".join(buffer))) >= max_chars // 2:
            pieces.append((clean_display_text("".join(buffer)), "weak"))
            buffer = []
    tail = clean_display_text("".join(buffer))
    if tail:
        pieces.append((tail, "strong"))
    return pieces or [(text, "strong")]


def split_document(document: str, max_chars: int = 100) -> list[DocumentUnit]:
    result: list[DocumentUnit] = []
    for raw_text, _ in _raw_units(document):
        for text, boundary in _split_long_unit(raw_text, max_chars):
            normalized = normalize_for_alignment(text)
            if normalized:
                result.append(DocumentUnit(len(result), text, normalized, boundary))
    if not result:
        raise ValueError("The document contains no alignable text")
    return result
