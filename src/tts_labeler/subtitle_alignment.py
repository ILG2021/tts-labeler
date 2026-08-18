from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from difflib import SequenceMatcher

from .models import SubtitleCue
from .text import normalize_for_alignment


def _normalize_with_source_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    source: list[int] = []
    for source_index, char in enumerate(text):
        normalized = normalize_for_alignment(char)
        chars.extend(normalized)
        source.extend([source_index] * len(normalized))
    return "".join(chars), source


def _unique_ngram_positions(text: str, size: int) -> dict[str, int]:
    first: dict[str, int] = {}
    repeated: set[str] = set()
    for index in range(max(0, len(text) - size + 1)):
        token = text[index : index + size]
        if token in first:
            repeated.add(token)
        else:
            first[token] = index
    return {token: index for token, index in first.items() if token not in repeated}


def _monotonic_anchors(asr: str, document: str, size: int = 12) -> list[tuple[int, int]]:
    if min(len(asr), len(document)) < size * 2:
        return []
    doc_unique = _unique_ngram_positions(document, size)
    asr_unique = _unique_ngram_positions(asr, size)
    pairs = sorted(
        (asr_pos, doc_unique[token])
        for token, asr_pos in asr_unique.items()
        if token in doc_unique
    )
    if not pairs:
        return []

    tails: list[int] = []
    tail_indices: list[int] = []
    previous = [-1] * len(pairs)
    for index, (_, doc_pos) in enumerate(pairs):
        slot = bisect_left(tails, doc_pos)
        if slot == len(tails):
            tails.append(doc_pos)
            tail_indices.append(index)
        else:
            tails[slot] = doc_pos
            tail_indices[slot] = index
        if slot:
            previous[index] = tail_indices[slot - 1]

    chain: list[tuple[int, int]] = []
    cursor = tail_indices[-1]
    while cursor >= 0:
        chain.append(pairs[cursor])
        cursor = previous[cursor]
    chain.reverse()

    # Remove overlapping anchors; dense adjacent n-grams carry no extra value.
    spaced: list[tuple[int, int]] = []
    for asr_pos, doc_pos in chain:
        if not spaced or (
            asr_pos >= spaced[-1][0] + size and doc_pos >= spaced[-1][1] + size
        ):
            spaced.append((asr_pos, doc_pos))
    return spaced


def _matching_pairs(asr: str, document: str) -> list[tuple[int, int]]:
    """Match inside anchor-bounded windows to limit drift and worst-case work."""
    anchor_size = 12
    anchors = _monotonic_anchors(asr, document, anchor_size)
    if len(anchors) < 2:
        matcher = SequenceMatcher(None, asr, document, autojunk=False)
        return [
            (block.a + offset, block.b + offset)
            for block in matcher.get_matching_blocks()
            for offset in range(block.size)
        ]

    pairs: list[tuple[int, int]] = []
    asr_cursor = 0
    doc_cursor = 0
    for asr_anchor, doc_anchor in anchors:
        matcher = SequenceMatcher(
            None,
            asr[asr_cursor:asr_anchor],
            document[doc_cursor:doc_anchor],
            autojunk=False,
        )
        for block in matcher.get_matching_blocks():
            pairs.extend(
                (asr_cursor + block.a + offset, doc_cursor + block.b + offset)
                for offset in range(block.size)
            )
        pairs.extend(
            (asr_anchor + offset, doc_anchor + offset) for offset in range(anchor_size)
        )
        asr_cursor = asr_anchor + anchor_size
        doc_cursor = doc_anchor + anchor_size
    matcher = SequenceMatcher(None, asr[asr_cursor:], document[doc_cursor:], autojunk=False)
    for block in matcher.get_matching_blocks():
        pairs.extend(
            (asr_cursor + block.a + offset, doc_cursor + block.b + offset)
            for offset in range(block.size)
        )
    return pairs


def align_subtitles_to_document(document: str, cues: list[SubtitleCue]) -> list[SubtitleCue]:
    if not cues:
        raise ValueError("The SRT contains no cues")
    doc_norm, doc_source = _normalize_with_source_map(document)
    if not doc_norm:
        raise ValueError("The document contains no alignable text")

    asr_chars: list[str] = []
    asr_owner: list[int] = []
    cue_norms: list[str] = []
    for cue_index, cue in enumerate(cues):
        normalized = normalize_for_alignment(cue.text)
        cue_norms.append(normalized)
        asr_chars.extend(normalized)
        asr_owner.extend([cue_index] * len(normalized))
    asr_norm = "".join(asr_chars)
    if not asr_norm:
        raise ValueError("The ASR SRT contains no alignable text")

    doc_positions: dict[int, list[int]] = defaultdict(list)
    matched_count: dict[int, int] = defaultdict(int)
    for asr_position, doc_position in _matching_pairs(asr_norm, doc_norm):
        cue_index = asr_owner[asr_position]
        doc_positions[cue_index].append(doc_position)
        matched_count[cue_index] += 1

    first = [min(doc_positions[index]) if doc_positions[index] else None for index in range(len(cues))]
    last = [max(doc_positions[index]) if doc_positions[index] else None for index in range(len(cues))]
    prefix_last: list[int | None] = []
    running_last: int | None = None
    for value in last:
        if value is not None:
            running_last = value if running_last is None else max(running_last, value)
        prefix_last.append(running_last)
    suffix_first: list[int | None] = [None] * len(cues)
    running_first: int | None = None
    for index in range(len(cues) - 1, -1, -1):
        value = first[index]
        if value is not None:
            running_first = value if running_first is None else min(running_first, value)
        suffix_first[index] = running_first

    boundaries = [0]
    total_asr_chars = max(1, len(asr_norm))
    cumulative_asr = 0
    for index in range(1, len(cues)):
        cumulative_asr += len(cue_norms[index - 1])
        left = prefix_last[index - 1]
        right = suffix_first[index]
        if left is not None and right is not None:
            boundary = max(left + 1, right)
        elif left is not None:
            boundary = left + 1
        elif right is not None:
            boundary = right
        else:
            boundary = round(len(doc_norm) * cumulative_asr / total_asr_chars)
        boundaries.append(max(boundaries[-1], min(len(doc_norm), boundary)))
    boundaries.append(len(doc_norm))

    source_boundaries = [0]
    for boundary in boundaries[1:-1]:
        source_boundaries.append(doc_source[boundary] if boundary < len(doc_source) else len(document))
    source_boundaries.append(len(document))

    aligned: list[SubtitleCue] = []
    for index, cue in enumerate(cues):
        text = document[source_boundaries[index] : source_boundaries[index + 1]].strip()
        assigned_norm = normalize_for_alignment(text)
        similarity = SequenceMatcher(None, cue_norms[index], assigned_norm, autojunk=False).ratio()
        coverage = matched_count[index] / max(1, len(assigned_norm))
        aligned.append(
            SubtitleCue(
                index=index,
                start=cue.start,
                end=cue.end,
                text=text,
                asr_text=cue.text,
                match_score=similarity,
                text_coverage=min(1.0, coverage),
            )
        )
    return aligned
