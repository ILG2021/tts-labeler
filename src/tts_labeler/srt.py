from __future__ import annotations

import re
from pathlib import Path

from .models import SubtitleCue


_TIMING_RE = re.compile(
    r"(?P<sh>\d{2,}):(?P<sm>\d{2}):(?P<ss>\d{2})[,.](?P<sms>\d{3})\s*-->\s*"
    r"(?P<eh>\d{2,}):(?P<em>\d{2}):(?P<es>\d{2})[,.](?P<ems>\d{3})"
)


def _seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def format_timestamp(value: float) -> str:
    millis = max(0, round(value * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    seconds, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def dumps(cues: list[SubtitleCue]) -> str:
    blocks: list[str] = []
    for position, cue in enumerate(cues, 1):
        blocks.append(
            f"{position}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n"
            f"{cue.text.strip()}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write(path: Path, cues: list[SubtitleCue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(dumps(cues), encoding="utf-8-sig", newline="\n")
    temporary.replace(path)


def loads(content: str) -> list[SubtitleCue]:
    content = content.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", content.strip()) if content.strip() else []
    cues: list[SubtitleCue] = []
    for block in blocks:
        lines = block.splitlines()
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        match = _TIMING_RE.search(lines[timing_index])
        if not match:
            raise ValueError(f"Invalid SRT timing line: {lines[timing_index]}")
        groups = match.groupdict()
        start = _seconds(groups["sh"], groups["sm"], groups["ss"], groups["sms"])
        end = _seconds(groups["eh"], groups["em"], groups["es"], groups["ems"])
        if end <= start:
            raise ValueError(f"SRT cue end must be after start: {lines[timing_index]}")
        text = "\n".join(lines[timing_index + 1 :]).strip()
        cues.append(SubtitleCue(len(cues), start, end, text))
    for previous, current in zip(cues, cues[1:]):
        if current.start < previous.start:
            raise ValueError("SRT cues must be ordered by start time")
    return cues


def read(path: Path) -> list[SubtitleCue]:
    return loads(path.read_text(encoding="utf-8-sig"))


def validate_timeline(
    cues: list[SubtitleCue], audio_duration: float, *, max_overlap: float = 0.0
) -> None:
    if audio_duration <= 0:
        raise ValueError("Audio duration must be positive")
    for index, cue in enumerate(cues):
        if cue.start < 0 or cue.end > audio_duration + 0.002:
            raise ValueError(
                f"SRT cue {index + 1} lies outside audio duration: {cue.start:.3f}..{cue.end:.3f}"
            )
        if index and cue.start < cues[index - 1].end - max_overlap - 0.002:
            overlap = cues[index - 1].end - cue.start
            raise ValueError(f"SRT cues {index} and {index + 1} overlap by {overlap:.3f}s")
