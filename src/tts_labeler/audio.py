from __future__ import annotations

import shutil
import subprocess
import wave
import math
from array import array
from pathlib import Path

from .models import OutputSegment, PipelineConfig


def require_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg was not found on PATH")
    return executable


def normalize_for_analysis(source: Path, target: Path, config: PipelineConfig) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial.wav")
    command = [
        require_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(config.analysis_sample_rate),
        "-c:a",
        "pcm_s16le",
        "-y",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_master(source: Path, target: Path, config: PipelineConfig) -> None:
    """Create the canonical PCM timeline used by analysis and every final cut."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial.wav")
    command = [
        require_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-map_metadata",
        "-1",
        "-vn",
        "-ac",
        str(config.output_channels),
        "-ar",
        str(config.output_sample_rate),
        "-c:a",
        "pcm_s16le",
        "-y",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def analyze_wav_quality(path: Path) -> dict[str, float]:
    total = 0
    sum_samples = 0
    sum_squares = 0
    clipped = 0
    peak = 0
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError("Quality analysis requires 16-bit PCM WAV")
        channels = handle.getnchannels()
        rate = handle.getframerate()
        while payload := handle.readframes(rate):
            samples = array("h")
            samples.frombytes(payload)
            total += len(samples)
            sum_samples += sum(samples)
            sum_squares += sum(sample * sample for sample in samples)
            clipped += sum(abs(sample) >= 32760 for sample in samples)
            if samples:
                peak = max(peak, max(abs(sample) for sample in samples))
    if not total:
        raise ValueError("Exported WAV is empty")
    rms = math.sqrt(sum_squares / total)
    rms_dbfs = 20 * math.log10(rms / 32768.0) if rms else -100.0
    peak_dbfs = 20 * math.log10(peak / 32768.0) if peak else -100.0
    return {
        "duration": total / (rate * channels),
        "rms_dbfs": round(rms_dbfs, 3),
        "peak_dbfs": round(peak_dbfs, 3),
        "clipping_ratio": clipped / total,
        "dc_offset": (sum_samples / total) / 32768.0,
    }


def export_interval(
    source: Path, target: Path, start: float, end: float, config: PipelineConfig
) -> None:
    proxy = OutputSegment(
        index=0,
        text="",
        asr_text="",
        start=start,
        end=end,
        match_score=1.0,
        text_coverage=1.0,
        mean_word_probability=None,
        pause_before=0.0,
        pause_after=0.0,
        accepted=True,
    )
    export_segment(source, target, proxy, config)


def export_segment(
    source: Path, target: Path, segment: OutputSegment, config: PipelineConfig
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.001, segment.end - segment.start)
    temporary = target.with_name(target.name + ".partial.wav")
    command = [
        require_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{segment.start:.4f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.4f}",
        "-ac",
        str(config.output_channels),
        "-ar",
        str(config.output_sample_rate),
        "-c:a",
        "pcm_s16le",
        "-y",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
