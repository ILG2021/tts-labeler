import math
import tempfile
import wave
from array import array
from pathlib import Path

from tts_labeler.acoustic import CutCandidate, _merge_short_pause_boundaries, detect_intervals
from tts_labeler.models import PipelineConfig


def _tone(rate: int, duration: float, frequency: float = 220.0) -> array:
    return array(
        "h",
        (
            round(9000 * math.sin(2 * math.pi * frequency * index / rate))
            for index in range(round(rate * duration))
        ),
    )


def test_acoustic_split_finds_middle_silence() -> None:
    rate = 16000
    samples = _tone(rate, 1.0)
    samples.extend(array("h", [0]) * round(rate * 0.6))
    samples.extend(_tone(rate, 1.0))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(samples.tobytes())
        config = PipelineConfig(
            min_duration=0.4,
            target_duration=1.0,
            max_duration=2.0,
            min_silence_duration=0.3,
            leading_silence=0.05,
            trailing_silence=0.05,
            max_silence_kept=1.0,
        )
        intervals, report = detect_intervals(path, config)
    assert len(intervals) == 2
    assert 1.15 <= intervals[0].end <= 1.4
    assert 1.2 <= intervals[1].start <= 1.45
    assert report["pause_candidates"] == 1


def test_hybrid_vad_rms_removes_middle_of_long_silence() -> None:
    rate = 16000
    samples = _tone(rate, 1.0)
    samples.extend(array("h", [0]) * round(rate * 2.0))
    samples.extend(_tone(rate, 1.0))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "long-pause.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(samples.tobytes())
        config = PipelineConfig(
            min_duration=0.4,
            target_duration=1.0,
            max_duration=3.0,
            min_silence_duration=0.3,
            leading_silence=0.05,
            trailing_silence=0.05,
            max_silence_kept=0.3,
        )
        intervals, report = detect_intervals(
            path, config, speech_regions=[(0.0, 1.0), (3.0, 4.0)]
        )
    assert len(intervals) == 2
    assert intervals[0].end < 1.4
    assert intervals[1].start > 2.5
    assert report["candidate_source"] == "vad+rms"
    assert report["removed_long_silence_seconds"] > 1.0


def test_vad_gap_requires_low_energy_confirmation() -> None:
    rate = 16000
    samples = _tone(rate, 3.0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "continuous-tone.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(samples.tobytes())
        config = PipelineConfig(min_duration=0.4, target_duration=2.0, max_duration=5.0)
        intervals, report = detect_intervals(
            path, config, speech_regions=[(0.0, 1.0), (2.0, 3.0)]
        )
    assert len(intervals) == 1
    assert report["pause_candidates"] == 0
    assert report["rejected_vad_gaps"] == 1


def test_segmentation_enforces_max_duration_without_pauses() -> None:
    rate = 16000
    samples = _tone(rate, 25.0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "long-continuous.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(samples.tobytes())
        config = PipelineConfig(
            min_duration=2.0,
            target_duration=6.0,
            max_duration=8.0,
            vad_backend="off",
        )
        intervals, report = detect_intervals(path, config)
    assert len(intervals) >= 4
    assert all(interval.duration <= 8.001 for interval in intervals)
    assert all(interval.duration >= 2.0 for interval in intervals)
    assert report["fallback_boundaries"] >= 3


def test_vad_bounds_trim_leading_and_trailing_silence() -> None:
    rate = 16000
    samples = array("h", [0]) * rate
    samples.extend(_tone(rate, 2.0))
    samples.extend(array("h", [0]) * rate)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "padded.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(samples.tobytes())
        config = PipelineConfig(
            min_duration=0.5,
            target_duration=2.0,
            max_duration=4.0,
            leading_silence=0.08,
            trailing_silence=0.08,
        )
        intervals, report = detect_intervals(path, config, speech_regions=[(1.0, 3.0)])
    assert len(intervals) == 1
    assert 0.90 <= intervals[0].start <= 0.95
    assert 3.05 <= intervals[0].end <= 3.10
    assert report["trimmed_leading_seconds"] > 0.9
    assert report["trimmed_trailing_seconds"] > 0.9


def test_pause_padding_is_retained_without_crossing_adjacent_speech() -> None:
    rate = 16000
    samples = _tone(rate, 1.0)
    samples.extend(array("h", [0]) * round(rate * 0.4))
    samples.extend(_tone(rate, 1.0))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "short-pause.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(samples.tobytes())
        config = PipelineConfig(
            min_duration=0.4,
            target_duration=1.0,
            max_duration=2.0,
            min_silence_duration=0.3,
            leading_silence=0.2,
            trailing_silence=0.2,
            vad_backend="off",
        )
        intervals, report = detect_intervals(path, config)
    assert len(intervals) == 2
    assert 1.15 <= intervals[0].end <= 1.401
    assert 0.999 <= intervals[1].start <= 1.25
    assert report["leading_silence_seconds"] == 0.2
    assert report["trailing_silence_seconds"] == 0.2


def test_short_pause_boundary_is_merged_when_duration_allows() -> None:
    config = PipelineConfig(min_duration=1.0, target_duration=8.0, max_duration=18.0)
    selected = [
        CutCandidate(0.0, 0.0, 0.0, -100.0, "start"),
        CutCandidate(7.0, 6.9, 7.1, -50.0, "fallback"),
        CutCandidate(14.0, 14.0, 14.0, -100.0, "end"),
    ]
    merged, merged_count, forced_count = _merge_short_pause_boundaries(selected, config)
    assert [item.time for item in merged] == [0.0, 14.0]
    assert merged_count == 1
    assert forced_count == 0


def test_short_pause_boundary_is_forced_when_merge_would_be_too_long() -> None:
    config = PipelineConfig(min_duration=1.0, target_duration=8.0, max_duration=12.0)
    selected = [
        CutCandidate(0.0, 0.0, 0.0, -100.0, "start"),
        CutCandidate(8.0, 7.9, 8.1, -50.0, "fallback"),
        CutCandidate(16.0, 16.0, 16.0, -100.0, "end"),
    ]
    merged, merged_count, forced_count = _merge_short_pause_boundaries(selected, config)
    assert [item.time for item in merged] == [0.0, 8.0, 16.0]
    assert merged_count == 0
    assert forced_count == 1
