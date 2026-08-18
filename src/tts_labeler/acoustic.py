from __future__ import annotations

import math
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

from .models import PipelineConfig


@dataclass(frozen=True)
class AcousticInterval:
    start: float
    end: float
    pause_before: float = 0.0
    pause_after: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class CutCandidate:
    time: float
    pause_start: float
    pause_end: float
    level_dbfs: float
    kind: str

    @property
    def pause_duration(self) -> float:
        return max(0.0, self.pause_end - self.pause_start)


def _dbfs(samples: array) -> float:
    if not samples:
        return -100.0
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    if mean_square <= 0:
        return -100.0
    return 20 * math.log10(math.sqrt(mean_square) / 32768.0)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return -100.0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _threshold(levels: list[float], config: PipelineConfig) -> tuple[float, float, float, float]:
    noise_floor = _percentile(levels, 0.20)
    speech_reference = _percentile(levels, 0.80)
    dynamic_range = speech_reference - noise_floor
    if dynamic_range >= 6.0:
        threshold = min(noise_floor + config.silence_margin_db, speech_reference - 6.0)
    else:
        threshold = min(config.absolute_silence_dbfs, speech_reference - 3.0)
    return noise_floor, speech_reference, dynamic_range, threshold


def read_pcm16_mono(path: Path) -> tuple[int, array]:
    """Compatibility helper for bounded callers; long-form analysis streams instead."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("Analysis WAV must be mono 16-bit PCM")
        rate = handle.getframerate()
        samples = array("h")
        samples.frombytes(handle.readframes(handle.getnframes()))
    return rate, samples


def _stream_levels(path: Path, frame_ms: int) -> tuple[int, int, int, list[float]]:
    levels: list[float] = []
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("Analysis WAV must be mono 16-bit PCM")
        rate = handle.getframerate()
        total_samples = handle.getnframes()
        frame_size = max(1, round(rate * frame_ms / 1000))
        while True:
            payload = handle.readframes(frame_size)
            if not payload:
                break
            samples = array("h")
            samples.frombytes(payload)
            levels.append(_dbfs(samples))
    return rate, total_samples, frame_size, levels


def _silence_runs(levels: list[float], threshold: float) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, level in enumerate(levels):
        if level <= threshold and start is None:
            start = index
        elif level > threshold and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(levels)))
    return runs


def _nearest_zero(path: Path, target: int, radius: int, total_samples: int) -> int:
    lower = max(1, target - radius)
    upper = min(total_samples - 1, target + radius)
    if upper <= lower:
        return max(0, min(total_samples - 1, target))
    with wave.open(str(path), "rb") as handle:
        handle.setpos(lower - 1)
        samples = array("h")
        samples.frombytes(handle.readframes(upper - lower + 2))
    local_target = max(1, min(len(samples) - 1, target - (lower - 1)))
    best = local_target
    best_amplitude = abs(samples[local_target])
    for index in range(1, len(samples)):
        crosses = (samples[index - 1] <= 0 < samples[index]) or (
            samples[index - 1] >= 0 > samples[index]
        )
        amplitude = abs(samples[index])
        if crosses and amplitude < best_amplitude:
            best, best_amplitude = index, amplitude
    return lower - 1 + best


def _quiet_frame(levels: list[float], start: int, end: int) -> int:
    start = max(0, min(len(levels) - 1, start))
    end = max(start + 1, min(len(levels), end))
    minimum = min(levels[start:end])
    midpoint = (start + end - 1) / 2
    near_minimum = [index for index in range(start, end) if levels[index] <= minimum + 0.5]
    return min(near_minimum, key=lambda index: abs(index - midpoint))


def _make_candidate(
    path: Path,
    levels: list[float],
    frame_start: int,
    frame_end: int,
    frame_size: int,
    frame_seconds: float,
    rate: int,
    total_samples: int,
    kind: str,
) -> CutCandidate:
    frame = _quiet_frame(levels, frame_start, frame_end)
    target = min(total_samples - 1, frame * frame_size + frame_size // 2)
    sample = _nearest_zero(path, target, max(1, round(rate * 0.005)), total_samples)
    return CutCandidate(
        time=sample / rate,
        pause_start=frame_start * frame_seconds,
        pause_end=min(total_samples / rate, frame_end * frame_seconds),
        level_dbfs=levels[frame],
        kind=kind,
    )


def _fallback_candidates(
    path: Path,
    levels: list[float],
    start: float,
    end: float,
    frame_size: int,
    frame_seconds: float,
    rate: int,
    total_samples: int,
    config: PipelineConfig,
) -> list[CutCandidate]:
    result: list[CutCandidate] = []
    effective_max = max(config.min_duration, config.max_duration - 2 * config.boundary_padding)
    effective_target = min(config.target_duration, effective_max)
    search_radius = min(
        config.max_boundary_search,
        max(0.0, (effective_max - effective_target) / 2),
    )
    ideal = start + effective_target
    while end - ideal >= config.min_duration:
        lower = max(start + config.min_duration, ideal - search_radius)
        upper = min(end - config.min_duration, ideal + search_radius)
        first = max(0, math.floor(lower / frame_seconds))
        last = min(len(levels), max(first + 1, math.ceil(upper / frame_seconds)))
        frame = _quiet_frame(levels, first, last)
        target = min(total_samples - 1, frame * frame_size + frame_size // 2)
        sample = _nearest_zero(path, target, max(1, round(rate * 0.005)), total_samples)
        result.append(
            CutCandidate(sample / rate, sample / rate, sample / rate, levels[frame], "fallback")
        )
        ideal += effective_target
    return result


def _deduplicate_candidates(candidates: list[CutCandidate], tolerance: float) -> list[CutCandidate]:
    result: list[CutCandidate] = []
    priority = {"end": 3, "pause": 2, "fallback": 1, "start": 3}
    for candidate in sorted(candidates, key=lambda item: item.time):
        if result and candidate.time - result[-1].time <= tolerance:
            if priority[candidate.kind] > priority[result[-1].kind]:
                result[-1] = candidate
        else:
            result.append(candidate)
    return result


def _select_boundaries(
    candidates: list[CutCandidate], config: PipelineConfig
) -> list[CutCandidate]:
    """Shortest-path segmentation with hard duration constraints."""
    count = len(candidates)
    effective_max = max(config.min_duration, config.max_duration - 2 * config.boundary_padding)
    effective_target = min(config.target_duration, effective_max)
    costs = [math.inf] * count
    previous: list[int | None] = [None] * count
    costs[0] = 0.0
    for index in range(1, count):
        current = candidates[index]
        for prior in range(index - 1, -1, -1):
            duration = current.time - candidates[prior].time
            if duration > effective_max + 1e-6:
                break
            if duration < config.min_duration - 1e-6 and not (
                prior == 0 and index == count - 1
            ):
                continue
            if not math.isfinite(costs[prior]):
                continue
            duration_cost = ((duration - effective_target) / effective_target) ** 2
            segment_cost = 0.18 + duration_cost
            if current.kind == "pause":
                segment_cost -= min(0.7, current.pause_duration * 0.45)
            elif current.kind == "fallback":
                segment_cost += 1.5
            total = costs[prior] + segment_cost
            if total < costs[index]:
                costs[index] = total
                previous[index] = prior
    if previous[-1] is None and count > 1:
        raise RuntimeError("No segmentation path satisfies the configured duration constraints")
    selected: list[CutCandidate] = []
    cursor: int | None = count - 1
    while cursor is not None:
        selected.append(candidates[cursor])
        cursor = previous[cursor]
    return list(reversed(selected))


def detect_intervals(
    path: Path,
    config: PipelineConfig,
    *,
    speech_regions: list[tuple[float, float]] | None = None,
) -> tuple[list[AcousticInterval], dict]:
    rate, total_samples, frame_size, levels = _stream_levels(path, config.analysis_frame_ms)
    if not levels or total_samples <= 0:
        raise ValueError("The analysis audio is empty")
    frame_seconds = frame_size / rate
    duration = total_samples / rate
    noise_floor, speech_reference, dynamic_range, threshold = _threshold(levels, config)
    min_frames = max(1, math.ceil(config.min_silence_duration / frame_seconds))
    energy_runs = [
        run for run in _silence_runs(levels, threshold) if run[1] - run[0] >= min_frames
    ]

    # Trim only confidently non-speech leading/trailing material, retaining the
    # configured boundary padding around the first and last detected speech.
    if speech_regions:
        timeline_start = max(0.0, speech_regions[0][0] - config.boundary_padding)
        timeline_end = min(duration, speech_regions[-1][1] + config.boundary_padding)
    else:
        active = [index for index, level in enumerate(levels) if level > threshold]
        if not active:
            raise ValueError("No active audio was found")
        timeline_start = max(0.0, active[0] * frame_seconds - config.boundary_padding)
        timeline_end = min(duration, (active[-1] + 1) * frame_seconds + config.boundary_padding)
    if timeline_end <= timeline_start:
        raise ValueError("Detected speech interval is empty")

    candidate_ranges: list[tuple[int, int]] = []
    rejected_vad_gaps = 0
    if speech_regions is not None:
        context_frames = max(1, round(5.0 / frame_seconds))
        for (_, speech_end), (next_start, _) in zip(speech_regions, speech_regions[1:]):
            if next_start - speech_end < config.min_silence_duration:
                continue
            frame_start = max(0, math.floor(speech_end / frame_seconds))
            frame_end = min(len(levels), math.ceil(next_start / frame_seconds))
            local_start = max(0, frame_start - context_frames)
            local_end = min(len(levels), frame_end + context_frames)
            *_, local_threshold = _threshold(levels[local_start:local_end], config)
            low_energy = sum(level <= local_threshold for level in levels[frame_start:frame_end])
            if low_energy < max(1, round((frame_end - frame_start) * 0.20)):
                rejected_vad_gaps += 1
                continue
            candidate_ranges.append((frame_start, frame_end))
    else:
        candidate_ranges = energy_runs

    pause_candidates = [
        _make_candidate(
            path,
            levels,
            frame_start,
            frame_end,
            frame_size,
            frame_seconds,
            rate,
            total_samples,
            "pause",
        )
        for frame_start, frame_end in candidate_ranges
        if frame_start > 0
        and frame_end < len(levels)
        and timeline_start + config.min_duration
        <= ((frame_start + frame_end) / 2) * frame_seconds
        <= timeline_end - config.min_duration
    ]
    fallbacks = _fallback_candidates(
        path,
        levels,
        timeline_start,
        timeline_end,
        frame_size,
        frame_seconds,
        rate,
        total_samples,
        config,
    )
    all_candidates = [
        CutCandidate(timeline_start, timeline_start, timeline_start, -100.0, "start"),
        *pause_candidates,
        *fallbacks,
        CutCandidate(timeline_end, timeline_end, timeline_end, -100.0, "end"),
    ]
    all_candidates = _deduplicate_candidates(all_candidates, frame_seconds)
    selected = _select_boundaries(all_candidates, config)

    boundary_pairs: list[tuple[float, float]] = [(timeline_start, timeline_start)]
    selected_pause_count = 0
    fallback_count = 0
    for candidate in selected[1:-1]:
        if candidate.kind != "pause" or candidate.pause_duration <= config.max_silence_kept:
            boundary_pairs.append((candidate.time, candidate.time))
            fallback_count += candidate.kind == "fallback"
            selected_pause_count += candidate.kind == "pause"
            continue
        selected_pause_count += 1
        left_end = min(candidate.pause_end, candidate.pause_start + config.max_silence_kept)
        right_start = max(candidate.pause_start, candidate.pause_end - config.max_silence_kept)
        left_frame = _quiet_frame(
            levels,
            math.floor(candidate.pause_start / frame_seconds),
            math.ceil(left_end / frame_seconds),
        )
        right_frame = _quiet_frame(
            levels,
            math.floor(right_start / frame_seconds),
            math.ceil(candidate.pause_end / frame_seconds),
        )
        left_sample = _nearest_zero(
            path,
            min(total_samples - 1, left_frame * frame_size + frame_size // 2),
            max(1, round(rate * 0.005)),
            total_samples,
        )
        right_sample = _nearest_zero(
            path,
            min(total_samples - 1, right_frame * frame_size + frame_size // 2),
            max(1, round(rate * 0.005)),
            total_samples,
        )
        boundary_pairs.append(
            (min(left_sample, right_sample) / rate, max(left_sample, right_sample) / rate)
        )
    boundary_pairs.append((timeline_end, timeline_end))

    intervals: list[AcousticInterval] = []
    for index in range(len(boundary_pairs) - 1):
        previous_right = boundary_pairs[index][1]
        next_left = boundary_pairs[index + 1][0]
        start = max(timeline_start, previous_right - (config.boundary_padding if index else 0))
        end = min(
            timeline_end,
            next_left + (config.boundary_padding if index + 1 < len(boundary_pairs) - 1 else 0),
        )
        if end <= start:
            raise RuntimeError("Segmentation produced an invalid interval")
        intervals.append(
            AcousticInterval(
                start,
                end,
                pause_before=max(0.0, boundary_pairs[index][1] - boundary_pairs[index][0]),
                pause_after=max(0.0, boundary_pairs[index + 1][1] - boundary_pairs[index + 1][0]),
            )
        )

    return intervals, {
        "duration": duration,
        "analyzed_frames": len(levels),
        "sample_rate": rate,
        "frame_ms": config.analysis_frame_ms,
        "noise_floor_dbfs": round(noise_floor, 2),
        "speech_reference_dbfs": round(speech_reference, 2),
        "dynamic_range_db": round(dynamic_range, 2),
        "silence_threshold_dbfs": round(threshold, 2),
        "pause_candidates": len(pause_candidates),
        "selected_pause_boundaries": selected_pause_count,
        "fallback_boundaries": fallback_count,
        "candidate_source": "vad+rms" if speech_regions is not None else "rms-only",
        "rejected_vad_gaps": rejected_vad_gaps,
        "trimmed_leading_seconds": round(timeline_start, 3),
        "trimmed_trailing_seconds": round(duration - timeline_end, 3),
        "removed_long_silence_seconds": round(
            sum(max(0.0, right - left) for left, right in boundary_pairs), 3
        ),
    }
