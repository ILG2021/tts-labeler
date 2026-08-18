from __future__ import annotations

import logging
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import PipelineConfig
from .audio import require_ffmpeg


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def _duration_by_speaker(turns: list[SpeakerTurn]) -> dict[str, float]:
    result: dict[str, float] = {}
    for turn in turns:
        result[turn.speaker] = result.get(turn.speaker, 0.0) + max(0.0, turn.end - turn.start)
    return result


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    result: list[tuple[float, float]] = []
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            result.append((current_start, current_end))
            current_start, current_end = start, end
    result.append((current_start, current_end))
    return result


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _intersection_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    total = 0.0
    for left_start, left_end in _merge_intervals(left):
        for right_start, right_end in _merge_intervals(right):
            total += max(0.0, min(left_end, right_end) - max(left_start, right_start))
    return total


def speaker_metrics(
    turns: list[SpeakerTurn], target_speaker: str, start: float, end: float
) -> dict[str, float | str]:
    clipped = [
        (max(start, turn.start), min(end, turn.end), turn.speaker)
        for turn in turns
        if turn.end > start and turn.start < end
    ]
    speech = _union_duration([(a, b) for a, b, _ in clipped if b > a])
    target_intervals = [(a, b) for a, b, label in clipped if label == target_speaker and b > a]
    foreign_intervals = [(a, b) for a, b, label in clipped if label != target_speaker and b > a]
    target = _union_duration(target_intervals)
    foreign = _union_duration(foreign_intervals)
    overlap = _intersection_duration(target_intervals, foreign_intervals)
    return {
        "target_speaker": target_speaker,
        "speech_seconds": round(speech, 4),
        "target_speech_seconds": round(target, 4),
        "foreign_speech_seconds": round(foreign, 4),
        "overlap_speech_seconds": round(overlap, 4),
        "target_speech_ratio": round(target / speech, 4) if speech else 0.0,
        "foreign_speech_ratio": round(foreign / speech, 4) if speech else 0.0,
        "overlap_speech_ratio": round(overlap / speech, 4) if speech else 0.0,
    }


def _cosine(left: Any, right: Any) -> float:
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else -1.0


class PyannoteSpeakerAnalyzer:
    def __init__(self, config: PipelineConfig) -> None:
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "pyannote speaker filtering is enabled but unavailable. "
                "Install with: pip install -e .[speaker]"
            ) from exc
        token = os.environ.get(config.speaker_token_env)
        try:
            self.pipeline = Pipeline.from_pretrained(config.speaker_model, token=token)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {config.speaker_model}. Accept its Hugging Face terms and set "
                f"{config.speaker_token_env}, or use --speaker-backend off."
            ) from exc
        if config.device != "cpu" and torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))
        self.torch = torch
        self.config = config

    def _waveform_input(self, audio: Path) -> dict[str, Any]:
        """Decode through FFmpeg and pass a waveform dict to pyannote.

        pyannote.audio 4.x may select torchcodec for path inputs. A successful
        `import torchcodec` does not guarantee that its native FFmpeg decoder
        can load the installed runtime, so this deliberately avoids path input.
        """
        command = [
            require_ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "FFmpeg is required for speaker analysis fallback but was not found on PATH"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg could not decode speaker audio {audio}: {detail}") from exc
        samples = self.torch.frombuffer(
            memoryview(result.stdout), dtype=self.torch.int16
        ).clone()
        waveform = samples.to(dtype=self.torch.float32).div_(32768.0).reshape(1, -1)
        if waveform.shape[1] == 0:
            raise RuntimeError(f"Speaker audio is empty: {audio}")
        return {"waveform": waveform, "sample_rate": 16000}

    @staticmethod
    def _unpack(output: Any) -> tuple[list[SpeakerTurn], dict[str, Any]]:
        annotation = getattr(output, "speaker_diarization", output)
        turns = [
            SpeakerTurn(float(segment.start), float(segment.end), str(label))
            for segment, _, label in annotation.itertracks(yield_label=True)
        ]
        labels = [str(label) for label in annotation.labels()]
        raw_embeddings = getattr(output, "speaker_embeddings", None)
        embeddings: dict[str, Any] = {}
        if raw_embeddings is not None and len(raw_embeddings) == len(labels):
            embeddings = dict(zip(labels, raw_embeddings))
        return turns, embeddings

    def _run(self, audio: Path) -> tuple[list[SpeakerTurn], dict[str, Any]]:
        return self._unpack(self.pipeline(self._waveform_input(audio)))

    def analyze(
        self, audio: Path, reference: Path | None = None
    ) -> tuple[list[SpeakerTurn], str, dict[str, Any]]:
        turns, embeddings = self._run(audio)
        if not turns:
            raise RuntimeError("pyannote found no speaker turns")
        durations = _duration_by_speaker(turns)
        similarities: dict[str, float] = {}
        if reference is None:
            target = max(durations, key=durations.get)
            selection = "dominant"
        else:
            reference_turns, reference_embeddings = self._run(reference)
            if not reference_turns or not reference_embeddings or not embeddings:
                raise RuntimeError("The pyannote output did not provide embeddings for reference matching")
            reference_durations = _duration_by_speaker(reference_turns)
            reference_label = max(reference_durations, key=reference_durations.get)
            reference_embedding = reference_embeddings[reference_label]
            similarities = {
                label: _cosine(embedding, reference_embedding)
                for label, embedding in embeddings.items()
            }
            target = max(similarities, key=similarities.get)
            if similarities[target] < self.config.min_speaker_similarity:
                raise RuntimeError(
                    f"No source speaker matches the reference (best={similarities[target]:.3f})"
                )
            selection = "reference"
        report = {
            "backend": "pyannote",
            "model": self.config.speaker_model,
            "selection": selection,
            "target_speaker": target,
            "speaker_durations": {key: round(value, 4) for key, value in durations.items()},
            "reference_similarities": {
                key: round(value, 4) for key, value in similarities.items()
            },
            "turns": [asdict(turn) for turn in turns],
        }
        LOGGER.info("Speaker diarization found %d speakers; target=%s", len(durations), target)
        return turns, target, report
