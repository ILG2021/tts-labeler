from __future__ import annotations

import wave
from array import array
from pathlib import Path

from .models import PipelineConfig


class SileroVAD:
    """Bounded-memory independent VAD used before ASR segmentation."""

    def __init__(self, config: PipelineConfig) -> None:
        try:
            import torch
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD is required by the default industrial mode. "
                "Install with: pip install -e .[vad], or explicitly use --vad off."
            ) from exc
        torch.set_num_threads(1)
        self.torch = torch
        self.get_speech_timestamps = get_speech_timestamps
        self.model = load_silero_vad(onnx=True)
        self.config = config

    def analyze(self, analysis_wav: Path) -> tuple[list[tuple[float, float]], dict]:
        regions: list[tuple[float, float]] = []
        chunks = 0
        peak_chunk_samples = 0
        with wave.open(str(analysis_wav), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise ValueError("Silero analysis WAV must be mono 16-bit PCM")
            rate = handle.getframerate()
            if rate not in {8000, 16000}:
                raise ValueError("Silero VAD requires an 8kHz or 16kHz analysis WAV")
            total_samples = handle.getnframes()
            chunk_samples = max(rate, round(rate * self.config.vad_chunk_duration))
            overlap_samples = round(rate * self.config.vad_chunk_overlap)
            position = 0
            while position < total_samples:
                handle.setpos(position)
                payload = handle.readframes(min(chunk_samples, total_samples - position))
                pcm = array("h")
                pcm.frombytes(payload)
                if not pcm:
                    break
                peak_chunk_samples = max(peak_chunk_samples, len(pcm))
                waveform = self.torch.tensor(pcm, dtype=self.torch.float32) / 32768.0
                timestamps = self.get_speech_timestamps(
                    waveform,
                    self.model,
                    sampling_rate=rate,
                    threshold=self.config.vad_threshold,
                    min_speech_duration_ms=round(
                        self.config.vad_min_speech_duration * 1000
                    ),
                    min_silence_duration_ms=round(self.config.min_silence_duration * 1000),
                    speech_pad_ms=round(self.config.vad_speech_pad * 1000),
                    return_seconds=True,
                )
                offset = position / rate
                regions.extend(
                    (offset + float(item["start"]), offset + float(item["end"]))
                    for item in timestamps
                )
                chunks += 1
                end_position = position + len(pcm)
                if end_position >= total_samples:
                    break
                position = max(position + 1, end_position - overlap_samples)

        merged: list[tuple[float, float]] = []
        for start, end in sorted(regions):
            if not merged or start > merged[-1][1] + 0.05:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged, {
            "backend": "silero",
            "threshold": self.config.vad_threshold,
            "speech_regions": len(merged),
            "speech_seconds": round(sum(end - start for start, end in merged), 3),
            "analysis_chunks": chunks,
            "peak_chunk_seconds": round(
                peak_chunk_samples / rate if peak_chunk_samples else 0.0, 3
            ),
        }
