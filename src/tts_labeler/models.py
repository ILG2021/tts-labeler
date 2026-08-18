from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    probability: float | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid word timing: {self.start}..{self.end}")


@dataclass
class SubtitleCue:
    index: int
    start: float
    end: float
    text: str
    asr_text: str | None = None
    match_score: float = 1.0
    text_coverage: float = 1.0
    mean_asr_probability: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class DocumentUnit:
    index: int
    text: str
    normalized: str
    boundary: str = "strong"


@dataclass
class AlignedUnit:
    document: DocumentUnit
    asr_text: str
    start: float
    end: float
    start_word: int
    end_word: int
    text_coverage: float
    similarity: float
    mean_word_probability: float | None


@dataclass
class OutputSegment:
    index: int
    text: str
    asr_text: str
    start: float
    end: float
    match_score: float
    text_coverage: float
    mean_word_probability: float
    pause_before: float
    pause_after: float
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    audio: str | None = None
    audio_metrics: dict[str, float] = field(default_factory=dict)
    speaker_metrics: dict[str, Any] = field(default_factory=dict)
    transcript_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = round(self.duration, 4)
        return data


@dataclass(frozen=True)
class PipelineConfig:
    language: str | None = None
    initial_prompt: str | None = None
    model: str = "large-v3"
    asr_backend: str = "faster-whisper"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    asr_batch_size: int = 8
    asr_chunk_length: int = 30
    analysis_sample_rate: int = 16000
    analysis_frame_ms: int = 20
    min_silence_duration: float = 0.32
    silence_margin_db: float = 10.0
    absolute_silence_dbfs: float = -38.0
    boundary_padding: float = 0.08
    max_silence_kept: float = 0.5
    vad_backend: str = "silero"
    vad_threshold: float = 0.5
    vad_min_speech_duration: float = 0.10
    vad_speech_pad: float = 0.0
    vad_chunk_duration: float = 600.0
    vad_chunk_overlap: float = 1.0
    min_duration: float = 1.5
    target_duration: float = 8.0
    max_duration: float = 18.0
    edge_padding: float = 0.08
    min_match_score: float = 0.72
    min_text_coverage: float = 0.70
    min_word_probability: float = 0.45
    max_chars_per_unit: int = 100
    max_boundary_search: float = 1.2
    output_sample_rate: int = 24000
    output_channels: int = 1
    max_clipping_ratio: float = 0.001
    min_rms_dbfs: float = -45.0
    max_dc_offset: float = 0.05
    speaker_backend: str = "off"
    speaker_model: str = "pyannote/speaker-diarization-community-1"
    speaker_token_env: str = "HF_TOKEN"
    max_foreign_speech_ratio: float = 0.05
    max_foreign_speech_seconds: float = 0.25
    min_speaker_speech_seconds: float = 0.5
    min_speaker_similarity: float = 0.55
    max_no_speech_probability: float = 0.60
    max_transcript_compression_ratio: float = 2.40
    max_transcript_repetition_ratio: float = 0.60
    max_characters_per_second: float = 30.0

    def __post_init__(self) -> None:
        if not 0 < self.min_duration <= self.target_duration <= self.max_duration:
            raise ValueError("Require 0 < min_duration <= target_duration <= max_duration")
        if self.analysis_frame_ms <= 0:
            raise ValueError("analysis_frame_ms must be positive")
        if self.min_silence_duration <= self.analysis_frame_ms / 1000:
            raise ValueError("min_silence_duration must exceed one analysis frame")
        if self.max_silence_kept <= 0 or self.boundary_padding < 0:
            raise ValueError("Silence retention must be positive and padding non-negative")
        if self.max_duration - 2 * self.boundary_padding < self.min_duration:
            raise ValueError("Duration range is too narrow for the configured boundary padding")
        if self.vad_backend not in {"silero", "off"}:
            raise ValueError("vad_backend must be 'silero' or 'off'")
        if not 0 < self.vad_threshold < 1:
            raise ValueError("vad_threshold must be between 0 and 1")
        if not 0 <= self.vad_chunk_overlap < self.vad_chunk_duration:
            raise ValueError("VAD chunk overlap must be non-negative and smaller than duration")
        if not 0 <= self.max_clipping_ratio < 1:
            raise ValueError("max_clipping_ratio must be in [0, 1)")
        if not 0 <= self.max_dc_offset < 1:
            raise ValueError("max_dc_offset must be in [0, 1)")
        if self.speaker_backend not in {"off", "pyannote"}:
            raise ValueError("speaker_backend must be 'off' or 'pyannote'")
        if not 0 <= self.max_foreign_speech_ratio <= 1:
            raise ValueError("max_foreign_speech_ratio must be in [0, 1]")
        if self.max_foreign_speech_seconds < 0 or self.min_speaker_speech_seconds < 0:
            raise ValueError("speaker duration thresholds must be non-negative")
        if not -1 <= self.min_speaker_similarity <= 1:
            raise ValueError("min_speaker_similarity must be in [-1, 1]")
        if not 0 <= self.max_no_speech_probability <= 1:
            raise ValueError("max_no_speech_probability must be in [0, 1]")


def path_string(path: Path) -> str:
    return path.as_posix()
