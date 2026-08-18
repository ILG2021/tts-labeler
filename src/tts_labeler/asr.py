from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

from .models import PipelineConfig, Word


class ASRBackend(Protocol):
    def transcribe(self, audio: Path) -> tuple[list[Word], dict]: ...


class FasterWhisperBackend:
    def __init__(self, config: PipelineConfig) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install with: pip install -e .[asr]"
            ) from exc
        self.config = config
        self.model = WhisperModel(
            config.model,
            device=config.device,
            compute_type=config.compute_type,
        )

    def transcribe(self, audio: Path) -> tuple[list[Word], dict]:
        segments, info = self.model.transcribe(
            str(audio),
            language=self.config.language,
            initial_prompt=self.config.initial_prompt,
            beam_size=self.config.beam_size,
            word_timestamps=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250, "speech_pad_ms": 160},
            condition_on_previous_text=False,
        )
        words: list[Word] = []
        segment_metrics: list[dict] = []
        for segment in segments:
            probability = max(0.0, min(1.0, math.exp(float(segment.avg_logprob))))
            words.append(
                Word(
                    text=segment.text,
                    start=float(segment.start),
                    end=float(segment.end),
                    probability=probability,
                )
            )
            segment_metrics.append(
                {
                    "avg_logprob": float(segment.avg_logprob),
                    "no_speech_probability": float(segment.no_speech_prob),
                    "compression_ratio": float(segment.compression_ratio),
                }
            )
        metadata = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "duration_after_vad": info.duration_after_vad,
            "segments": segment_metrics,
            "max_no_speech_probability": max(
                (item["no_speech_probability"] for item in segment_metrics), default=0.0
            ),
            "max_compression_ratio": max(
                (item["compression_ratio"] for item in segment_metrics), default=0.0
            ),
        }
        return words, metadata


class TransformersWhisperBackend:
    """Whisper backend for Hugging Face or local fine-tuned checkpoints.

    Unlike faster-whisper, this backend loads regular Transformers checkpoints
    directly, so a fine-tuned model does not need CTranslate2 conversion first.
    The checkpoint must be a complete/merged Whisper model, not an unmerged LoRA
    adapter directory.
    """

    def __init__(self, config: PipelineConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Transformers Whisper dependencies are missing. Install with: "
                "pip install -e .[transformers]"
            ) from exc

        if config.device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif config.device == "cuda":
            device = "cuda:0"
        else:
            device = config.device

        if config.compute_type in {"float16", "auto"} and device.startswith("cuda"):
            dtype = torch.float16
        elif config.compute_type == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            config.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(config.model)
        self.config = config
        self.prompt_ids = None
        if config.initial_prompt:
            self.prompt_ids = processor.get_prompt_ids(
                config.initial_prompt, return_tensors="pt"
            ).to(model.device)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
        )

    def transcribe(self, audio: Path) -> tuple[list[Word], dict]:
        generate_kwargs = {"task": "transcribe"}
        if self.config.language:
            generate_kwargs["language"] = self.config.language
        if self.prompt_ids is not None:
            generate_kwargs["prompt_ids"] = self.prompt_ids
        result = self.pipe(
            str(audio),
            return_timestamps=True,
            chunk_length_s=self.config.asr_chunk_length,
            batch_size=self.config.asr_batch_size,
            generate_kwargs=generate_kwargs,
        )
        words: list[Word] = []
        for chunk in result.get("chunks", []):
            timestamp = chunk.get("timestamp") or (None, None)
            start, end = timestamp
            if start is None or end is None:
                continue
            words.append(Word(str(chunk.get("text", "")), float(start), float(end), None))
        return words, {
            "backend": "transformers",
            "model": self.config.model,
            "language": self.config.language or "auto",
        }


class JsonASRBackend:
    """Deterministic backend for testing or importing external word timestamps."""

    def __init__(self, json_path: Path) -> None:
        self.json_path = json_path

    def transcribe(self, audio: Path) -> tuple[list[Word], dict]:
        del audio
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [Word(**item) for item in payload], {"source": "json"}
        words = [Word(**item) for item in payload["words"]]
        return words, payload.get("metadata", {"source": "json"})
