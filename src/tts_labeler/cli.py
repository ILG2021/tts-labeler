from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .asr import FasterWhisperBackend, JsonASRBackend, TransformersWhisperBackend
from .batch import BatchLabelingPipeline
from .models import PipelineConfig
from .pipeline import LabelingPipeline


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-duration", type=float, default=1.5)
    parser.add_argument("--target-duration", type=float, default=8.0)
    parser.add_argument("--max-duration", type=float, default=18.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument(
        "--speaker-backend", choices=("off", "pyannote"), default="off",
        help="Use Community-1 to reject segments containing other speakers",
    )
    parser.add_argument("--speaker-reference", type=Path, help="Clean target-speaker audio")
    parser.add_argument("--speaker-model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--speaker-token-env", default="HF_TOKEN")
    parser.add_argument("--max-foreign-speaker-ratio", type=float, default=0.05)
    parser.add_argument("--max-foreign-speaker-seconds", type=float, default=0.25)
    parser.add_argument("--min-speaker-speech-seconds", type=float, default=0.5)
    parser.add_argument("--min-speaker-similarity", type=float, default=0.55)
    parser.add_argument("--max-no-speech-probability", type=float, default=0.60)
    parser.add_argument("--max-transcript-compression-ratio", type=float, default=2.40)
    parser.add_argument("--max-transcript-repetition-ratio", type=float, default=0.60)
    parser.add_argument("--max-characters-per-second", type=float, default=30.0)


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("faster-whisper", "transformers"),
        default="faster-whisper",
        help="Use transformers for a regular Hugging Face fine-tuned checkpoint",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Built-in model name, Hugging Face model ID, or local model directory",
    )
    parser.add_argument("--language", default="auto")
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Optional Whisper context prompt for vocabulary and punctuation style",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=30)
    parser.add_argument("--min-silence", type=float, default=0.32)
    parser.add_argument("--silence-margin-db", type=float, default=10.0)
    parser.add_argument("--silence-dbfs", type=float, default=-38.0)
    parser.add_argument("--max-silence-kept", type=float, default=0.5)
    parser.add_argument(
        "--vad",
        choices=("silero", "off"),
        default="silero",
        help="Industrial mode requires Silero; off is an explicit RMS-only fallback",
    )
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-speech", type=float, default=0.10)
    parser.add_argument("--vad-speech-pad", type=float, default=0.0)
    parser.add_argument("--vad-chunk-duration", type=float, default=600.0)
    parser.add_argument("--vad-chunk-overlap", type=float, default=1.0)
    parser.add_argument("--min-match-score", type=float, default=0.72)
    parser.add_argument("--min-text-coverage", type=float, default=0.70)
    parser.add_argument("--asr-json", type=Path, help="Testing/import backend for one segment")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts-labeler",
        description="Acoustic-first segmentation and document-guided SRT alignment",
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Segment, transcribe, align SRT, and export")
    run.add_argument("audio", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument(
        "--document",
        type=Path,
        help="Optional document file, or a same-layout .txt directory for folder input",
    )
    _add_output_options(run)
    _add_run_options(run)

    export = subparsers.add_parser("export", help="Export a dataset from an edited SRT")
    export.add_argument("audio", type=Path)
    export.add_argument("subtitle", type=Path)
    export.add_argument("output", type=Path)
    _add_output_options(export)
    return parser


def _config(args: argparse.Namespace) -> PipelineConfig:
    common = dict(
        min_duration=args.min_duration,
        target_duration=args.target_duration,
        max_duration=args.max_duration,
        output_sample_rate=args.sample_rate,
        speaker_backend=args.speaker_backend,
        speaker_model=args.speaker_model,
        speaker_token_env=args.speaker_token_env,
        max_foreign_speech_ratio=args.max_foreign_speaker_ratio,
        max_foreign_speech_seconds=args.max_foreign_speaker_seconds,
        min_speaker_speech_seconds=args.min_speaker_speech_seconds,
        min_speaker_similarity=args.min_speaker_similarity,
        max_no_speech_probability=args.max_no_speech_probability,
        max_transcript_compression_ratio=args.max_transcript_compression_ratio,
        max_transcript_repetition_ratio=args.max_transcript_repetition_ratio,
        max_characters_per_second=args.max_characters_per_second,
    )
    if args.command == "export":
        return PipelineConfig(**common)
    return PipelineConfig(
        **common,
        model=args.model,
        language=None if args.language.lower() == "auto" else args.language,
        initial_prompt=args.initial_prompt,
        asr_backend=args.backend,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        asr_batch_size=args.batch_size,
        asr_chunk_length=args.chunk_length,
        min_silence_duration=args.min_silence,
        silence_margin_db=args.silence_margin_db,
        absolute_silence_dbfs=args.silence_dbfs,
        max_silence_kept=args.max_silence_kept,
        vad_backend=args.vad,
        vad_threshold=args.vad_threshold,
        vad_min_speech_duration=args.vad_min_speech,
        vad_speech_pad=args.vad_speech_pad,
        vad_chunk_duration=args.vad_chunk_duration,
        vad_chunk_overlap=args.vad_chunk_overlap,
        min_match_score=args.min_match_score,
        min_text_coverage=args.min_text_coverage,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _config(args)
    if args.command == "export":
        LabelingPipeline(config).export_edited_srt(
            args.audio, args.subtitle, args.output, args.speaker_reference
        )
        return 0
    if args.asr_json:
        backend = JsonASRBackend(args.asr_json)
    elif args.backend == "transformers":
        backend = TransformersWhisperBackend(config)
    else:
        backend = FasterWhisperBackend(config)
    pipeline = LabelingPipeline(config, backend)
    if args.audio.is_dir():
        result = BatchLabelingPipeline(pipeline).run(
            args.audio, args.document, args.output, args.speaker_reference
        )
        return 1 if result.failed else 0
    pipeline.run(args.audio, args.document, args.output, args.speaker_reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
