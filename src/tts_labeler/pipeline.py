from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import asdict
from pathlib import Path

from .acoustic import detect_intervals
from .acoustic import AcousticInterval
from .asr import ASRBackend
from .audio import (
    analyze_wav_quality,
    export_interval,
    normalize_for_analysis,
    normalize_master,
    wav_duration,
)
from .models import OutputSegment, PipelineConfig, SubtitleCue, Word, path_string
from .quality import transcript_quality
from .srt import read as read_srt
from .srt import validate_timeline
from .srt import write as write_srt
from .speaker import PyannoteSpeakerAnalyzer, SpeakerTurn, speaker_metrics
from .subtitle_alignment import align_subtitles_to_document
from .vad import SileroVAD
from .state import atomic_write_json, atomic_write_text, load_json, run_fingerprint


LOGGER = logging.getLogger(__name__)


class LabelingPipeline:
    def __init__(self, config: PipelineConfig, asr: ASRBackend | None = None) -> None:
        self.config = config
        self.asr = asr

    def run(
        self,
        audio: Path,
        document: Path | None,
        output: Path,
        speaker_reference: Path | None = None,
    ) -> list[OutputSegment]:
        try:
            return self._run_impl(audio, document, output, speaker_reference)
        except Exception as exc:
            state_path = output / "work" / "state.json"
            if state_path.exists():
                state = load_json(state_path)
                if isinstance(state, dict) and state.get("status") == "running":
                    atomic_write_json(
                        state_path,
                        {
                            **state,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
            raise

    def _run_impl(
        self,
        audio: Path,
        document: Path | None,
        output: Path,
        speaker_reference: Path | None = None,
    ) -> list[OutputSegment]:
        """Segment and transcribe; optionally align the ASR text to a document."""
        if not audio.is_file():
            raise FileNotFoundError(audio)
        if document is not None and not document.is_file():
            raise FileNotFoundError(document)
        if speaker_reference is not None and not speaker_reference.is_file():
            raise FileNotFoundError(speaker_reference)
        if self.asr is None:
            raise RuntimeError("An ASR backend is required for the run workflow")
        output.mkdir(parents=True, exist_ok=True)
        work = output / "work"
        state_path = work / "state.json"
        fingerprint = run_fingerprint(audio, document, self.config, speaker_reference)
        if state_path.exists():
            state = load_json(state_path)
            if not isinstance(state, dict) or state.get("fingerprint") != fingerprint:
                raise FileExistsError(
                    "Output directory belongs to a different input/configuration; use a new directory"
                )
        elif (output / "manifest.jsonl").exists() or (output / "report.json").exists():
            raise FileExistsError("Output directory contains an untracked previous run")
        atomic_write_json(state_path, {"fingerprint": fingerprint, "status": "running"})
        master_wav = work / "master.wav"
        analysis_wav = work / "analysis.wav"

        if not master_wav.exists():
            LOGGER.info("Creating canonical PCM master timeline")
            normalize_master(audio, master_wav, self.config)
        if not analysis_wav.exists():
            normalize_for_analysis(master_wav, analysis_wav, self.config)

        speaker_turns, target_speaker, speaker_report = self._analyze_speakers(
            analysis_wav, work, speaker_reference
        )

        acoustic_cache = work / "acoustic.json"
        if acoustic_cache.exists():
            cached = load_json(acoustic_cache)
            if not isinstance(cached, dict):
                raise ValueError("Invalid acoustic cache")
            intervals = [AcousticInterval(**item) for item in cached["intervals"]]
            acoustic_report = cached["report"]
        else:
            vad_report: dict = {"backend": "off"}
            speech_regions = None
            if self.config.vad_backend == "silero":
                speech_regions, vad_report = SileroVAD(self.config).analyze(analysis_wav)
                if not speech_regions:
                    raise RuntimeError("Silero VAD found no speech in the source audio")
            intervals, acoustic_report = detect_intervals(
                analysis_wav, self.config, speech_regions=speech_regions
            )
            acoustic_report["vad"] = vad_report
            atomic_write_json(
                acoustic_cache,
                {
                    "intervals": [asdict(interval) for interval in intervals],
                    "report": acoustic_report,
                },
            )
        LOGGER.info("Acoustic segmentation produced %d intervals", len(intervals))

        raw_cues: list[SubtitleCue] = []
        asr_reports: list[dict] = []
        source_clips = work / "acoustic_segments"
        asr_cache_dir = work / "asr"
        for index, interval in enumerate(intervals):
            clip = source_clips / f"{index:06d}.wav"
            if not clip.exists():
                export_interval(master_wav, clip, interval.start, interval.end, self.config)
            cache_path = asr_cache_dir / f"{index:06d}.json"
            if cache_path.exists():
                cached_asr = load_json(cache_path)
                if not isinstance(cached_asr, dict):
                    raise ValueError(f"Invalid ASR cache: {cache_path}")
                words = [Word(**item) for item in cached_asr["words"]]
                metadata = cached_asr["metadata"]
            else:
                LOGGER.info("Transcribing acoustic segment %d/%d", index + 1, len(intervals))
                words, metadata = self.asr.transcribe(clip)
                atomic_write_json(
                    cache_path,
                    {"words": [asdict(word) for word in words], "metadata": metadata},
                )
            text = "".join(word.text for word in words).strip()
            known_probabilities = [
                word.probability for word in words if word.probability is not None
            ]
            probability = (
                sum(known_probabilities) / len(known_probabilities)
                if known_probabilities
                else None
            )
            raw_cues.append(
                SubtitleCue(
                    index=index,
                    start=interval.start,
                    end=interval.end,
                    text=text,
                    mean_asr_probability=probability,
                )
            )
            asr_reports.append(metadata)

        raw_srt = output / "raw.srt"
        write_srt(raw_srt, raw_cues)
        LOGGER.info("Wrote raw ASR subtitles: %s", raw_srt)

        if document is None:
            aligned_cues = raw_cues
            source_subtitle = raw_srt
            LOGGER.info("No reference document supplied; using raw ASR subtitles")
        else:
            document_text = document.read_text(encoding="utf-8-sig")
            aligned_cues = align_subtitles_to_document(document_text, raw_cues)
            for aligned, raw in zip(aligned_cues, raw_cues, strict=True):
                aligned.mean_asr_probability = raw.mean_asr_probability
            aligned_srt = output / "aligned.srt"
            write_srt(aligned_srt, aligned_cues)
            source_subtitle = None
            LOGGER.info("Wrote document-aligned subtitles: %s", aligned_srt)

        segments = self.export_dataset(
            audio,
            aligned_cues,
            output,
            cut_source=master_wav,
            source_document=document,
            source_subtitle=source_subtitle,
            acoustic_report=acoustic_report,
            asr_reports=asr_reports,
            speaker_turns=speaker_turns,
            target_speaker=target_speaker,
            speaker_report=speaker_report,
            document_guided=document is not None,
        )
        atomic_write_json(state_path, {"fingerprint": fingerprint, "status": "complete"})
        return segments

    def export_edited_srt(
        self,
        audio: Path,
        subtitle: Path,
        output: Path,
        speaker_reference: Path | None = None,
    ) -> list[OutputSegment]:
        """Build a dataset from an SRT edited in any subtitle application."""
        if not audio.is_file():
            raise FileNotFoundError(audio)
        if speaker_reference is not None and not speaker_reference.is_file():
            raise FileNotFoundError(speaker_reference)
        cues = read_srt(subtitle)
        if not cues:
            raise ValueError("The SRT contains no valid cues")
        output.mkdir(parents=True, exist_ok=True)
        if (output / "manifest.jsonl").exists() or (output / "report.json").exists():
            raise FileExistsError("Export output already contains a dataset; use a new directory")
        master_wav = output / "work" / "master.wav"
        normalize_master(audio, master_wav, self.config)
        analysis_wav = output / "work" / "analysis.wav"
        normalize_for_analysis(master_wav, analysis_wav, self.config)
        speaker_turns, target_speaker, speaker_report = self._analyze_speakers(
            analysis_wav, output / "work", speaker_reference
        )
        validate_timeline(
            cues,
            wav_duration(master_wav),
            max_overlap=self.config.boundary_padding * 2 + 0.01,
        )
        return self.export_dataset(
            audio,
            cues,
            output,
            cut_source=master_wav,
            source_subtitle=subtitle,
            edited=True,
            speaker_turns=speaker_turns,
            target_speaker=target_speaker,
            speaker_report=speaker_report,
        )

    def _analyze_speakers(
        self, audio: Path, work: Path, reference: Path | None
    ) -> tuple[list[SpeakerTurn] | None, str | None, dict]:
        if self.config.speaker_backend == "off":
            return None, None, {"backend": "off"}
        cache_path = work / "speaker.json"
        if cache_path.exists():
            cached = load_json(cache_path)
            if not isinstance(cached, dict):
                raise ValueError("Invalid speaker cache")
            report = cached["report"]
            turns = [SpeakerTurn(**item) for item in cached["turns"]]
            return turns, str(report["target_speaker"]), report
        turns, target, report = PyannoteSpeakerAnalyzer(self.config).analyze(audio, reference)
        atomic_write_json(
            cache_path, {"turns": [asdict(turn) for turn in turns], "report": report}
        )
        return turns, target, report

    def export_dataset(
        self,
        audio: Path,
        cues: list[SubtitleCue],
        output: Path,
        *,
        cut_source: Path | None = None,
        source_document: Path | None = None,
        source_subtitle: Path | None = None,
        acoustic_report: dict | None = None,
        asr_reports: list[dict] | None = None,
        speaker_turns: list[SpeakerTurn] | None = None,
        target_speaker: str | None = None,
        speaker_report: dict | None = None,
        edited: bool = False,
        document_guided: bool = True,
    ) -> list[OutputSegment]:
        segments: list[OutputSegment] = []
        for index, cue in enumerate(cues):
            reasons: list[str] = []
            if cue.duration < self.config.min_duration:
                reasons.append("too_short")
            if cue.duration > self.config.max_duration:
                reasons.append("too_long")
            if not cue.text.strip():
                reasons.append("empty_text")
            if document_guided and not edited and cue.match_score < self.config.min_match_score:
                reasons.append("low_match_score")
            if document_guided and not edited and cue.text_coverage < self.config.min_text_coverage:
                reasons.append("low_text_coverage")
            if (
                not edited
                and cue.mean_asr_probability is not None
                and cue.mean_asr_probability < self.config.min_word_probability
            ):
                reasons.append("low_asr_confidence")
            segment = OutputSegment(
                index=index,
                text=cue.text.strip(),
                asr_text=cue.asr_text or cue.text.strip(),
                start=cue.start,
                end=cue.end,
                match_score=cue.match_score,
                text_coverage=cue.text_coverage,
                mean_word_probability=cue.mean_asr_probability,
                pause_before=0.0,
                pause_after=0.0,
                accepted=not reasons,
                reasons=reasons,
            )
            transcript_metrics = transcript_quality(cue.asr_text or cue.text, cue.duration)
            segment.transcript_metrics = transcript_metrics
            if not edited:
                asr_metadata = asr_reports[index] if asr_reports and index < len(asr_reports) else {}
                if (
                    float(asr_metadata.get("max_no_speech_probability", 0.0))
                    > self.config.max_no_speech_probability
                ):
                    segment.reasons.append("asr_no_speech")
                if (
                    transcript_metrics["normalized_characters"] >= 50
                    and transcript_metrics["compression_ratio"]
                    > self.config.max_transcript_compression_ratio
                ):
                    segment.reasons.append("asr_compression_hallucination")
                if (
                    transcript_metrics["normalized_characters"] >= 12
                    and transcript_metrics["repetition_ratio"]
                    > self.config.max_transcript_repetition_ratio
                ):
                    segment.reasons.append("asr_repetition")
                if (
                    transcript_metrics["characters_per_second"]
                    > self.config.max_characters_per_second
                ):
                    segment.reasons.append("asr_text_too_fast")
            if speaker_turns is not None and target_speaker is not None:
                metrics = speaker_metrics(speaker_turns, target_speaker, cue.start, cue.end)
                segment.speaker_metrics = metrics
                if float(metrics["speech_seconds"]) < self.config.min_speaker_speech_seconds:
                    segment.reasons.append("speaker_uncertain")
                if (
                    float(metrics["foreign_speech_seconds"])
                    > self.config.max_foreign_speech_seconds
                    or float(metrics["foreign_speech_ratio"])
                    > self.config.max_foreign_speech_ratio
                ):
                    segment.reasons.append("mixed_speaker")
            directory = output / ("wavs" if segment.accepted else "rejected")
            target = directory / f"{index:06d}.wav"
            export_interval(cut_source or audio, target, cue.start, cue.end, self.config)
            metrics = analyze_wav_quality(target)
            segment.audio_metrics = metrics
            if metrics["clipping_ratio"] > self.config.max_clipping_ratio:
                segment.reasons.append("audio_clipping")
            if metrics["rms_dbfs"] < self.config.min_rms_dbfs:
                segment.reasons.append("audio_too_quiet")
            if abs(metrics["dc_offset"]) > self.config.max_dc_offset:
                segment.reasons.append("audio_dc_offset")
            segment.accepted = not segment.reasons
            final_directory = output / ("wavs" if segment.accepted else "rejected")
            final_target = final_directory / f"{index:06d}.wav"
            if final_target != target:
                final_target.parent.mkdir(parents=True, exist_ok=True)
                target.replace(final_target)
                target = final_target
            segment.audio = path_string(target.relative_to(output))
            segments.append(segment)

        self._write_outputs(
            output,
            audio,
            segments,
            source_document=source_document,
            source_subtitle=source_subtitle,
            acoustic_report=acoustic_report,
            asr_reports=asr_reports,
            speaker_report=speaker_report,
            edited=edited,
        )
        accepted = sum(item.accepted for item in segments)
        LOGGER.info("Dataset complete: %d accepted, %d rejected", accepted, len(segments) - accepted)
        return segments

    def _write_outputs(
        self,
        output: Path,
        audio: Path,
        segments: list[OutputSegment],
        *,
        source_document: Path | None,
        source_subtitle: Path | None,
        acoustic_report: dict | None,
        asr_reports: list[dict] | None,
        speaker_report: dict | None,
        edited: bool,
    ) -> None:
        manifest_text = "".join(
            json.dumps(segment.to_dict(), ensure_ascii=False) + "\n" for segment in segments
        )
        atomic_write_text(output / "manifest.jsonl", manifest_text)
        csv_buffer = io.StringIO(newline="")
        csv_writer = csv.writer(csv_buffer, delimiter="|", lineterminator="\n")
        for segment in segments:
            if segment.accepted:
                csv_writer.writerow([segment.audio, segment.text])
        atomic_write_text(output / "metadata.csv", csv_buffer.getvalue(), encoding="utf-8-sig")
        report = {
            "source_audio": str(audio.resolve()),
            "source_document": str(source_document.resolve()) if source_document else None,
            "source_subtitle": str(source_subtitle.resolve()) if source_subtitle else None,
            "edited_srt_export": edited,
            "config": asdict(self.config),
            "acoustic_analysis": acoustic_report,
            "asr_segments": asr_reports,
            "speaker_analysis": speaker_report,
            "counts": {
                "segments": len(segments),
                "accepted": sum(item.accepted for item in segments),
                "rejected": sum(not item.accepted for item in segments),
            },
            "accepted_duration": round(sum(x.duration for x in segments if x.accepted), 3),
        }
        atomic_write_json(output / "report.json", report)
