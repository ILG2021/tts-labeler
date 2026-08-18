from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .pipeline import LabelingPipeline
from .state import atomic_write_json, atomic_write_text


LOGGER = logging.getLogger(__name__)
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}


@dataclass(frozen=True)
class BatchResult:
    files: int
    completed: int
    failed: int
    segments: int
    accepted: int
    rejected: int


def discover_audio_files(root: Path) -> list[Path]:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    stems: dict[str, Path] = {}
    for path in files:
        key = path.stem.casefold()
        if key in stems:
            raise ValueError(
                "Batch input contains duplicate file stems, which would collide in the dataset: "
                f"{stems[key]} and {path}"
            )
        stems[key] = path
    return files


def _matching_document(audio: Path, audio_root: Path, document: Path | None) -> Path | None:
    if document is None:
        return None
    if document.is_file():
        raise ValueError("For folder input, --document must be a folder containing matching .txt files")
    if not document.is_dir():
        raise FileNotFoundError(document)
    relative = audio.relative_to(audio_root).with_suffix(".txt")
    candidate = document / relative
    return candidate if candidate.is_file() else None


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise FileExistsError(f"Existing batch artifact differs: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


class BatchLabelingPipeline:
    def __init__(self, pipeline: LabelingPipeline) -> None:
        self.pipeline = pipeline

    def run(
        self,
        audio_root: Path,
        document_root: Path | None,
        output: Path,
        speaker_reference: Path | None = None,
    ) -> BatchResult:
        if not audio_root.is_dir():
            raise NotADirectoryError(audio_root)
        files = discover_audio_files(audio_root)
        if not files:
            raise ValueError(f"No supported audio files found in {audio_root}")
        output.mkdir(parents=True, exist_ok=True)
        item_root = output / "work" / "items"
        manifests: list[dict] = []
        item_reports: list[dict] = []
        failures: list[dict] = []
        missing_documents: list[str] = []

        for position, audio in enumerate(files, start=1):
            relative_audio = audio.relative_to(audio_root).as_posix()
            LOGGER.info("Batch file %d/%d: %s", position, len(files), relative_audio)
            document = _matching_document(audio, audio_root, document_root)
            if document_root is not None and document is None:
                missing_documents.append(relative_audio)
            item_output = item_root / audio.stem
            try:
                self.pipeline.run(audio, document, item_output, speaker_reference)
                item_manifest = item_output / "manifest.jsonl"
                for line in item_manifest.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    source_clip = item_output / record["audio"]
                    target_clip = output / record["audio"]
                    _link_or_copy(source_clip, target_clip)
                    record["source_audio"] = str(audio.resolve())
                    manifests.append(record)
                subtitle_dir = output / "subtitles" / audio.stem
                for name in ("raw.srt", "aligned.srt"):
                    source_subtitle = item_output / name
                    if source_subtitle.is_file():
                        _link_or_copy(source_subtitle, subtitle_dir / name)
                report = json.loads((item_output / "report.json").read_text(encoding="utf-8"))
                item_reports.append(
                    {
                        "audio": relative_audio,
                        "document": str(document) if document else None,
                        "counts": report["counts"],
                        "accepted_duration": report["accepted_duration"],
                    }
                )
            except Exception as exc:
                LOGGER.exception("Batch file failed: %s", relative_audio)
                failures.append(
                    {
                        "audio": relative_audio,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        manifest_text = "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in manifests
        )
        atomic_write_text(output / "manifest.jsonl", manifest_text)
        csv_buffer = io.StringIO(newline="")
        writer = csv.writer(csv_buffer, delimiter="|", lineterminator="\n")
        for record in manifests:
            if record["accepted"]:
                writer.writerow([record["audio"], record["text"]])
        atomic_write_text(output / "metadata.csv", csv_buffer.getvalue(), encoding="utf-8-sig")
        accepted = sum(bool(record["accepted"]) for record in manifests)
        result = BatchResult(
            files=len(files),
            completed=len(item_reports),
            failed=len(failures),
            segments=len(manifests),
            accepted=accepted,
            rejected=len(manifests) - accepted,
        )
        atomic_write_json(
            output / "report.json",
            {
                "mode": "batch",
                "source_directory": str(audio_root.resolve()),
                "document_directory": str(document_root.resolve()) if document_root else None,
                "counts": {
                    "files": result.files,
                    "completed": result.completed,
                    "failed": result.failed,
                    "segments": result.segments,
                    "accepted": result.accepted,
                    "rejected": result.rejected,
                },
                "missing_documents": missing_documents,
                "items": item_reports,
                "failures": failures,
            },
        )
        LOGGER.info(
            "Batch complete: %d/%d files, %d accepted, %d rejected, %d failed",
            result.completed,
            result.files,
            result.accepted,
            result.rejected,
            result.failed,
        )
        return result
