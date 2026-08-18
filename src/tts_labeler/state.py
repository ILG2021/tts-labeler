from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .models import PipelineConfig


PIPELINE_SCHEMA = "acoustic-srt-v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_fingerprint(
    audio: Path, document: Path | None, config: PipelineConfig, speaker_reference: Path | None = None
) -> str:
    payload = {
        "schema": PIPELINE_SCHEMA,
        "audio_sha256": file_sha256(audio),
        "document_sha256": file_sha256(document) if document else None,
        "speaker_reference_sha256": file_sha256(speaker_reference) if speaker_reference else None,
        "config": asdict(config),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(content, encoding=encoding, newline="\n")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))
