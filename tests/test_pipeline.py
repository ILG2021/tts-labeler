import math
import tempfile
import wave
from array import array
from pathlib import Path

from tts_labeler.models import PipelineConfig, Word
from tts_labeler.pipeline import LabelingPipeline
from tts_labeler.srt import read as read_srt
from tts_labeler.state import load_json


class _SequenceASR:
    def __init__(self) -> None:
        self.texts = iter(["Hello word", "Good bye"])
        self.calls = 0

    def transcribe(self, audio: Path):
        self.calls += 1
        text = next(self.texts)
        return [Word(text, 0.0, 0.5, 0.95)], {"test": True}


class _FailingASR:
    def transcribe(self, audio: Path):
        del audio
        raise RuntimeError("deliberate ASR failure")


def _write_source(path: Path) -> None:
    rate = 16000
    tone = array(
        "h",
        (round(9000 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(rate)),
    )
    samples = array("h", tone)
    samples.extend(array("h", [0]) * round(rate * 0.6))
    samples.extend(tone)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def test_end_to_end_acoustic_srt_alignment_and_export() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audio = root / "source.wav"
        document = root / "document.txt"
        output = root / "output"
        _write_source(audio)
        document.write_text("Hello world! Goodbye.", encoding="utf-8")
        config = PipelineConfig(
            min_duration=0.4,
            target_duration=1.0,
            max_duration=2.0,
            min_silence_duration=0.3,
            boundary_padding=0.05,
            min_match_score=0.5,
            min_text_coverage=0.5,
            vad_backend="off",
        )
        backend = _SequenceASR()
        pipeline = LabelingPipeline(config, backend)
        segments = pipeline.run(audio, document, output)
        raw = read_srt(output / "raw.srt")
        aligned = read_srt(output / "aligned.srt")
        assert [cue.text for cue in raw] == ["Hello word", "Good bye"]
        assert [cue.text for cue in aligned] == ["Hello world!", "Goodbye."]
        assert [segment.text for segment in segments] == ["Hello world!", "Goodbye."]
        assert all((output / segment.audio).is_file() for segment in segments)
        assert (output / "work" / "master.wav").is_file()
        pipeline.run(audio, document, output)
        assert backend.calls == 2


def test_output_directory_rejects_different_run() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audio = root / "source.wav"
        document = root / "document.txt"
        output = root / "output"
        _write_source(audio)
        document.write_text("Hello world! Goodbye.", encoding="utf-8")
        config = PipelineConfig(
            min_duration=0.4,
            target_duration=1.0,
            max_duration=2.0,
            min_silence_duration=0.3,
            vad_backend="off",
            min_match_score=0.5,
            min_text_coverage=0.5,
        )
        LabelingPipeline(config, _SequenceASR()).run(audio, document, output)
        document.write_text("Changed document.", encoding="utf-8")
        try:
            LabelingPipeline(config, _SequenceASR()).run(audio, document, output)
        except FileExistsError as error:
            assert "different input" in str(error)
        else:
            raise AssertionError("Expected run fingerprint mismatch to fail")


def test_run_without_document_uses_raw_srt() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audio = root / "source.wav"
        output = root / "output"
        _write_source(audio)
        config = PipelineConfig(
            min_duration=0.4, target_duration=1.0, max_duration=2.0,
            min_silence_duration=0.3, boundary_padding=0.05, vad_backend="off",
        )
        segments = LabelingPipeline(config, _SequenceASR()).run(audio, None, output)
        assert [cue.text for cue in read_srt(output / "raw.srt")] == ["Hello word", "Good bye"]
        assert not (output / "aligned.srt").exists()
        assert [segment.text for segment in segments] == ["Hello word", "Good bye"]
        report = (output / "report.json").read_text(encoding="utf-8")
        assert '"source_document": null' in report


def test_failed_run_records_failure_state() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audio = root / "source.wav"
        output = root / "output"
        _write_source(audio)
        config = PipelineConfig(
            min_duration=0.4, target_duration=1.0, max_duration=2.0,
            min_silence_duration=0.3, vad_backend="off",
        )
        try:
            LabelingPipeline(config, _FailingASR()).run(audio, None, output)
        except RuntimeError as error:
            assert "deliberate" in str(error)
        else:
            raise AssertionError("Expected ASR failure")
        state = load_json(output / "work" / "state.json")
        assert state["status"] == "failed"
        assert state["error_type"] == "RuntimeError"
