import json
import math
import tempfile
import wave
from array import array
from pathlib import Path

from tts_labeler.batch import BatchLabelingPipeline, discover_audio_files
from tts_labeler.models import PipelineConfig, Word
from tts_labeler.pipeline import LabelingPipeline


class _ConstantASR:
    def transcribe(self, audio: Path):
        del audio
        return [Word("test sentence", 0.0, 0.5, 0.95)], {"test": True}


def _write_tone(path: Path) -> None:
    rate = 16000
    samples = array(
        "h",
        (round(9000 * math.sin(2 * math.pi * 220 * index / rate)) for index in range(rate)),
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def test_batch_folder_aggregates_named_audio_directories() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        inputs = root / "inputs"
        inputs.mkdir()
        _write_tone(inputs / "文件1.wav")
        _write_tone(inputs / "文件2.wav")
        output = root / "output"
        config = PipelineConfig(
            min_duration=0.4,
            target_duration=1.0,
            max_duration=2.0,
            min_silence_duration=0.3,
            vad_backend="off",
        )
        result = BatchLabelingPipeline(LabelingPipeline(config, _ConstantASR())).run(
            inputs, None, output
        )
        assert result.completed == 2
        assert result.failed == 0
        assert (output / "wavs" / "文件1" / "文件1_1.wav").is_file()
        assert (output / "wavs" / "文件2" / "文件2_1.wav").is_file()
        metadata = (output / "metadata.csv").read_text(encoding="utf-8-sig")
        assert "wavs/文件1/文件1_1.wav|test sentence" in metadata
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["counts"]["files"] == 2


def test_batch_rejects_duplicate_stems() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "a").mkdir()
        (root / "b").mkdir()
        (root / "a" / "same.wav").touch()
        (root / "b" / "same.mp3").touch()
        try:
            discover_audio_files(root)
        except ValueError as error:
            assert "duplicate file stems" in str(error)
        else:
            raise AssertionError("Expected duplicate stems to be rejected")
