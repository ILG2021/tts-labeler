import tempfile
import wave
from array import array
from pathlib import Path

from tts_labeler.audio import analyze_wav_quality


def test_audio_quality_detects_clipping_and_dc_offset() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "clipped.wav"
        samples = array("h", [32767]) * 16000
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(samples.tobytes())
        metrics = analyze_wav_quality(path)
    assert metrics["clipping_ratio"] == 1.0
    assert metrics["dc_offset"] > 0.99
    assert metrics["peak_dbfs"] > -0.01


def test_audio_quality_detects_silence() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "silent.wav"
        samples = array("h", [0]) * 16000
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(samples.tobytes())
        metrics = analyze_wav_quality(path)
    assert metrics["rms_dbfs"] == -100.0
    assert metrics["peak_dbfs"] == -100.0
