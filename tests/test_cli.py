from tts_labeler.cli import build_parser


def test_custom_transformers_model_arguments() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "audio.wav",
            "document.txt",
            "output",
            "--backend",
            "transformers",
            "--model",
            "organization/custom-whisper",
            "--language",
            "ar",
        ]
    )
    assert args.backend == "transformers"
    assert args.model == "organization/custom-whisper"
    assert args.language == "ar"


def test_local_ctranslate2_model_arguments() -> None:
    args = build_parser().parse_args(
        ["run", "audio.wav", "document.txt", "output", "--model", "models/custom-ct2"]
    )
    assert args.backend == "faster-whisper"
    assert args.model == "models/custom-ct2"
    assert args.language == "auto"


def test_export_edited_srt_arguments() -> None:
    args = build_parser().parse_args(["export", "audio.wav", "edited.srt", "dataset"])
    assert args.command == "export"
    assert str(args.subtitle) == "edited.srt"


def test_industrial_vad_is_default_and_fallback_is_explicit() -> None:
    default = build_parser().parse_args(["run", "audio.wav", "doc.txt", "out"])
    fallback = build_parser().parse_args(
        ["run", "audio.wav", "doc.txt", "out", "--vad", "off"]
    )
    assert default.vad == "silero"
    assert fallback.vad == "off"


def test_no_document_run_arguments() -> None:
    args = build_parser().parse_args(["run", "audio.wav", "output", "--no-document"])
    assert args.no_document is True
    assert str(args.document) == "output"
    assert args.output is None


def test_pyannote_speaker_arguments() -> None:
    args = build_parser().parse_args(
        [
            "run", "audio.wav", "doc.txt", "out",
            "--speaker-backend", "pyannote",
            "--speaker-reference", "target.wav",
        ]
    )
    assert args.speaker_backend == "pyannote"
    assert str(args.speaker_reference) == "target.wav"
