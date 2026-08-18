from tts_labeler.cli import build_parser


def test_custom_transformers_model_arguments() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "audio.wav",
            "output",
            "--document",
            "document.txt",
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
        ["run", "audio.wav", "output", "--document", "document.txt", "--model", "models/custom-ct2"]
    )
    assert args.backend == "faster-whisper"
    assert args.model == "models/custom-ct2"
    assert args.language == "auto"


def test_export_edited_srt_arguments() -> None:
    args = build_parser().parse_args(["export", "audio.wav", "edited.srt", "dataset"])
    assert args.command == "export"
    assert str(args.subtitle) == "edited.srt"


def test_industrial_vad_is_default_and_fallback_is_explicit() -> None:
    default = build_parser().parse_args(["run", "audio.wav", "out", "--document", "doc.txt"])
    fallback = build_parser().parse_args(
        ["run", "audio.wav", "out", "--document", "doc.txt", "--vad", "off"]
    )
    assert default.vad == "silero"
    assert fallback.vad == "off"


def test_run_without_document_uses_second_path_as_output() -> None:
    args = build_parser().parse_args(["run", "audio.wav", "output"])
    assert args.document is None
    assert str(args.output) == "output"


def test_run_with_document_preserves_three_paths() -> None:
    args = build_parser().parse_args(
        ["run", "audio.wav", "output", "--document", "document.txt"]
    )
    assert str(args.document) == "document.txt"
    assert str(args.output) == "output"


def test_pyannote_speaker_arguments() -> None:
    args = build_parser().parse_args(
        [
            "run", "audio.wav", "out", "--document", "doc.txt",
            "--speaker-backend", "pyannote",
            "--speaker-reference", "target.wav",
        ]
    )
    assert args.speaker_backend == "pyannote"
    assert str(args.speaker_reference) == "target.wav"
