#!/usr/bin/env python3
"""
split_by_silence.py

按静音区切分音频文件的命令行工具。

功能：
- 检测音频中的静音区（可配置最小静音长度 / 响度阈值）
- 依据 "最长静音优先、从中间切分" 的规则递归切分
- 支持片段最小/最大长度约束
- 支持保留切分点两侧的静音长度（避免掐头掐尾过狠或保留过多空白）
- 支持预览模式（只打印切分点和片段时长，不生成文件）
- 支持导出 SRT 字幕（时间戳对应切分后的片段，文字可用占位符）
- 支持调用本地 Whisper 模型转录每个片段（可自动填充字幕文字）

依赖：
    uv add pydub
    # 另需系统安装 ffmpeg（pydub 依赖它做音频解码/编码）

    # 转录功能按你的模型格式三选一安装（也可以都装，脚本会自动检测该用哪个）：
    uv add openai-whisper        # 官方 .pt 权重文件 / 官方模型名（如 base、small）
    uv add transformers torch    # HuggingFace 格式目录（Whisper-Finetune merge_lora 后常见）
    uv add faster-whisper        # CTranslate2 格式目录（ct2-transformers-converter 转换后）

用法示例：
    # 仅预览切分结果，调试参数
    python split_by_silence.py -i input.mp3 --preview

    # 正式切分，保留切分点两侧各 200ms 静音
    python split_by_silence.py -i input.mp3 -o out_dir --keep-silence 200

    # 切分并导出字幕（占位符文本）
    python split_by_silence.py -i input.mp3 --export-srt

    # 切分 + 本地 whisper 转录 + 自动填充字幕
    python split_by_silence.py -i input.mp3 --export-srt --transcribe \
        --whisper-model /path/to/whisper-model-or-name --whisper-language zh
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


warnings.filterwarnings(action='ignore', category=FutureWarning)
logger = logging.getLogger("split_by_silence")


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #

@dataclass
class Segment:
    """一个切分后的音频片段。

    Attributes:
        raw_start: 切分计算得到的原始起点（毫秒），未做静音裁剪。
        raw_end: 切分计算得到的原始终点（毫秒），未做静音裁剪。
        export_start: 实际导出时使用的起点（毫秒），已按 keep_silence 裁剪。
        export_end: 实际导出时使用的终点（毫秒），已按 keep_silence 裁剪。
        index: 片段序号（从 1 开始）。
        text: 转录 / 占位文本，导出字幕时使用。
    """

    raw_start: int
    raw_end: int
    export_start: int
    export_end: int
    index: int = 0
    text: str = ""

    @property
    def duration_ms(self) -> int:
        """导出片段的实际时长（毫秒）。"""
        return self.export_end - self.export_start


# --------------------------------------------------------------------------- #
# 静音检测
# --------------------------------------------------------------------------- #

def detect_silences(
    audio,
    min_silence_len: int,
    silence_thresh: float,
    seek_step: int = 1,
) -> list[list[int]]:
    """检测音频中的所有静音区间。

    Args:
        audio: pydub.AudioSegment 对象。
        min_silence_len: 判定为静音区的最小持续时长（毫秒）。
        silence_thresh: 响度阈值（dBFS），低于该值视为静音。
        seek_step: 检测时的步进（毫秒），越小越精确但越慢。

    Returns:
        静音区间列表，每项为 [start_ms, end_ms]，按起始时间升序排列，互不重叠。
    """
    from pydub.silence import detect_silence

    silences = detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=seek_step,
    )
    logger.info("检测到 %d 个静音区（min_len=%dms, thresh=%.1fdBFS）", len(silences), min_silence_len, silence_thresh)
    for s, e in silences:
        logger.debug("  静音区 %6dms - %6dms (时长 %dms)", s, e, e - s)
    return [list(x) for x in silences]


# --------------------------------------------------------------------------- #
# 切分算法
# --------------------------------------------------------------------------- #

def compute_segments(
    total_len: int,
    silences: list[list[int]],
    min_segment_len: int,
    max_segment_len: int,
) -> list[tuple[int, int]]:
    """根据静音区递归计算切分点。

    规则：
        1. 若某片段长度超过 max_segment_len，则必须在该片段内寻找“最长的静音区”，
           在其中点处切开（规则 1 + 规则 3）。
        2. 若某次切分会导致两侧任一片段短于 min_segment_len，则放弃该次切分，
           改尝试片段内次长的静音区，依次类推；若所有候选都不满足，则该片段不再切分（规则 2）。
        3. 重复以上过程直至没有片段可以再被合法切分。

    Args:
        total_len: 整段音频总长度（毫秒）。
        silences: detect_silences 得到的静音区列表 [[start, end], ...]。
        min_segment_len: 片段允许的最小长度（毫秒）。
        max_segment_len: 片段允许的最大长度（毫秒），超过则继续尝试切分。

    Returns:
        切分后的片段边界列表 [(start, end), ...]，按起始时间升序排列。
    """
    segments: list[tuple[int, int]] = [(0, total_len)]
    changed = True

    while changed:
        changed = False
        next_segments: list[tuple[int, int]] = []

        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start

            if seg_len <= max_segment_len:
                # 未超长，无需强制切分
                next_segments.append((seg_start, seg_end))
                continue

            # 找到完全落在该片段内部的静音区，按时长从长到短排序
            candidates = [s for s in silences if s[0] >= seg_start and s[1] <= seg_end]
            candidates.sort(key=lambda s: s[1] - s[0], reverse=True)

            split_done = False
            for s_start, s_end in candidates:
                mid = (s_start + s_end) // 2
                left_len = mid - seg_start
                right_len = seg_end - mid
                if left_len >= min_segment_len and right_len >= min_segment_len:
                    next_segments.append((seg_start, mid))
                    next_segments.append((mid, seg_end))
                    split_done = True
                    changed = True
                    break

            if not split_done:
                # 没有满足最小长度约束的候选静音区，放弃切分（该片段会保持超长）
                if candidates:
                    logger.warning(
                        "片段 %dms-%dms 超过最大长度但因 min_segment_len 限制无法继续切分，保留原样",
                        seg_start, seg_end,
                    )
                next_segments.append((seg_start, seg_end))

        segments = next_segments

    segments.sort(key=lambda x: x[0])
    return segments


def _find_leading_silence_end(seg_start: int, seg_end: int, silences: list[list[int]]) -> Optional[int]:
    """找到片段开头所处静音区的结束位置（若片段开头位于某个静音区内）。"""
    for s_start, s_end in silences:
        if s_start <= seg_start < s_end and s_end <= seg_end:
            return s_end
    return None


def _find_trailing_silence_start(seg_start: int, seg_end: int, silences: list[list[int]]) -> Optional[int]:
    """找到片段结尾所处静音区的起始位置（若片段结尾位于某个静音区内）。"""
    for s_start, s_end in silences:
        if s_start < seg_end <= s_end and s_start >= seg_start:
            return s_start
    return None


def apply_keep_silence(
    raw_segments: list[tuple[int, int]],
    silences: list[list[int]],
    keep_silence_ms: int,
    total_len: int,
) -> list[Segment]:
    """按 keep_silence_ms 缩短每个片段首尾保留的静音长度。

    若片段开头/结尾落在一个静音区内，且该静音区在片段内部分长于 keep_silence_ms，
    则把边界向内收缩，只保留紧邻内容的 keep_silence_ms 毫秒静音。

    Args:
        raw_segments: compute_segments 得到的原始切分边界。
        silences: 全局静音区列表。
        keep_silence_ms: 每个片段首尾希望保留的静音长度（毫秒）。
        total_len: 音频总长度（毫秒），用于边界保护。

    Returns:
        Segment 列表，包含原始边界与裁剪后导出边界。
    """
    result: list[Segment] = []
    for idx, (raw_start, raw_end) in enumerate(raw_segments, start=1):
        export_start = raw_start
        export_end = raw_end

        leading_end = _find_leading_silence_end(raw_start, raw_end, silences)
        if leading_end is not None and (leading_end - raw_start) > keep_silence_ms:
            export_start = max(raw_start, leading_end - keep_silence_ms)

        trailing_start = _find_trailing_silence_start(raw_start, raw_end, silences)
        if trailing_start is not None and (raw_end - trailing_start) > keep_silence_ms:
            export_end = min(raw_end, trailing_start + keep_silence_ms)

        export_start = max(0, min(export_start, total_len))
        export_end = max(export_start, min(export_end, total_len))

        result.append(
            Segment(
                raw_start=raw_start,
                raw_end=raw_end,
                export_start=export_start,
                export_end=export_end,
                index=idx,
            )
        )
    return result


# --------------------------------------------------------------------------- #
# 预览
# --------------------------------------------------------------------------- #

def _fmt_ms(ms: int) -> str:
    """把毫秒格式化为 HH:MM:SS.mmm，便于阅读。"""
    total_seconds, millis = divmod(ms, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def print_preview(segments: list[Segment]) -> None:
    """把切分结果以表格形式打印到控制台，不生成任何文件。"""
    print(f"\n共 {len(segments)} 个片段：\n")
    print(f"{'序号':<6}{'开始':<14}{'结束':<14}{'时长(ms)':<10}{'原始边界(裁剪前)'}")
    print("-" * 80)
    for seg in segments:
        raw_info = f"[{seg.raw_start} - {seg.raw_end}]"
        print(
            f"{seg.index:<6}{_fmt_ms(seg.export_start):<14}{_fmt_ms(seg.export_end):<14}"
            f"{seg.duration_ms:<10}{raw_info}"
        )
    print()


# --------------------------------------------------------------------------- #
# 导出音频
# --------------------------------------------------------------------------- #

def export_segments(
    audio,
    segments: list[Segment],
    output_dir: Path,
    prefix: str,
    fmt: str,
) -> list[Path]:
    """把每个片段导出为独立的音频文件。

    Args:
        audio: 完整的 pydub.AudioSegment。
        segments: 待导出的片段列表。
        output_dir: 输出目录。
        prefix: 输出文件名前缀。
        fmt: 导出格式（例如 "wav" / "mp3"）。

    Returns:
        导出文件路径列表，与 segments 一一对应。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(len(segments))))
    paths: list[Path] = []
    for seg in segments:
        clip = audio[seg.export_start:seg.export_end]
        filename = f"{prefix}_{seg.index:0{width}d}.{fmt}"
        out_path = output_dir / filename
        clip.export(out_path, format=fmt)
        logger.info("导出片段 %d: %s (%dms)", seg.index, out_path, seg.duration_ms)
        paths.append(out_path)
    return paths


# --------------------------------------------------------------------------- #
# 字幕导出
# --------------------------------------------------------------------------- #

def _srt_timestamp(ms: int) -> str:
    """把毫秒转换为 SRT 时间戳格式 HH:MM:SS,mmm。"""
    total_seconds, millis = divmod(ms, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def write_srt(segments: list[Segment], srt_path: Path) -> None:
    """把片段列表写为 SRT 字幕文件。

    时间戳使用片段导出后的起止时间；文字使用 segment.text
    （若未转录，则默认使用占位符 "[Segment N]"）。

    Args:
        segments: 待写入的片段列表。
        srt_path: 输出的 .srt 文件路径。
    """
    lines: list[str] = []
    for seg in segments:
        text = seg.text if seg.text else f"[Segment {seg.index}]"
        lines.append(str(seg.index))
        lines.append(f"{_srt_timestamp(seg.export_start)} --> {_srt_timestamp(seg.export_end)}")
        lines.append(text)
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("字幕已写入 %s", srt_path)


# --------------------------------------------------------------------------- #
# 转录（本地 Whisper，支持多种模型格式）
# --------------------------------------------------------------------------- #
#
# 本地 whisper 模型常见有三种存放形式，需要用不同的库加载：
#
#   1. openai-whisper 格式：单个 .pt 文件，或 "base"/"small" 等官方模型名
#      -> 使用 openai-whisper 库加载
#   2. HuggingFace 格式目录：一堆文件，包含 config.json +
#      pytorch_model.bin / model.safetensors + tokenizer/preprocessor 配置
#      （Whisper-Finetune 等微调项目 merge_lora 后常见就是这种目录）
#      -> 使用 transformers 库加载
#   3. CTranslate2 格式目录：包含 model.bin + config.json + vocabulary 等
#      （用 ct2-transformers-converter 转换过的模型，配合 faster-whisper 使用）
#      -> 使用 faster-whisper 库加载
#
# detect_whisper_backend() 会根据传入路径自动判断属于哪一种；也可以用
# --whisper-backend 显式指定，跳过自动判断。


def detect_whisper_backend(model_path: str) -> str:
    """根据模型路径的形态，猜测应该使用哪个库加载。

    Args:
        model_path: 模型名称或本地路径（文件或目录）。

    Returns:
        "openai-whisper" / "transformers" / "faster-whisper" 之一。
    """
    path = Path(model_path)

    if path.is_file():
        return "openai-whisper"

    if path.is_dir():
        names = {f.name for f in path.iterdir()}
        if "model.bin" in names:
            return "faster-whisper"
        if "config.json" in names:
            return "transformers"
        logger.warning("无法从目录内容判断模型格式，默认按 transformers 处理: %s", model_path)
        return "transformers"

    # 既不是已存在的文件也不是目录 -> 当作 openai-whisper 的官方模型名（如 "base"）
    return "openai-whisper"


class BaseTranscriber:
    """转录后端的统一接口，屏蔽不同库之间的差异。"""

    def transcribe(self, audio_path: Path, language: Optional[str]) -> str:
        """转录一个音频文件，返回识别出的文字。"""
        raise NotImplementedError


class OpenAIWhisperTranscriber(BaseTranscriber):
    """使用 openai-whisper 库（.pt 权重文件或官方模型名）。"""

    def __init__(self, model_path: str, device: Optional[str] = None):
        import whisper

        logger.info("使用 openai-whisper 加载模型: %s", model_path)
        self.model = whisper.load_model(model_path, device=device)

    def transcribe(self, audio_path: Path, language: Optional[str]) -> str:
        result = self.model.transcribe(str(audio_path), language=language)
        return result.get("text", "").strip()


class TransformersWhisperTranscriber(BaseTranscriber):
    """使用 HuggingFace transformers 加载 Whisper 模型目录。

    适用于 Whisper-Finetune 等项目 merge_lora 后导出的标准 HF 格式目录
    （含 config.json / pytorch_model.bin 或 model.safetensors / tokenizer 等文件）。
    """

    def __init__(self, 
        model_path: str, 
        device: Optional[str] = None, 
        use_flash_attention_2=False, 
        use_bettertransformer=False, 
        use_compile=False, 
        assistant_model_path=None,
        batch_size=16,
    ):
        import torch
        import platform
        # from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM

        logger.info("使用 transformers 加载模型目录: %s", model_path)
        # 获取Whisper的特征提取器、编码器和解码器
        self.processor = AutoProcessor.from_pretrained(model_path)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch_dtype = torch.float16 if device.startswith('cuda') and torch.cuda.is_available() else torch.float32
        # 获取模型
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True,
            use_flash_attention_2=use_flash_attention_2,
        )
        model.generation_config.forced_decoder_ids = None
        if use_bettertransformer and not use_flash_attention_2:
            model = model.to_bettertransformer()
        # 使用Pytorch2.0的编译器
        if use_compile:
            if torch.__version__ >= "2" and platform.system().lower() != 'windows':
                model = torch.compile(model)
        model.to(device)

        # 获取助手模型
        generate_kwargs_pipeline = {"max_new_tokens": 445}
        if assistant_model_path is not None:
            assistant_model = AutoModelForCausalLM.from_pretrained(
                assistant_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            assistant_model.to(device)
            generate_kwargs_pipeline = {"assistant_model": assistant_model}

        infer_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            generate_kwargs=generate_kwargs_pipeline,
            device=device
        )

        self.pipeline = infer_pipe

        # 部分微调模型（含 Whisper-Finetune 导出的模型）在 generation_config.json 里
        # 遗留了旧版的 forced_decoder_ids 字段，新版 transformers 只要检测到它非空就会
        # 直接抛 ValueError，即使我们自己不传这个参数也会触发。这里主动清空。
        # # if getattr(self.model.generation_config, "forced_decoder_ids", None) is not None:
        # #     logger.debug("清除 checkpoint 自带的 forced_decoder_ids")
        # #     self.model.generation_config.forced_decoder_ids = None

    def transcribe(self, 
        audio_path: Path, 
        language: Optional[str], 
        task='transcribe', 
        num_beams=1
    ) -> str:

        generate_kwargs = {"task": task, "num_beams": num_beams}
        if language is not None:
            generate_kwargs["language"] = language
        # 推理
        result = self.pipeline(str(audio_path), return_timestamps=True, generate_kwargs=generate_kwargs)

        txt = ""
        for chunk in result["chunks"]:
            # TODO: remove print
            print(f"[{chunk['timestamp'][0]}-{chunk['timestamp'][1]}s] {chunk['text']}")
            txt += chunk['text'] + " "

        return txt.rstrip(" ")

        import numpy as np
        import torch
        from pydub import AudioSegment

        audio = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

        inputs = self.processor(samples, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        with torch.no_grad():
            try:
                # 新版 transformers（>=4.39 左右）推荐直接传 language/task，
                # 不再支持 forced_decoder_ids。
                generate_kwargs = {}
                if language:
                    generate_kwargs["language"] = language
                    generate_kwargs["task"] = "transcribe"
                generated_ids = self.model.generate(input_features, **generate_kwargs)
            except (TypeError, ValueError) as exc:
                logger.debug("新版 generate(language=...) 调用失败，回退到 forced_decoder_ids: %s", exc)
                forced_decoder_ids = None
                if language:
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
                generated_ids = self.model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()


class FasterWhisperTranscriber(BaseTranscriber):
    """使用 faster-whisper（CTranslate2 格式模型目录）。"""

    def __init__(self, model_path: str, device: Optional[str] = None):
        from faster_whisper import WhisperModel

        logger.info("使用 faster-whisper 加载模型目录: %s", model_path)
        self.model = WhisperModel(model_path, device=device or "auto")

    def transcribe(self, audio_path: Path, language: Optional[str]) -> str:
        segments, _info = self.model.transcribe(str(audio_path), language=language)
        return "".join(seg.text for seg in segments).strip()


def load_transcriber(
    model_path: str,
    backend: str = "auto",
    device: Optional[str] = None,
) -> BaseTranscriber:
    """按指定（或自动检测的）后端加载转录器。

    Args:
        model_path: 模型名称或本地路径（.pt 文件 / HF 目录 / CTranslate2 目录）。
        backend: "auto" / "openai-whisper" / "transformers" / "faster-whisper"。
        device: 运行设备，默认自动选择。

    Returns:
        实现了 .transcribe(path, language) 的转录器实例。
    """
    if backend == "auto":
        backend = detect_whisper_backend(model_path)
        logger.info("自动检测到模型格式，使用后端: %s", backend)

    if backend == "openai-whisper":
        return OpenAIWhisperTranscriber(model_path, device)
    if backend == "transformers":
        return TransformersWhisperTranscriber(model_path, device)
    if backend == "faster-whisper":
        return FasterWhisperTranscriber(model_path, device)

    raise ValueError(f"未知的 whisper backend: {backend}")


def transcribe_segments(
    transcriber: BaseTranscriber,
    audio_paths: list[Path],
    segments: list[Segment],
    language: Optional[str],
) -> None:
    """对每个已导出的音频片段调用转录器，结果写回 segment.text。

    Args:
        transcriber: load_transcriber 返回的转录器实例。
        audio_paths: export_segments 返回的文件路径列表，与 segments 一一对应。
        segments: 待填充文本的片段列表（原地修改）。
        language: 转录语言代码（如 "zh" / "en"），为 None 时自动检测（不同后端支持程度不同）。
    """
    for seg, path in zip(segments, audio_paths):
        logger.info("转录片段 %d: %s", seg.index, path)
        text = transcriber.transcribe(path, language)
        seg.text = text
        logger.debug("  -> %s", text)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def process_file(args: argparse.Namespace, input_path: Path) -> None:
    """处理单个音频文件：检测静音、切分、（可选）预览/导出/字幕/转录。

    Args:
        args: 解析后的命令行参数。
        input_path: 待处理的音频文件路径。
    """
    from pydub import AudioSegment

    logger.info("处理文件: %s", input_path)
    audio = AudioSegment.from_file(input_path)
    total_len = len(audio)

    silence_thresh = args.silence_thresh
    if silence_thresh is None:
        silence_thresh = audio.dBFS - 16
        logger.info("未指定 --silence-thresh，使用自动阈值 %.1f dBFS (audio.dBFS - 16)", silence_thresh)

    silences = detect_silences(
        audio,
        min_silence_len=args.min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=args.seek_step,
    )

    raw_segments = compute_segments(
        total_len=total_len,
        silences=silences,
        min_segment_len=args.min_segment_len,
        max_segment_len=args.max_segment_len,
    )

    segments = apply_keep_silence(
        raw_segments=raw_segments,
        silences=silences,
        keep_silence_ms=args.keep_silence,
        total_len=total_len,
    )

    if args.preview:
        print_preview(segments)
        if args.transcribe:
            logger.warning("预览模式下不会执行转录，如需转录请去掉 --preview")
        return

    output_dir = args.output_dir or (input_path.parent / f"{input_path.stem}_segments")
    fmt = args.format or (input_path.suffix.lstrip(".") or "wav")
    prefix = args.prefix or input_path.stem

    audio_paths = export_segments(audio, segments, output_dir, prefix, fmt)

    if args.transcribe:
        if not args.whisper_model:
            logger.error("启用了 --transcribe 但未指定 --whisper-model，跳过转录")
        else:
            transcriber = load_transcriber(
                args.whisper_model,
                backend=args.whisper_backend,
                device=args.whisper_device,
            )
            transcribe_segments(transcriber, audio_paths, segments, args.whisper_language)

    if args.export_srt:
        srt_path = args.srt_path or (output_dir / f"{prefix}.srt")
        write_srt(segments, srt_path)

    print(f"\n完成：{len(segments)} 个片段已导出至 {output_dir}\n")


def expand_inputs(patterns: list[str]) -> list[Path]:
    """展开命令行传入的文件路径（兼容 Windows 下 shell 不自动展开通配符的情况）。

    Args:
        patterns: 用户传入的路径 / 通配符列表。

    Returns:
        去重后的文件路径列表。
    """
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            # 不是通配符，或没有匹配到，直接当作字面路径使用
            paths.append(Path(pattern))
    # 去重并保持顺序
    seen = set()
    unique_paths = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    return unique_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="按静音区切分音频，支持预览、字幕导出与本地 Whisper 转录。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", nargs="+", required=True,
        help="输入音频文件路径，可传多个或使用通配符（如 *.mp3）",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="输出目录，默认在输入文件旁创建 <文件名>_segments")
    parser.add_argument("--prefix", type=str, default=None, help="输出文件名前缀，默认使用输入文件名（不含扩展名）")
    parser.add_argument("--format", type=str, default=None, help="输出音频格式，默认与输入文件相同")

    # 静音检测参数
    parser.add_argument("--min-silence-len", type=int, default=500, help="判定为静音区的最小长度（毫秒）")
    parser.add_argument("--silence-thresh", type=float, default=None, help="静音响度阈值（dBFS）。默认自动计算为 audio.dBFS - 16")
    parser.add_argument("--seek-step", type=int, default=1, help="静音检测步进（毫秒），越小越精确但越慢")

    # 切分规则参数
    parser.add_argument("--min-segment-len", type=int, default=3000, help="片段最小长度（毫秒），低于该值不再继续切分")
    parser.add_argument("--max-segment-len", type=int, default=15000, help="片段最大长度（毫秒），超过该值会继续按最长静音切分")
    parser.add_argument("--keep-silence", type=int, default=300, help="切分后每个片段首尾保留的静音长度（毫秒）")

    # 预览
    parser.add_argument("--preview", action="store_true", help="仅预览切分位置和片段时长，不生成文件")

    # 字幕
    parser.add_argument("--export-srt", action="store_true", help="导出 SRT 字幕文件")
    parser.add_argument("--srt-path", type=Path, default=None, help="字幕输出路径，默认在输出目录下 <前缀>.srt")

    # 转录
    parser.add_argument("--transcribe", action="store_true", help="调用本地 Whisper 模型转录每个片段")
    parser.add_argument(
        "--whisper-model", type=str, default=None,
        help="Whisper 模型路径：官方模型名(如 base) / 本地 .pt 文件 / "
             "HuggingFace 格式目录(如 Whisper-Finetune merge 后的目录) / CTranslate2 格式目录",
    )
    parser.add_argument(
        "--whisper-backend", type=str, default="auto",
        choices=["auto", "openai-whisper", "transformers", "faster-whisper"],
        help="加载模型使用的库。auto 会根据 --whisper-model 路径自动判断",
    )
    parser.add_argument("--whisper-language", type=str, default=None, help="转录语言代码，如 zh / en，默认自动检测")
    parser.add_argument("--whisper-device", type=str, default=None, help="运行 Whisper 的设备，如 cpu / cuda，默认自动选择")

    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细调试日志")

    return parser


def main() -> None:
    """命令行入口。"""
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.min_segment_len * 2 > args.max_segment_len:
        logger.warning(
            "min_segment_len (%d) 的两倍已超过 max_segment_len (%d)，超长片段可能无法被切分",
            args.min_segment_len, args.max_segment_len,
        )

    input_paths = expand_inputs(args.input)
    if not input_paths:
        logger.error("未找到任何匹配的输入文件")
        sys.exit(1)

    for input_path in input_paths:
        if not input_path.exists():
            logger.error("文件不存在，跳过: %s", input_path)
            continue
        try:
            process_file(args, input_path)
        except Exception:
            logger.exception("处理文件失败: %s", input_path)


if __name__ == "__main__":
    main()

