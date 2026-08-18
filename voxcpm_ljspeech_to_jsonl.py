#!/usr/bin/env python3
"""
LJSpeech 数据集转换工具

将 LJSpeech 格式的音频数据集转换为 JSONL 格式。

设计说明：
    - 默认情况下，所有解析到的样本都视为训练数据（不再从总量中划分出独立的参考集）。
    - 当指定 --ref-audio-ratio 时，会从训练数据本身抽取一部分样本作为某条训练样本的
      ref_audio（参考音频），因此“参考数据”与“训练数据”是同一份数据，允许重叠。
    - 出于说话人一致性的考虑，为某条训练样本挑选 ref_audio 时，只会从与该样本
      **音频文件位于同一目录** 的训练样本中挑选（不同目录通常代表不同说话人）。
    - 为了让所有目录下的训练样本都有机会被分配到 ref_audio，脚本按“音频所在目录”
      分组处理，分组内部独立抽样，最后再统一汇总为一份 JSONL。

依赖安装（使用 uv 管理依赖）：
    uv add tqdm

    如果不使用 uv，也可以：
    pip install tqdm

用法示例：
    python voxcpm_ljspeech_to_jsonl.py --input ./LJSpeech-1.1 --output ./dataset.jsonl
    python voxcpm_ljspeech_to_jsonl.py --input ./LJSpeech-1.1 --output ./dataset.jsonl --ref-audio-ratio 0.40
    python voxcpm_ljspeech_to_jsonl.py --input ./LJSpeech-1.1 --output ./dataset.jsonl --ref-audio-ratio 0.40 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ──────────────────────────────────────────────
# LJSpeech 格式说明
# ──────────────────────────────────────────────
# 目录结构：
#   <dataset_dir>/
#       wavs/               ← 音频文件（.wav），允许包含子目录（如按说话人分子目录）
#       metadata.csv        ← 格式：filename|text|normalized_text
#                             （无表头，| 分隔，第三列可选；filename 可包含子目录前缀，
#                              例如 "speaker1/LJ001-0001"）
# ──────────────────────────────────────────────

logger = logging.getLogger("ljspeech_to_jsonl")


def setup_logging(verbose: bool = False) -> None:
    """配置全局日志格式与级别。

    Args:
        verbose: 是否输出 DEBUG 级别日志。
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stdout,
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(
        description="将 LJSpeech 数据集转换为 JSONL 格式，并按目录（说话人）抽取参考音频。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 必填参数
    parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="DIR",
        help="LJSpeech 数据集根目录（包含 metadata.csv 和 wavs/ 子目录）",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="输出的 JSONL 文件路径",
    )

    # 参考音频比例
    parser.add_argument(
        "--ref-audio-ratio",
        type=float,
        default=0.40,
        metavar="FLOAT",
        help="每个目录内，添加 ref_audio 字段的样本比例（默认 0.40，即 40%%）。"
             "设为 0 则不添加 ref_audio 字段。",
    )
    parser.add_argument(
        "--allow-self-reference",
        action="store_true",
        help="允许某条样本的 ref_audio 指向它自己（默认不允许；"
             "仅当所在目录只有 1 个文件时才会退化为使用自身）",
    )

    # 音频路径风格
    parser.add_argument(
        "--audio-ext",
        default=".wav",
        metavar="EXT",
        help="音频文件扩展名（默认 .wav）",
    )
    parser.add_argument(
        "--absolute-path",
        action="store_true",
        help="在 JSONL 中写入音频的绝对路径（默认写相对于数据集根目录的相对路径）",
    )

    # 输出顺序
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="不打乱最终输出顺序（默认会打乱，使不同目录/说话人的样本混合分布）",
    )

    # 随机种子
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="随机种子，保证结果可复现（默认不固定）",
    )

    # 其他
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="不显示进度条",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出更详细的日志（包含每个目录的处理情况）",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────
# 解析 metadata.csv
# ──────────────────────────────────────────────

def load_metadata(dataset_dir: Path, audio_ext: str, absolute: bool) -> list[dict[str, str]]:
    """读取 metadata.csv，返回样本列表。

    Args:
        dataset_dir: 数据集根目录（包含 metadata.csv 和 wavs/）。
        audio_ext: 音频文件扩展名，如 ".wav"。
        absolute: 是否在返回的 audio 字段中使用绝对路径。

    Returns:
        list[dict[str, str]]: 每个元素形如
            {"audio": str, "text": str, "normalized": str}
    """
    metadata_path = dataset_dir / "metadata.csv"
    wavs_dir = dataset_dir / "wavs"

    if not metadata_path.exists():
        sys.exit(f"[错误] 找不到 metadata.csv：{metadata_path}")
    if not wavs_dir.exists():
        sys.exit(f"[错误] 找不到 wavs 目录：{wavs_dir}")

    samples: list[dict[str, str]] = []
    missing = 0

    with open(metadata_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 2:
                logger.warning("[警告] 第 %d 行格式异常，已跳过：%r", lineno, line)
                continue

            filename = parts[0].strip()
            text = parts[1].strip()
            normalized = parts[2].strip() if len(parts) >= 3 else text

            # 拼接音频路径（filename 可能已带子目录前缀，如 "speaker1/LJ001-0001"）
            audio_file = wavs_dir / (filename + audio_ext)
            if not audio_file.exists():
                # 也尝试文件名本身已经带扩展名的情况
                audio_file_alt = wavs_dir / filename
                if audio_file_alt.exists():
                    audio_file = audio_file_alt
                else:
                    missing += 1
                    continue

            if absolute:
                audio_path = str(audio_file.resolve())
            else:
                audio_path = str(audio_file.relative_to(dataset_dir))

            samples.append({
                "audio": audio_path,
                "text": text,
                "normalized": normalized,
            })

    if missing:
        logger.warning("[警告] 共 %d 条记录找不到对应音频文件，已跳过。", missing)

    if not samples:
        sys.exit("[错误] 未解析到任何有效样本，请检查数据集目录结构。")

    return samples


# ──────────────────────────────────────────────
# 按目录分组 + 组内抽取参考音频
# ──────────────────────────────────────────────

def group_by_directory(samples: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """按音频文件所在目录对样本分组。

    不同目录通常代表不同说话人，分组后可保证参考音频只在同一说话人内部抽取。

    Args:
        samples: load_metadata 返回的样本列表。

    Returns:
        dict[str, list[dict[str, str]]]: 目录路径（字符串）-> 该目录下的样本列表。
    """
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample in samples:
        directory = str(Path(sample["audio"]).parent)
        groups[directory].append(sample)
    return groups


def assign_ref_audio(
    group_samples: list[dict[str, str]],
    ref_audio_ratio: float,
    rng: random.Random,
    allow_self_reference: bool,
    directory_name: str,
) -> list[dict[str, Any]]:
    """在单个目录（单个说话人）内部，为部分样本分配 ref_audio。

    Args:
        group_samples: 同一目录下的样本列表。
        ref_audio_ratio: 需要添加 ref_audio 字段的样本比例。
        rng: 随机数生成器实例，保证可复现。
        allow_self_reference: 是否允许样本引用自身作为 ref_audio。
        directory_name: 当前目录名，仅用于日志提示。

    Returns:
        list[dict[str, Any]]: 每条记录包含 "audio"、"text"，并可能包含 "ref_audio"。
    """
    n = len(group_samples)
    records: list[dict[str, Any]] = [
        {"audio": s["audio"], "text": s["normalized"]} for s in group_samples
    ]

    if n == 0 or ref_audio_ratio <= 0:
        return records

    count = min(n, round(n * ref_audio_ratio))
    if count == 0:
        return records

    selected_indices = rng.sample(range(n), count)
    audio_pool = [s["audio"] for s in group_samples]

    self_ref_fallback = False
    for idx in selected_indices:
        if allow_self_reference or n == 1:
            candidates = audio_pool
            if n == 1:
                self_ref_fallback = True
        else:
            # 排除自身，从同目录其余样本中随机挑选
            candidates = audio_pool[:idx] + audio_pool[idx + 1:]

        records[idx]["ref_audio"] = rng.choice(candidates)

    if self_ref_fallback:
        logger.debug(
            "[调试] 目录 %r 下只有 1 个文件，ref_audio 退化为使用自身。",
            directory_name,
        )

    return records


# ──────────────────────────────────────────────
# 写 JSONL
# ──────────────────────────────────────────────

def write_jsonl(records: list[dict[str, Any]], path: Path, desc: str, show_progress: bool) -> None:
    """将记录列表写为 JSONL 文件。

    Args:
        records: 待写出的记录列表。
        path: 输出文件路径。
        desc: 进度条描述文本。
        show_progress: 是否显示进度条。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = records
    if show_progress and HAS_TQDM:
        iterator = tqdm(records, desc=desc, unit="条")

    with open(path, "w", encoding="utf-8") as f:
        for rec in iterator:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main() -> None:
    """脚本入口：解析参数、读取数据集、按目录分配参考音频并写出 JSONL。"""
    args = parse_args()
    setup_logging(args.verbose)

    # 参数校验
    if not (0 <= args.ref_audio_ratio <= 1):
        sys.exit("[错误] --ref-audio-ratio 必须在 [0, 1] 之间")

    dataset_dir = Path(args.input).resolve()
    if not dataset_dir.is_dir():
        sys.exit(f"[错误] 输入目录不存在：{dataset_dir}")

    show_progress = not args.no_progress
    if show_progress and not HAS_TQDM:
        logger.info("[提示] 未安装 tqdm，进度条不可用。可运行 uv add tqdm（或 pip install tqdm）安装。")

    rng = random.Random(args.seed)
    if args.seed is not None:
        logger.info("[信息] 随机种子：%d", args.seed)

    # ── 1. 读取数据集（全部视为训练数据）──
    logger.info("[信息] 正在解析数据集：%s", dataset_dir)
    samples = load_metadata(dataset_dir, args.audio_ext, args.absolute_path)
    total = len(samples)
    logger.info("[信息] 共读取 %d 条有效样本（全部作为训练数据）", total)

    # ── 2. 按音频所在目录分组（不同目录 = 不同说话人）──
    groups = group_by_directory(samples)
    logger.info("[信息] 共检测到 %d 个目录（说话人）", len(groups))

    # ── 3. 分组内独立抽取 ref_audio，再统一汇总 ──
    all_records: list[dict[str, Any]] = []
    total_ref_audio_count = 0

    for directory, group_samples in sorted(groups.items()):
        group_records = assign_ref_audio(
            group_samples=group_samples,
            ref_audio_ratio=args.ref_audio_ratio,
            rng=rng,
            allow_self_reference=args.allow_self_reference,
            directory_name=directory,
        )
        group_ref_count = sum(1 for r in group_records if "ref_audio" in r)
        total_ref_audio_count += group_ref_count

        logger.debug(
            "[调试] 目录 %r：%d 条样本，%d 条含 ref_audio",
            directory, len(group_records), group_ref_count,
        )

        all_records.extend(group_records)

    # ── 4. 是否打乱最终顺序 ──
    if not args.no_shuffle:
        rng.shuffle(all_records)

    ratio_actual = total_ref_audio_count / total * 100 if total else 0.0
    logger.info(
        "[信息] 共 %d 条训练样本含 ref_audio 字段（%.1f%%，目标比例 %.1f%%）",
        total_ref_audio_count, ratio_actual, args.ref_audio_ratio * 100,
    )

    # ── 5. 写出 JSONL ──
    output_path = Path(args.output)
    write_jsonl(all_records, output_path, "写出数据集", show_progress)
    logger.info("[完成] 数据集已保存至：%s（%d 条）", output_path, len(all_records))

    # ── 6. 打印简要统计 ──
    logger.info("\n── 统计摘要 ──────────────────────────────")
    logger.info("  总样本数（训练集）      : %d", total)
    logger.info("  目录（说话人）数        : %d", len(groups))
    logger.info(
        "  含 ref_audio (%.1f%%)     : %d",
        args.ref_audio_ratio * 100, total_ref_audio_count,
    )
    logger.info("──────────────────────────────────────────")


if __name__ == "__main__":
    main()
