"""
功能：根据字幕文件（支持 .srt / .ass 两种格式，按文件后缀自动识别）切分音频，
     并可对切分后片段的首尾静音进行填充或移除处理。
"""
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from glob import glob
from typing import List, Dict, Tuple

try:
    import pysrt
    from pydub import AudioSegment
    from pydub.silence import detect_leading_silence
    from tqdm import tqdm
except ImportError:
    print("Lacking moduless, please run `pip install pydub pysrt tqdm`")
    raise


_version = '1.7.1'

total_time = 0

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """将秒数格式化为 SRT 时间戳字符串 (HH:MM:SS,mmm)。

    Args:
        seconds (float): 时长，单位秒。

    Returns:
        str: 格式化后的时间字符串。
    """
    millis = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes = seconds // 60
    seconds = seconds % 60
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def do_skip_line(line_text: str) -> bool:
    """Skip a line when condition is True

    Args:
        line_text (str): text of one line

    Returns:
        bool: True or False
    """
    if line_text.strip().startswith('*'):
        return True
    if line_text.strip().startswith('('):
        return True
    return False


@dataclass
class SubtitleItem:
    """统一的字幕条目结构，屏蔽 srt / ass 两种格式的差异。

    Attributes:
        start (int): 字幕开始时间，单位毫秒。
        end (int): 字幕结束时间，单位毫秒。
        text (str): 字幕文本内容。
    """
    start: int
    end: int
    text: str


def load_srt_subs(srt_path: Path) -> List[SubtitleItem]:
    """解析 .srt 字幕文件，转换为统一的 SubtitleItem 列表。

    Args:
        srt_path (Path): .srt 字幕文件路径。

    Returns:
        List[SubtitleItem]: 按出现顺序排列的字幕条目列表。
    """
    subs = pysrt.open(srt_path)
    return [
        SubtitleItem(start=sub.start.ordinal, end=sub.end.ordinal, text=sub.text)
        for sub in subs
    ]


# 匹配 ass 文本中的样式覆盖标签，如 {\an8}、{\pos(0,0)} 等
_ASS_OVERRIDE_TAG_RE = re.compile(r'\{.*?\}')


def _parse_ass_time(time_str: str) -> int:
    """将 ass 时间字符串（H:MM:SS.cc，cc 为百分之一秒）转换为毫秒整数。

    Args:
        time_str (str): ass 格式的时间字符串。

    Returns:
        int: 对应的毫秒数。
    """
    hours, minutes, sec_cs = time_str.strip().split(':')
    seconds, centiseconds = sec_cs.split('.')
    total_ms = (
        (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000
        + int(centiseconds) * 10
    )
    return total_ms


def _clean_ass_text(raw_text: str) -> str:
    """清理 ass 字幕文本：去除样式覆盖标签，并将换行符替换为空格。

    Args:
        raw_text (str): ass Dialogue 行中的原始 Text 字段。

    Returns:
        str: 清理后的纯文本。
    """
    text = _ASS_OVERRIDE_TAG_RE.sub('', raw_text)
    text = text.replace('\\N', ' ').replace('\\n', ' ')
    return text.strip()


def load_ass_subs(ass_path: Path) -> List[SubtitleItem]:
    """解析 .ass 字幕文件，转换为统一的 SubtitleItem 列表。

    仅解析 [Events] 小节下的 Dialogue 行，会根据该小节的 Format 行动态定位
    Start / End / Text 字段的位置，以兼容不同工具导出的字段顺序。

    Args:
        ass_path (Path): .ass 字幕文件路径。

    Returns:
        List[SubtitleItem]: 按开始时间排序的字幕条目列表。
    """
    subs: List[SubtitleItem] = []
    in_events = False
    # Dialogue 标准字段顺序: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
    start_idx, end_idx, text_idx = 1, 2, 9

    with open(ass_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('['):
                in_events = stripped.lower() == '[events]'
                continue
            if not in_events:
                continue

            if stripped.lower().startswith('format:'):
                # 根据 Format 行重新定位字段索引，兼容自定义字段顺序
                fields = [f.strip().lower() for f in stripped[len('format:'):].split(',')]
                start_idx = fields.index('start')
                end_idx = fields.index('end')
                text_idx = fields.index('text')
                continue

            if stripped.lower().startswith('dialogue:'):
                # Text 字段本身可能包含逗号，因此按 Format 中 Text 的位置限制切分次数
                parts = stripped[len('dialogue:'):].split(',', text_idx)
                if len(parts) <= max(start_idx, end_idx, text_idx):
                    continue
                start_ms = _parse_ass_time(parts[start_idx])
                end_ms = _parse_ass_time(parts[end_idx])
                text = _clean_ass_text(parts[text_idx])
                subs.append(SubtitleItem(start=start_ms, end=end_ms, text=text))

    # ass 文件中的 Dialogue 行不一定按时间顺序排列，这里统一按开始时间排序
    subs.sort(key=lambda item: item.start)
    return subs


def load_subtitles(sub_path: Path) -> List[SubtitleItem]:
    """根据文件后缀自动选择解析器，加载字幕文件（支持 .srt 和 .ass）。

    Args:
        sub_path (Path): 字幕文件路径。

    Returns:
        List[SubtitleItem]: 统一格式的字幕条目列表。

    Raises:
        ValueError: 当字幕文件后缀不受支持时抛出。
    """
    suffix = sub_path.suffix.lower()
    if suffix == '.srt':
        return load_srt_subs(sub_path)
    elif suffix == '.ass':
        return load_ass_subs(sub_path)
    else:
        raise ValueError(f"不支持的字幕格式: {suffix}，仅支持 .srt 和 .ass")


def _random_target_duration(target_ms: int, jitter_ms: int) -> int:
    """在目标静音时长基础上，添加一个 [-jitter_ms, +jitter_ms] 范围内的随机浮动。

    Args:
        target_ms (int): 目标静音时长，单位毫秒。
        jitter_ms (int): 随机浮动的最大幅度，单位毫秒。取 0 表示不浮动。

    Returns:
        int: 随机化后的目标时长（毫秒），保证不小于 0。
    """
    if jitter_ms <= 0:
        return max(0, target_ms)
    offset = random.uniform(-jitter_ms, jitter_ms)
    return max(0, int(round(target_ms + offset)))


def _detect_edge_silences(
    segment: AudioSegment, silence_thresh: float, chunk_size: int
) -> Tuple[int, int]:
    """检测音频片段头部与尾部的静音时长。

    Args:
        segment (AudioSegment): 待检测的音频片段。
        silence_thresh (float): 判定为静音的音量阈值（dBFS）。
        chunk_size (int): 静音检测时每次扫描的采样块大小（毫秒）。

    Returns:
        Tuple[int, int]: (头部静音时长, 尾部静音时长)，单位毫秒。
    """
    leading_silence = detect_leading_silence(segment, silence_thresh, chunk_size)
    # 通过反转音频，复用 detect_leading_silence 来检测尾部静音
    trailing_silence = detect_leading_silence(segment.reverse(), silence_thresh, chunk_size)
    return leading_silence, trailing_silence


def adjust_edge_silence(
    segment: AudioSegment,
    action: str,
    target_ms: int,
    jitter_ms: int,
    silence_thresh: float = -50.0,
    chunk_size: int = 10,
) -> AudioSegment:
    """对音频片段首尾的静音区域进行填充或移除处理。

    - 填充(pad)：若原有静音时长已经 >= 目标时长，则不做任何处理；
      否则在该端补齐静音，使其达到（随机浮动后的）目标时长。
    - 移除(remove)：若原有静音时长已经 <= 目标时长，则不做任何处理；
      否则裁剪该端静音，使其缩短至（随机浮动后的）目标时长。

    头部和尾部分别独立计算随机目标时长，因此两端的静音时长可能不同。

    Args:
        segment (AudioSegment): 原始音频片段。
        action (str): 处理动作，'none' 不处理，'pad' 填充静音，'remove' 移除静音。
        target_ms (int): 目标静音时长，单位毫秒。
        jitter_ms (int): 目标时长随机浮动的最大幅度，单位毫秒。
        silence_thresh (float, optional): 静音判定阈值（dBFS）。默认 -50.0。
        chunk_size (int, optional): 静音检测采样块大小（毫秒）。默认 10。

    Returns:
        AudioSegment: 处理后的音频片段。
    """
    if action not in ("pad", "remove"):
        return segment

    leading_silence, trailing_silence = _detect_edge_silences(
        segment, silence_thresh, chunk_size
    )

    # 头尾各自独立随机化目标时长，避免两端静音时长完全一致
    head_target = _random_target_duration(target_ms, jitter_ms)
    tail_target = _random_target_duration(target_ms, jitter_ms)

    if action == "pad":
        # 头部：现有静音不足目标时长时才补齐，否则跳过
        if leading_silence < head_target:
            pad_amount = head_target - leading_silence
            head_pad = AudioSegment.silent(
                duration=pad_amount, frame_rate=segment.frame_rate
            )
            segment = head_pad + segment
        # 尾部：现有静音不足目标时长时才补齐，否则跳过
        if trailing_silence < tail_target:
            pad_amount = tail_target - trailing_silence
            tail_pad = AudioSegment.silent(
                duration=pad_amount, frame_rate=segment.frame_rate
            )
            segment = segment + tail_pad
    elif action == "remove":
        # 头部：现有静音超过目标时长时才裁剪，否则跳过
        if leading_silence > head_target:
            trim_amount = leading_silence - head_target
            segment = segment[trim_amount:]
        # 尾部：现有静音超过目标时长时才裁剪，否则跳过
        if trailing_silence > tail_target:
            trim_amount = trailing_silence - tail_target
            segment = segment[: len(segment) - trim_amount]

    return segment


def segment_audio(
    audio_path: Path,
    sub_path: Path,
    root_path: Path,
    independent: bool = True,
    only_metadata: bool = False,
    silence_action: str = "none",
    silence_duration: int = 300,
    silence_jitter: int = 50,
    silence_thresh: float = -50.0,
    silence_chunk_size: int = 10,
) -> List[str]:
    """根据字幕文件切分音频（支持 .srt 和 .ass，按文件后缀自动识别）

    Args:
        audio_path (Path): 音频文件
        sub_path (Path): 字幕文件，支持 .srt 或 .ass 格式
        root_path (Path): 保存所有音频文件的总目录
        independent (bool, optional): 是否将每个音频文件保存至单独目录. 默认 True.
        only_metadata (bool, optional): 是否仅生成 metadata，不导出音频文件. 默认 False.
        silence_action (str, optional): 首尾静音处理动作，'none'/'pad'/'remove'. 默认 'none'.
        silence_duration (int, optional): 静音处理的目标时长，单位毫秒. 默认 300.
        silence_jitter (int, optional): 目标时长随机浮动的最大幅度，单位毫秒. 默认 50.
        silence_thresh (float, optional): 静音判定阈值（dBFS）. 默认 -50.0.
        silence_chunk_size (int, optional): 静音检测采样块大小（毫秒）. 默认 10.

    Returns:
        list: 列表，元素为一行文本
    """
    metadata_list = []
    global total_time
    # audioPath = Path(audio_path)
    if not sub_path.exists() or not audio_path.exists():
        logger.warning("%s 或 %s 不存在", audio_path, sub_path)
        return metadata_list

    audioFileName = audio_path.stem

    # 读取字幕文件，根据后缀自动识别 srt / ass 格式
    subs = load_subtitles(sub_path)

    # 加载音频文件
    logger.info("Loading audio %s", audio_path)
    audio = AudioSegment.from_file(audio_path)

    sub_info_list = []
    sub_info = {}
    # 处理第一个sub：起始时间直接取字幕开始时间（不再额外增加 800ms 提前量）
    sub = subs[0]
    next_sub = subs[1]

    sub_info["start_time"] = sub.start
    # 结束时间取当前字幕与下一条字幕之间的中点，不再限制最大 800ms 的延展范围
    sub_info["end_time"] = int((sub.end + next_sub.start) / 2)
    sub_info["text"] = sub.text
    sub_info_list.append(sub_info)
    # 处理中间的sub
    for i in range(1, len(subs) - 1):
        pre_sub = subs[i - 1]
        sub = subs[i]
        next_sub = subs[i + 1]
        if do_skip_line(sub.text):
            continue

        sub_info = {}
        # 起始/结束时间均取相邻字幕之间的中点，不再限制最大 800ms 的延展范围
        sub_info["start_time"] = int((pre_sub.end + sub.start) / 2)
        sub_info["end_time"] = int((sub.end + next_sub.start) / 2)
        sub_info["text"] = sub.text
        sub_info_list.append(sub_info)

    # 处理最后一个sub
    pre_sub = subs[-2]
    sub = subs[-1]
    sub_info = {}
    sub_info["start_time"] = int((pre_sub.end + sub.start) / 2)
    # 结束时间取当前字幕结束时间与音频总时长之间的中点，不再限制最大 800ms 的延展范围
    sub_info["end_time"] = int((sub.end + (audio.duration_seconds * 1000)) / 2)
    sub_info["text"] = sub.text
    sub_info_list.append(sub_info)

    if independent:
        wav_folder = root_path / 'wavs' / audioFileName
    else:
        wav_folder = root_path / 'wavs'

    wav_folder.mkdir(exist_ok=True, parents=True)
    for i, sub_info in tqdm(enumerate(sub_info_list, start=1), total=len(sub_info_list)):
        total_time += sub_info["end_time"] - sub_info["start_time"]
        file_name = f"{audioFileName}_{i}".replace(" ", "_") + '.wav'
        output_file = wav_folder / file_name
        if not only_metadata:
            # 切分音频片段
            audio_segment = audio[sub_info["start_time"]:sub_info["end_time"]]
            # 按需对片段首尾静音进行填充或移除
            if silence_action != "none":
                audio_segment = adjust_edge_silence(
                    audio_segment,
                    action=silence_action,
                    target_ms=silence_duration,
                    jitter_ms=silence_jitter,
                    silence_thresh=silence_thresh,
                    chunk_size=silence_chunk_size,
                )
            audio_segment.export(output_file, format="wav", parameters=["-c:a", "pcm_s24le"])
        if independent:
            metadata_list.append(f"{audioFileName}/{file_name}|{sub_info['text']}")
        else:
            metadata_list.append(f"{file_name}|{sub_info['text']}")

    return metadata_list


def main(args):
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)

    sub_file_list = []
    for pattern in args.srt_files:
        # 兼容 Windows 下 shell 不自动展开通配符的情况，手动展开
        if '*' in pattern or '?' in pattern:
            sub_file_list.extend(list(glob(pattern)))
        else:
            sub_file_list.append(pattern)

    for sub_file_path in sub_file_list:
        sub_path = Path(sub_file_path)
        audio_path = sub_path.with_suffix('.wav')
        metadata_list = segment_audio(
            audio_path,
            sub_path,
            output_path,
            args.keep_folder,
            args.meta_only,
            silence_action=args.silence_action,
            silence_duration=args.silence_duration,
            silence_jitter=args.silence_jitter,
            silence_thresh=args.silence_thresh,
            silence_chunk_size=args.silence_chunk_size,
        )

        with open(output_path / "metadata.csv", "a", encoding="utf8") as f:
            for data in metadata_list:
                f.write(data + "\n")

    global total_time
    logger.info(format_time(total_time / 1000))
    logger.info("完成-------")


if __name__ == '__main__':
    from argparse import ArgumentParser
    # Arguments list
    parser = ArgumentParser(description='根据字幕文件切割音频（支持 .srt / .ass，按文件后缀自动识别）')
    parser.add_argument('-v', '--version', action='version', version=f"version: {_version}", help='显示版本并退出')
    parser.add_argument('-o', '--output', default='output', help='输出目录，缺省 output/')
    parser.add_argument('-k', '--keep-folder', action='store_true', help='保存音频文件至独立目录')
    parser.add_argument('--meta-only', action='store_true', help='仅生成 metadata.csv')

    # 首尾静音处理相关参数
    parser.add_argument(
        '--silence-action',
        choices=['none', 'pad', 'remove'],
        default='none',
        help='切分片段首尾静音处理动作：none 不处理，pad 填充静音，remove 移除静音。缺省 none',
    )
    parser.add_argument(
        '--silence-duration',
        type=int,
        default=480,
        help='静音处理的目标时长（毫秒）。填充时若原有静音已达到该时长则跳过；'
             '移除时若原有静音已短于该时长则跳过。缺省 480',
    )
    parser.add_argument(
        '--silence-jitter',
        type=int,
        default=20,
        help='目标静音时长的随机浮动幅度（毫秒），实际目标为 [duration-jitter, duration+jitter] 内的随机值。缺省 20',
    )
    parser.add_argument(
        '--silence-thresh',
        type=float,
        default=-45.0,
        help='静音判定阈值（dBFS），音量低于该值视为静音。缺省 -45.0',
    )
    parser.add_argument(
        '--silence-chunk-size',
        type=int,
        default=10,
        help='静音检测时的采样块大小（毫秒），值越小检测越精细但速度越慢。缺省 10',
    )

    parser.add_argument(
        'srt_files',
        metavar='SUB_FILES',
        type=str,
        nargs='+',
        help='来源的字幕文件，支持 .srt 和 .ass 格式（按文件后缀自动识别）',
    )

    args = parser.parse_args()
    main(args)