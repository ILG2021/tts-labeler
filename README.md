# TTS Labeler

用于制作 TTS / ASR 语音数据集的一组 Python 命令行工具，覆盖“音频切分 → 转录/字幕校对 → 生成训练清单”的数据预处理阶段。

## 工具概览

| 文件 | 用途 |
| --- | --- |
| `audio_split_transcribe_v2.py` | 按静音区递归切分音频，可预览、导出 SRT，并调用本地 Whisper 转录 |
| `split_audio_by_srt.py` | 根据 `.srt` 或 `.ass` 字幕时间轴切分音频，生成 WAV 和 `metadata.csv` |
| `transcribed_diff_original_newline.py` | 根据转录稿与原稿的差异，为原稿补齐断行，可选插入 `&` 标记 |
| `pyrenamer.py` | 批量将文件名中的空格、引号等字符替换为下划线 |
| `voxcpm_ljspeech_to_jsonl.py` | 将 LJSpeech 目录转换为 JSONL，并按目录抽取参考音频 |

## 环境准备

建议使用 Python 3.10+ 和虚拟环境。按需要安装依赖：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install pydub pysrt tqdm diff-match-patch
```

还需要系统级 `ffmpeg`，并确保 `ffmpeg -version` 可执行。Whisper 转录按模型格式安装对应依赖：

```bash
pip install openai-whisper      # .pt 权重或官方模型名
pip install faster-whisper     # CTranslate2 模型目录
pip install transformers torch  # HuggingFace 模型目录
```

Linux 使用 cuDNN 且动态库未被找到时，可设置：

```bash
export LD_LIBRARY_PATH=$(realpath ./.venv/lib/python3.10/site-packages/nvidia/cudnn/lib)
```

## 推荐制作流程

### 1. 整理原始音频

```bash
python pyrenamer.py -d ./raw_audio -e .wav .mp3 --recursive --dry-run
python pyrenamer.py -d ./raw_audio -e .wav .mp3 --recursive
```

### 2. 按静音切分并检查

先预览切分点：

```bash
python audio_split_transcribe_v2.py -i ./raw_audio/*.wav --preview
```

确认后正式切分并导出 SRT（默认片段 3–15 秒，首尾保留 300 ms 静音）：

```bash
python audio_split_transcribe_v2.py -i ./raw_audio/*.wav -o ./segments --keep-silence 300 --export-srt
```

自动转录示例：

```bash
python audio_split_transcribe_v2.py -i ./raw_audio/*.wav -o ./segments --export-srt --transcribe --whisper-model base --whisper-language zh --whisper-device cuda
```

常用调参：`--min-silence-len`、`--silence-thresh`、`--min-segment-len`、`--max-segment-len`。

### 3. 已有字幕时按时间轴切分

字幕与音频需同名，例如 `book.srt` 对应 `book.wav`：

```bash
python split_audio_by_srt.py -o ./ljspeech_dataset --silence-action pad --silence-duration 480 book.srt
```

支持 `.srt` / `.ass`，输出目录会生成 `wavs/` 和 `metadata.csv`。`--silence-action` 可选 `none`、`pad`、`remove`。

### 4. 校对转录稿断行

```bash
python transcribed_diff_original_newline.py -t ./transcribed.txt -o ./original.txt --mark
```

`--mark` 会在断行处加入 `&` 标记；不加则只调整换行。

### 5. 生成 VoxCPM JSONL

输入目录需包含 `metadata.csv` 和 `wavs/`：

```bash
python voxcpm_ljspeech_to_jsonl.py -i ./dataset -o ./dataset.jsonl --ref-audio-ratio 0.40 --seed 42
```

默认使用相对路径并打乱输出顺序。可用 `--absolute-path` 写入绝对路径，`--no-shuffle` 保持顺序，`--ref-audio-ratio 0` 关闭参考音频。

## 验证建议

1. 先用 `--preview` 检查片段数量和时长。
2. 抽查 WAV 与 SRT/`metadata.csv` 的文字、时间和音频是否对应。
3. 删除空白过长、爆音、截断或多人说话的片段。
4. 确认文本为 UTF-8，文件名不含空格和引号，JSONL 中的音频路径真实存在。
5. 使用固定 `--seed` 保证参考音频划分可复现。

## 查看帮助

```bash
python audio_split_transcribe_v2.py --help
python split_audio_by_srt.py --help
python transcribed_diff_original_newline.py --help
python pyrenamer.py --help
python voxcpm_ljspeech_to_jsonl.py --help
```
