# TTS Labeler

面向多语言、单说话人长音频的 TTS 数据集自动制作工具。

系统先依据音频本身的语音活动、停顿和能量特征确定切点，再逐片转录为 SRT，并将原始文档与 SRT 做全局顺序对齐。最终 SRT 可以直接用于数据集导出，也可以在 Subtitle Edit、Aegisub 等字幕软件中核对后重新导出。

> 当前状态：具备工业化基础，但尚未经过具体业务语料的工业验收。上线前应使用目标录音环境建立边界基准并标定阈值。

## 核心原则

- **声学先于 ASR**：Whisper 不参与初始切点决策。
- **不依赖词级时间戳**：Whisper 只转录已经切好的声学片段。
- **文档是真值**：最终训练文本来自原始文档，ASR 只用于定位和质量判断。
- **SRT 是中间标准**：时间轴和文本可以通过通用字幕软件检查和修改。
- **不确定数据自动隔离**：低匹配、异常时长、削波或其他质量问题不会进入正式训练集。
- **可恢复、可审计**：支持逐段缓存、断点续跑、运行指纹和完整报告。

## 处理流程

```text
原始长音频
  ↓
规范化 PCM master
  ↓
Silero VAD + 自适应短时能量分析
  ↓
全局最优声学切分
  ↓
每个片段独立 Whisper 转录
  ↓
raw.srt
  ↓
原始文档与字幕全局顺序对齐
  ↓
aligned.srt
  ↓
可选：字幕软件核对、修改文本或时间轴
  ↓
按最终 SRT 导出 WAV 数据集
```

## 声学切分算法

### 1. 统一时间轴

输入音频首先通过 FFmpeg 转换为规范化 PCM master。后续分析、声学片段、SRT 时间轴和最终 WAV 全部基于该 master，避免 MP3、AAC、视频容器或编码延迟造成切点漂移。

默认输出格式：

- 单声道；
- 24 kHz；
- 16-bit PCM WAV。

### 2. VAD 与能量联合判断

默认使用 Silero VAD 检测语音区间。VAD 输出的非语音间隙不会直接成为切点，还必须通过局部低能量检查。

```text
VAD 判定为非语音
        +
局部 RMS 确认为低能量
        ↓
可靠停顿候选
```

这种联合判断可降低以下误切风险：

- 背景音乐；
- 稳定但较响的环境噪声；
- VAD 短暂漏检；
- 轻声和呼吸声。

### 3. 自适应局部阈值

系统每 20ms 计算一次短时 RMS/dBFS，并根据录音的噪声底、语音参考能量和动态范围生成自适应阈值。

对于 VAD 间隙，还会使用附近约 5 秒的局部能量分布重新估计阈值，以适应：

- 录音中途增益变化；
- 空调、风扇等底噪变化；
- 不同章节录音条件不同。

### 4. 精确切点

可靠停顿内部先选择最低能量区域，再在附近约 ±5ms 搜索零交叉点，降低切割爆音。

短停顿通常在一个位置切开；长停顿只保留两端有限上下文，中间无信息区域会被删除。

### 5. 全局切点规划

系统使用动态规划综合考虑：

- 停顿长度和质量；
- 目标片段时长；
- 最短片段时长；
- 最长片段时长；
- 是否必须使用低能量兜底切点。

最短和最长时长是硬约束。如果最长范围内没有可靠停顿，系统才会在允许范围内寻找最低能量位置兜底。

### 6. 首尾处理

系统会根据 VAD 结果删除多余的前导和尾部静音，同时保留可配置的安全边缘，避免截断吸气、尾音和爆破音释放。

## 文档与 SRT 对齐

对齐过程不会修改声学时间轴，只把每个 SRT 条目的 ASR 文本替换为对应的原文片段。

系统支持 Unicode 多语言文本，包括：

- 中文、日文、韩文；
- 英文及其他拉丁文字语言；
- 斯拉夫文字；
- 阿拉伯文字；
- 印度文字及其他 Unicode 脚本。

对齐使用唯一 n-gram 锚点约束全文，再在锚点窗口内执行局部顺序匹配，以降低重复句、漏读和局部识别错误导致的全文漂移。

## 安装

### 环境要求

- Python 3.10 或更高版本；
- FFmpeg；
- 推荐 NVIDIA GPU；
- CPU 也可以运行，但大型 Whisper 模型会较慢。

确认 FFmpeg：

```powershell
ffmpeg -version
```

### 默认工业模式

安装 faster-whisper、Silero VAD 和 ONNX Runtime：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[industrial]"
```

### Hugging Face 微调模型

如需直接加载普通 Transformers Whisper 检查点：

```powershell
pip install -e ".[transformers,vad]"
```

### 开发和测试

```powershell
pip install -e ".[industrial,dev]"
```

## 快速开始

```powershell
tts-labeler run input.wav output --document document.txt `
  --backend faster-whisper `
  --model large-v3 `
  --language auto `
  --initial-prompt "以下内容为普通话朗读，使用规范中文标点。" `
  --device cuda `
  --compute-type float16
```

需要详细日志时，`--verbose` 放在子命令之前：

```powershell
tts-labeler --verbose run input.wav output --document document.txt
```

`--initial-prompt` 用于向 Whisper 提供固定的术语和书写风格上下文，支持 faster-whisper 和 Transformers 微调模型。默认不传入任何提示词。为了保证批量结果一致，应为同一批任务使用完全相同的提示词；提示词会写入配置报告和任务指纹。

```powershell
tts-labeler run input.wav output `
  --language zh `
  --initial-prompt "以下内容为普通话朗读，使用规范中文标点。"
```

### 过滤其他说话人

使用当前开源 Community-1 管线进行说话人分段。首次使用前，需要在 Hugging Face 接受模型访问条件，并把令牌放入 `HF_TOKEN` 环境变量；令牌本身不会写入任务报告。

```powershell
$env:HF_TOKEN="hf_..."
tts-labeler run input.wav output --document document.txt `
  --speaker-backend pyannote `
  --speaker-reference target-speaker.wav
```

不提供 `--speaker-reference` 时，系统将语音总时长最大的说话人簇视为目标说话人：

```powershell
tts-labeler run input.wav output --document document.txt --speaker-backend pyannote
```

说话人分析只用于质量门控，不改变声学切点。其他说话人超过 0.25 秒或占有效语音超过 5% 时，片段进入 `rejected/` 并标记为 `mixed_speaker`。人工修改 SRT 后再次执行 `export` 时，也应传入相同的说话人参数。

没有对照文档时只传入音频和输出目录，系统会自动跳过文档对齐并直接使用 ASR 文本：

```powershell
tts-labeler run input.wav output
```

该模式仍执行声学切分、ASR、音频质量检查和数据集导出，生成 `raw.srt`，但不会生成 `aligned.srt`。建议先在字幕软件中核对 `raw.srt`，再使用 `export` 导出最终数据集。

## 常用切分参数

```powershell
tts-labeler run input.wav output --document document.txt `
  --vad silero `
  --vad-threshold 0.5 `
  --vad-min-speech 0.10 `
  --min-silence 0.32 `
  --silence-margin-db 10 `
  --silence-dbfs -38 `
  --boundary-padding 0.08 `
  --max-silence-kept 0.5 `
  --min-duration 1.5 `
  --target-duration 8 `
  --max-duration 18 `
  --sample-rate 24000
```

参数含义：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--vad` | `silero` | VAD 后端；`off` 为明确的 RMS 降级模式 |
| `--vad-threshold` | `0.5` | Silero 语音概率阈值 |
| `--vad-min-speech` | `0.10` | 最短语音持续时间，单位秒 |
| `--min-silence` | `0.32` | 可用于切分的最短停顿，单位秒 |
| `--boundary-padding` | `0.08` | 切点两侧安全边缘，单位秒 |
| `--max-silence-kept` | `0.5` | 长静音两侧最多参与保留的范围 |
| `--min-duration` | `1.5` | 最短片段时长 |
| `--target-duration` | `8` | 优化器倾向的目标时长 |
| `--max-duration` | `18` | 最长片段时长 |

工业模式默认要求 Silero VAD 可用。如果显式使用：

```powershell
--vad off
```

系统会降级为纯自适应 RMS 切分，并在 `report.json` 中记录 `candidate_source: rms-only`。系统不会静默降级。

## 使用自定义 Whisper

### CTranslate2 / faster-whisper 模型

`--model` 可以是模型名称、兼容的 Hugging Face ID 或本地 CTranslate2 目录：

```powershell
tts-labeler run input.wav output --document document.txt `
  --backend faster-whisper `
  --model D:\models\my-whisper-ct2 `
  --language auto `
  --device cuda `
  --compute-type float16
```

### Transformers 微调检查点

```powershell
tts-labeler run input.wav output --document document.txt `
  --backend transformers `
  --model organization/my-finetuned-whisper `
  --language ar `
  --device cuda `
  --compute-type float16
```

也可以传入本地完整模型目录：

```powershell
--model D:\models\my-whisper-checkpoint
```

Transformers 检查点应包含完整模型、processor/tokenizer 和生成配置。LoRA/PEFT adapter 需要先与基础 Whisper 模型合并。

已知语言时，显式指定 `zh`、`en`、`de`、`ar` 等通常比短片段自动检测更稳定。真正的单文件多语言混合应使用多语言 Whisper 模型。

## SRT 核对和修改

自动运行生成两个字幕文件：

- `raw.srt`：逐声学片段的原始 ASR 结果；
- `aligned.srt`：时间轴不变，文本替换为文档对齐结果。

可以使用 Subtitle Edit、Aegisub 或其他标准字幕软件打开 `aligned.srt`，修改文本或时间轴。

修改完成后，不需要重新运行 VAD、声学分析或 Whisper：

```powershell
tts-labeler export input.wav edited.srt final-dataset
```

`export` 会重新生成规范化 PCM master，并检查：

- 时间是否递增；
- 是否超出音频时长；
- 相邻字幕是否存在异常重叠；
- 文本是否为空；
- 片段时长是否符合要求。

## 输出结构

```text
output/
├── raw.srt
├── aligned.srt
├── wavs/                       自动验收通过的 WAV
├── rejected/                   自动隔离的 WAV
├── metadata.csv                audio|text
├── manifest.jsonl              每段完整信息和质量指标
├── report.json                 本次任务配置和汇总
└── work/
    ├── master.wav              唯一规范化主时间轴
    ├── analysis.wav            16 kHz 声学分析文件
    ├── acoustic.json           声学切分缓存
    ├── state.json              运行指纹和状态
    ├── acoustic_segments/      Whisper 输入片段
    └── asr/                    逐片 ASR 缓存
```

`metadata.csv`：

```text
wavs/000000.wav|对应的标准文档文本
wavs/000001.wav|下一段标准文档文本
```

`manifest.jsonl` 每段包含：

- 标准文本和 ASR 文本；
- 开始、结束和时长；
- 文本匹配分和覆盖率；
- ASR 置信度（后端不提供时为 `null`）；
- WAV 路径；
- 音频质量指标；
- 是否验收；
- 拒收原因。

## 自动质量控制与门禁

默认检查：

- 片段过短或过长；
- 文档与 ASR 匹配度不足；
- 文档字符覆盖率不足；
- 可用时检查 ASR 置信度；
- 空文本；
- 音频过静；
- 削波比例过高；
- 直流偏移过大。

不合格片段进入 `rejected/`，不会写入正式 `metadata.csv`。

自动转录模式还会检查 Whisper 的无语音概率、压缩率、重复 n-gram 和字符速率，用于隔离空音频幻觉、重复短语和异常长文本。对应拒收原因包括 `asr_no_speech`、`asr_compression_hallucination`、`asr_repetition` 和 `asr_text_too_fast`。人工确认后的 SRT 不执行这些 ASR 文本门控。

任务执行异常时，`work/state.json` 会记录 `failed`、异常类型和错误信息；已完成的声学与 ASR 分段缓存仍会保留，使用相同输入、模型和配置重新执行即可继续。

从人工确认后的 SRT 执行 `export` 时，SRT 文本被视为最终真值，不再依据 ASR 匹配分拒收，但仍执行时间轴、时长和音频质量检查。

## 断点续跑和目录安全

系统根据以下内容生成运行指纹：

- 输入音频 SHA-256；
- 文档 SHA-256；
- 完整运行配置；
- 流水线缓存版本。

同一任务中断后，再次运行相同命令会复用：

- PCM master；
- 声学分析结果；
- 已完成片段；
- 已完成 ASR 结果。

如果输入、文档或配置已经变化，系统会拒绝复用原输出目录。请使用新的输出目录，避免不同任务的数据混合。

## 性能说明

- PCM 能量分析按帧流式进行，不会把整段波形常驻内存；
- Silero VAD 默认按有限长度分块处理，并在块间保留重叠；
- 只保留轻量的帧级能量序列；
- 每个 ASR 片段完成后立即缓存；
- 最终文件通过临时文件写完后原子替换。

当前 Whisper 转录按片段顺序执行。GPU 批处理和多任务调度可作为部署层扩展，但不影响数据正确性。

## 测试

```powershell
pytest
ruff check .
```

当前测试覆盖：

- 多语言文本规范化；
- 基础和重复文本对齐；
- VAD 与 RMS 冲突；
- 长静音删除；
- 首尾静音裁剪；
- 严格最大时长；
- SRT 往返、重叠和越界；
- 削波、静音和直流偏移；
- PCM/SRT/数据集端到端流程；
- ASR 缓存和断点续跑；
- 不同任务输出目录冲突。

## 上线前验收

代码通过测试不等于已经完成工业验收。正式部署前建议使用目标业务录音建立人工真值集，并至少测量：

- 语音截断率；
- 错误切分率；
- 边界时间误差 P50/P95；
- 文档错配率；
- 自动验收准确率；
- 数小时音频的内存和耗时；
- 不同语言、设备和噪声条件下的稳定性。

推荐先使用 30–60 分钟代表性录音标定参数，再进行数小时压力测试。
