## 免责声明

本项目基于B站开源项目进行二次开发，由本人对项目进行了ComfyUI的实现，并进行了部分功能优化与调整与进阶功能的开发。然而，需要强调的是，本项目严禁用于任何非法目的以及与侵犯版权相关的任何行为！本项目仅用于开源社区内的交流与学习，以促进技术共享与创新，旨在为开发者提供有益的参考和学习资源。

在此郑重声明，本项目所有个人使用行为与开发者本人及本项目本身均无任何关联。开发者对于项目使用者的行为不承担任何责任，使用者应自行承担使用过程中可能产生的所有风险和法律责任。请广大使用者在遵守法律法规及相关规定的前提下，合理、合法地使用本项目，维护开源社区的良好秩序与健康发展。

感谢您的理解与支持！


# ComfyUI-Index-TTS（IndexTTS-2.5 专用版）

在 ComfyUI 中使用 [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) 进行零样本音色克隆语音合成的自定义节点。

> 本分支已精简为 **仅支持 IndexTTS-2.5**。旧版 Index-TTS / IndexTTS-1.5 / IndexTTS-2 节点、小说多角色解析、Audio Cleaner、Timbre 音色库均已移除。

## 功能

- **多语言合成**：中文 / 英语 / 日语 / 西班牙语 / 阿拉伯语，支持**跨语种克隆**（用中文参考音频直接说英文/日文）
- **语速控制**：`duration_factor` 0.5 – 2.0，大于 1 变慢、小于 1 变快
- **发音标注**：在文本里直接标注多音字/发音
  - 中文拼音：`他在银<行|XING2>里<行|HANG2>走了半天`
  - 英文音素（CMU）：`He had a <minute|M IH1 . N AH0 T> to examine the <minute|M AY0 . N UW1 T> details.`
  - 日语假名：`彼は料理が<上手|じょうず>だ`
- **情感控制**：情感参考音频 / 8 维情感向量 / Qwen 情感文本分析
- **音色持久化**：一次提取参考音频的 conditioning 保存为 `.pt`，之后直接复用
- **bf16 推理**（CUDA），显存/缓存可控

## 节点

| 节点 | 说明 |
|---|---|
| `Index TTS 2.5 - Base (多语言/语速)` | 基础合成：参考音频 + 文本 + 语言 + 语速 |
| `Index TTS 2.5 - Emotion Audio` | 情感参考音频（音色与情感解耦），`emotion_weight` 调强度 |
| `Index TTS 2.5 - Emotion Vector` | 8 维情感向量滑条：高兴/愤怒/悲伤/恐惧/反感/低落/惊讶/平静 |
| `Index TTS 2.5 - Emotion Text` | 情感描述控制，留空则自动分析主文本情感（需要 Qwen 情感模型） |
| `Index TTS 2.5 - Cache Control` | 生成后是否保留模型在显存中 |
| `Index TTS 2.5 - Extract Voice` | 从参考音频提取音色 conditioning |
| `Index TTS 2.5 - Save Voice` | 将音色 conditioning 保存到 `output/`（`.pt`） |
| `Index TTS 2.5 - Load Voice` | 从 `input/` / `output/` 选择或上传 `.pt` 音色 |
| `Index TTS 2.5 - Voice Base` | 使用已保存的音色 conditioning 直接合成 |

示例工作流：`./workflow/TTS2.5.json`（直接拖进 ComfyUI 页面即可导入）。

## 安装

1. 将此代码库克隆到 ComfyUI 的 `custom_nodes` 目录：

   ```bash
   cd ComfyUI/custom_nodes
   git clone <本仓库地址> ComfyUI-Index-TTS
   ```

2. 安装依赖：

   ```bash
   cd ComfyUI-Index-TTS
   python -m pip install -r requirements.txt
   ```

## 模型下载与放置

全部放到 `ComfyUI/models/IndexTTS-2.5/`：

1) 基础模型（整仓）
   - 页面：[IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5/tree/main)
   - 国内镜像：[hf-mirror 镜像页](https://hf-mirror.com/IndexTeam/IndexTTS-2.5/tree/main)
   - 包含：`gpt.pth`（主模型约 3.3GB）、`codec.pth`（统一语义编解码器）、`s2mel.pth`、`config.yaml`、`feat1.pt`、`feat2.pt`、`wav2vec2bert_stats.pt`、`multilingual_zh_ja_yue_char_del.tiktoken`（多语言分词器）

2) qwen 模型（情绪分类，仅 Emotion Text 节点需要）
   - 页面：[IndexTTS-2.5/qwen0.6bemo4-merge](https://huggingface.co/IndexTeam/IndexTTS-2.5/tree/main/qwen0.6bemo4-merge)
   - 放置：`ComfyUI/models/IndexTTS-2.5/qwen0.6bemo4-merge/`

3) CampPlus 说话人嵌入
   - 直链：[campplus_cn_common.bin](https://huggingface.co/funasr/campplus/resolve/main/campplus_cn_common.bin)
   - 放置：`ComfyUI/models/IndexTTS-2.5/campplus_cn_common.bin`

4) Wav2Vec2Bert 特征提取器（facebook/w2v-bert-2.0）
   - 页面：[facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0/tree/main)
   - 放置（离线优先）：`ComfyUI/models/IndexTTS-2.5/w2v-bert-2.0/`（整仓文件夹）

5) BigVGAN 声码器
   - 页面：[nvidia/bigvgan_v2_22khz_80band_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x/tree/main)
   - 放置：`ComfyUI/models/IndexTTS-2.5/bigvgan/`（`config.json` + `bigvgan_generator.pt`）

目录结构（详见仓库内 `TTS2.5模型路径.txt`）：

```text
ComfyUI/models/IndexTTS-2.5/
│  campplus_cn_common.bin
│  codec.pth
│  config.yaml
│  feat1.pt
│  feat2.pt
│  gpt.pth
│  multilingual_zh_ja_yue_char_del.tiktoken
│  s2mel.pth
│  wav2vec2bert_stats.pt
│
├─bigvgan
│      bigvgan_generator.pt
│      config.json
│
├─qwen0.6bemo4-merge
│      config.json
│      model.safetensors
│      tokenizer.json
│      ...
│
└─w2v-bert-2.0
        config.json
        conformer_shaw.pt
        model.safetensors
        preprocessor_config.json
```

### 一键下载（推荐）

支持断点续传、国内镜像：

```powershell
python .\ComfyUI\custom_nodes\ComfyUI-Index-TTS\TTS2_5_download.py
```

- 运行后根据提示选择 2 使用国内镜像（默认）或 1 使用官方源。
- 若镜像设置不生效，可先在控制台设置：`$env:HF_ENDPOINT = "https://hf-mirror.com"`（PowerShell）/ `export HF_ENDPOINT=https://hf-mirror.com`（Linux）
- 依赖：`huggingface_hub`；可选加速：`hf_transfer`、`hf_xet`。

## 使用方法

1. **加载参考音频**：用 `LoadAudio` 节点选择一段 3–15 秒的干净人声
2. **接到 `Index TTS 2.5 - Base`**：填文本、选语言（ZH/EN/JA/ES/AR 或 ZH/EN 混合）、按需调速（`duration_factor`）
3. （可选）接一个 `Index TTS 2.5 - Cache Control` 到 `cache_control` 输入
4. **输出接 `SaveAudio` 或 `PreviewAudio`** 即可

情感控制三选一：

- 有情感参考音频 → `Emotion Audio` 节点，`emotion_weight` 调强度（0–1）
- 想精细配比 → `Emotion Vector` 节点，8 个滑条
- 只想写一句话 → `Emotion Text` 节点，填情感描述（留空则自动分析主文本），`emotion_weight` 建议 0.6 左右

> 提示：Emotion Text 节点首次使用会额外加载 Qwen 情感小模型（约 1.2GB 显存）；其余节点不需要。

### 音色持久化

`LoadAudio → Extract Voice → Save Voice` 保存一次音色（`.pt`，默认在 `output/IndexTTS2.5/`），之后用 `Load Voice → Voice Base` 直接合成，无需再处理参考音频。`Load Voice` 节点上有「上传 .pt」按钮，可从本地上传音色文件。

### 显存/缓存控制

`Index TTS 2.5 - Cache Control` 输出 `{"keep_cached": true/false}`，连到各合成节点的 `cache_control` 输入：

- 关闭（默认）：推理结束后卸载模型并清理 CUDA 缓存，适合 12GB 显卡日常使用。
- 开启：保留已加载的权重，连续多次生成更快，但显存占用更高。

### 主要参数

- **temperature** (默认 0.8)、**top_p** (0.8)、**top_k** (30)：采样随机性
- **num_beams** (3)：束搜索宽度
- **repetition_penalty** (10.0)、**length_penalty** (0.0)
- **max_mel_tokens** (1500)：单段最大音频 token 数
- **max_tokens_per_sentence** (120)：长文本分句的最大 token 数
- **interval_silence_ms** (200)：句间静音
- **text_normalization**：是否做文本归一化（数字等）

## 常见问题

**w2v-bert-2.0 加载失败 / 401 Unauthorized**：从 [HuggingFace](https://huggingface.co/facebook/w2v-bert-2.0/tree/main) 下载全部文件放到 `ComfyUI/models/IndexTTS-2.5/w2v-bert-2.0/`，重启 ComfyUI 即可离线加载。

**`'SafeTensorFile' object has no attribute 'get_slice'`**：升级 safetensors：`pip install safetensors --upgrade`（>= 0.4.3）。

**模型加载失败**：检查 `ComfyUI/models/IndexTTS-2.5/` 下文件是否完整，缺失文件会在报错中列出。

**pynini 安装失败**：pynini 不支持 Python 3.13，Linux 下建议 Python 3.10 / 3.11；Windows 不需要 pynini。

## 鸣谢

- 基于原始 [IndexTTS](https://github.com/index-tts/index-tts) 模型
- 上游 ComfyUI 插件：[chenpipi0807/ComfyUI-Index-TTS](https://github.com/chenpipi0807/ComfyUI-Index-TTS)

## 许可证

请参考原始 IndexTTS 项目许可证。
