# 视频自动剪辑 Agent

面向视频剪辑师的 AI 辅助粗剪工具。上传视频，自动完成转录、关键片段提取、字幕导出、说话人分离，审核后一键下载剪辑片段。

---

## 功能概览

| 功能 | 说明 | 费用 |
|------|------|------|
| 🔍 **关键词提取** | 输入关键词，自动找出所有命中片段，可设置前后留白 | 免费 |
| 🎙 **按说话人剪辑** | 识别多位说话人，按人筛选提取发言片段 | 免费 |
| 📄 **导出字幕（SRT）** | 生成标准 SRT 字幕文件，可导入剪辑软件 | 免费 |
| ✂️ **去除静音** | 检测并切除静音段，保留有效说话内容 | 免费 |
| 💬 **自定义需求** | 用自然语言描述需求，AI 理解并执行 | 少量费用 |

所有本地处理（转录、剪辑、导出）完全离线，素材不离开本机。

---

## 平台选择

| | Mac（Apple Silicon）| Windows |
|--|--|--|
| 推荐方式 | **本地直接运行**（速度最快）| Docker |
| 转录引擎 | mlx-whisper（神经引擎）| WhisperX（CPU）|
| medium 模型处理 1 小时视频 | ~1–2 分钟 | ~20–30 分钟 |
| 需要安装 Docker | 否 | 是 |

---

## Mac 安装指南（Apple Silicon M 系列芯片）

### 系统要求

- macOS 12 或更高版本
- Apple Silicon（M1 / M2 / M3 / M4）
- 内存 8 GB 以上
- 已安装 [Homebrew](https://brew.sh)

### 第一步：安装 pyenv 并配置 Python 3.11

**安装 pyenv：**
```bash
brew install pyenv
```

**配置 shell（写入 ~/.zshrc）：**
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'alias python=python3' >> ~/.zshrc
source ~/.zshrc
```

**安装 Python 3.11 并设为默认：**
```bash
pyenv install 3.11
pyenv global 3.11
python3 --version   # 应显示 Python 3.11.x
```

### 第二步：下载项目

```bash
git clone <仓库地址>
cd auto-vedio-edit-agent
```

### 第三步：填写 API Key

```bash
cp api_keys.env.example api_keys.env
```

用文本编辑器打开 `api_keys.env`，填入以下内容（详见 [API Key 配置](#api-key-配置)）：

```dotenv
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx   # 必填
HF_TOKEN=hf_xxxxxxxxxxxxxxxx                # 可选，说话人分离功能需要
```

### 第四步：安装依赖

```bash
./setup_mac.sh
```

脚本自动完成：检查 ffmpeg → 创建虚拟环境 → 安装 mlx-whisper 等依赖（约 3–5 分钟）。

### 第五步：启动应用

```bash
./run_mac.sh
```

看到以下输出即启动成功：

```
Apple Silicon — mlx-whisper, model: medium (Neural Engine)
Running on local URL: http://0.0.0.0:7860
```

浏览器打开 **http://localhost:7860**。

### 后续启动

每次只需运行：
```bash
./run_mac.sh
```

### 更换模型（可选）

在 `api_keys.env` 中添加：

```dotenv
WHISPER_MODEL=large-v3   # 可选：tiny / small / medium / large-v3
```

| 模型 | M 芯片处理 1 小时视频 | 准确率 |
|------|----------------------|--------|
| small | ~30 秒 | 一般 |
| medium | ~1–2 分钟（默认）| 好 |
| large-v3 | ~3–5 分钟 | 最佳 |

---

## Windows 安装指南

### 系统要求

- Windows 10 / 11（64 位）
- 内存 8 GB 以上（建议 16 GB）
- 磁盘空间 10 GB 以上
- 已安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### 第一步：安装并启动 Docker Desktop

前往 [docker.com](https://www.docker.com/products/docker-desktop/) 下载 Windows 版，安装后重启电脑，确认任务栏出现鲸鱼图标且状态为 **Running**。

### 第二步：下载项目

下载 ZIP 压缩包解压，或使用 Git：

```bat
git clone <仓库地址>
cd auto-vedio-edit-agent
```

> 路径中不要包含中文或空格。

### 第三步：填写 API Key

复制模板文件：
```bat
copy api_keys.env.example api_keys.env
```

用文本编辑器打开 `api_keys.env`，填入（详见 [API Key 配置](#api-key-配置)）：

```dotenv
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx   # 必填
HF_TOKEN=hf_xxxxxxxxxxxxxxxx                # 可选
```

### 第四步：构建并启动

双击 `setup.bat`，或在命令提示符中运行：

```bat
setup.bat
```

首次构建约需 5–10 分钟（下载镜像和模型）。看到以下提示即成功：

```
=== 启动完成 ===
打开浏览器访问：http://localhost:7860
```

### 后续启动与停止

```bat
start.bat    # 启动
stop.bat     # 停止
```

### 更换模型（可选）

编辑 `docker.env`：

```dotenv
WHISPER_MODEL=small   # tiny / small / medium（默认）/ large-v3
```

保存后运行 `start.bat` 重启生效。

| 模型 | CPU 处理 1 小时视频 | 准确率 |
|------|---------------------|--------|
| small | ~8–10 分钟 | 一般 |
| medium | ~20–30 分钟（默认）| 好 |
| large-v3 | ~60–90 分钟 | 最佳 |

---

## API Key 配置

所有密钥统一存放在项目根目录的 `api_keys.env`（不会上传到 git）。

### Anthropic API Key（必填）

用于「自定义需求」AI 解析功能。

1. 访问 [console.anthropic.com](https://console.anthropic.com) 注册登录
2. 进入 **API Keys** → **Create Key**，复制 Key
3. 填入 `api_keys.env`：

```dotenv
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

### HuggingFace Token（可选，说话人分离需要）

1. 注册 [huggingface.co](https://huggingface.co)（免费）
2. 进入 [Settings → Tokens](https://huggingface.co/settings/tokens)，创建 **Read** 权限 Token
3. 访问以下页面接受模型授权（点击 **Agree and access repository**）：
   - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
4. 填入 `api_keys.env`：

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

不填 `HF_TOKEN` 时，说话人分离功能不可用，其余功能完全正常。

---

## 使用指南

启动后打开浏览器访问 **http://localhost:7860**。

### 第一步：上传视频

拖拽或点击左侧上传区域，上传视频文件。

支持格式：`mp4`、`mov`、`avi`、`mkv`、`webm` 等。

---

### 第二步：选择功能

提供两种操作方式，可单独或组合使用。

#### 方式 A — 功能按钮（免费）

点击按钮高亮选中，可多选：

**🔍 关键词提取**
- 选中后展开关键词输入框，输入关键词（逗号分隔）
- 可调整片段前后留白时间（默认各 3 秒）
- 示例：`竞品, 价格, 优惠`

**🎙 按说话人剪辑**
- 需配置 HuggingFace Token
- 点击「开始分析」后转录并识别说话人，在界面中选择目标说话人，再点「确认筛选」

**📄 导出字幕（SRT）**
- 转录完成后自动生成 SRT 字幕文件，可直接下载
- 兼容 Premiere Pro、Final Cut、DaVinci Resolve 等主流剪辑软件

**✂️ 去除静音**
- 自动检测静音段，将连续有声内容作为独立片段返回

**多选说明：** 同时选中多个按钮时规则叠加执行，转录只进行一次，不增加等待时间。

---

#### 方式 B — 自定义需求（AI 解析，少量费用）

在文字输入框中用自然语言描述需求：

```
找出所有情绪激动的片段
提取最精彩的三个观点
截取前五分钟内容
去掉废话和停顿，保留干货部分
```

> 费用参考：10 分钟视频约 ¥0.03，1 小时视频约 ¥0.14（Claude Haiku 定价）

---

### 第三步：开始分析

点击「🚀 开始分析」，界面显示三步进度：

| 步骤 | 内容 | Mac M 芯片 | Windows（Docker）|
|------|------|-----------|-----------------|
| ① 音频转录 | mlx-whisper / WhisperX 转录 | 10 分钟视频 ~10–30 秒 | 10 分钟视频 ~1–3 分钟 |
| ② 解析意图 | Claude AI 理解需求（仅自定义路径）| ~5–10 秒 | ~5–10 秒 |
| ③ 执行规则 | 匹配候选片段 | ~1 秒 | ~1 秒 |

**使用按钮时跳过②，无 AI 费用。**

---

### 第四步：审核候选片段

处理完成后，结果以表格呈现：

| 序号 | 说话人 | 时间范围 | 内容预览 | 置信度 | 包含 |
|------|--------|---------|---------|------|-----|
| 1 | SPEAKER_00 | 12.3s–28.5s | 竞品的价格策略非常激进… | 0.95 | ☑ |
| 2 | SPEAKER_01 | 45.0s–62.1s | 续航方面我们实测是… | 0.88 | ☑ |

- 取消勾选不需要的片段
- 配置 HF_TOKEN 后显示说话人标签；未配置时显示「—」

---

### 第五步：下载

- **字幕文件**：如选择了「导出字幕」，处理完成后自动出现 SRT 下载链接
- **视频片段**：勾选要保留的片段，点击「⬇️ 下载选中片段」，批量下载 mp4 文件

导出文件同时保存在项目 `output/` 目录。

---

## 常见问题

**Q：Mac 上运行比 Windows 快很多？**

是的。Mac M 芯片使用神经引擎（Neural Engine），mlx-whisper 可直接调用，处理速度是 CPU 的 10–15 倍。Windows 的 Docker 方案只能使用 CPU。

---

**Q：大文件（超过 30 分钟）会自动分块处理吗？**

会。系统自动将超过 30 分钟的视频按 10 分钟一块分段转录，完成后时间戳自动合并，无需手动处理。

---

**Q：处理过程中页面刷新了怎么办？**

Mac 本地运行：处理在后台进行，刷新后需重新上传视频。建议处理期间不要关闭终端。

Windows Docker：处理期间保持页面开启，每 5 秒有状态更新保持连接。

---

**Q：字幕文件用什么剪辑软件打开？**

导出的 `.srt` 文件是标准格式：
- **Premiere Pro**：序列 → 字幕 → 导入字幕文件
- **Final Cut Pro**：文件 → 导入 → 字幕
- **DaVinci Resolve**：时间线 → 字幕 → 导入 SRT

---

**Q：关键词没有匹配到结果？**

1. 检查关键词是否与转录文字一致（转录可能与实际发音有出入）
2. 尝试换用「自定义需求」输入框，AI 可理解模糊描述

---

**Q：说话人识别返回为空？**

- 确认 `api_keys.env` 中已填写 `HF_TOKEN`
- 确认已在 HuggingFace 接受模型授权（见 [API Key 配置](#api-key-配置)）
- 确认视频有清晰的多人对话音频

---

**Q：如何彻底卸载（Mac）？**

```bash
rm -rf .venv              # 删除依赖环境
brew uninstall pyenv      # 卸载 pyenv（可选）
```

---

## 停止与重启

**Mac：**
```bash
# 停止：在终端按 Ctrl+C
# 重新启动：
./run_mac.sh
```

**Windows：**
```bat
stop.bat     # 停止
start.bat    # 启动
```

---

## 项目总结

本项目是一套**完全本地运行、全程免费模型**的视频粗剪辅助工具，在这一前提下已接近最优解：

- **转录**：Apple Silicon 使用 mlx-whisper + Neural Engine，large-v3 模型在 M 系列芯片上实时倍率可达 10x 以上；双语并行转录（中文 + 英文）解决混合语言视频的字幕随机性问题
- **AI 解析**：自定义需求通过 Claude Haiku 完成，成本极低（1 小时视频约 ¥0.14）
- **视频处理**：ffmpeg 并行剪切，I/O 密集型任务无 CPU 瓶颈
- **隐私**：所有音视频素材留在本机，不经过任何外部服务

**当前瓶颈与提升路径：**

| 瓶颈 | 当前方案 | 提升方向 |
|------|---------|---------|
| 转录速度（Windows/Linux）| WhisperX CPU，large-v3 约 60–90 分/小时视频 | 租用 GPU 服务器（A10/A100），速度提升 20–50x |
| 转录准确率 | Whisper large-v3，中英文混说已双语并行 | 付费 API（如 AssemblyAI、Deepgram）准确率更高且支持实时流 |
| 意图理解复杂度 | Claude Haiku，轻量快速 | 换用 Claude Sonnet/Opus 处理更复杂的剪辑指令 |
| 说话人分离质量 | pyannote community 免费版 | pyannote 商业版或 AssemblyAI 说话人分离 |

本地免费方案已覆盖绝大多数粗剪场景。如需专业级准确率或处理大批量视频，可考虑接入付费 API 或租用 GPU 服务器。

---

## 项目结构

```
auto-vedio-edit-agent/
│
├── api_keys.env             ← API 密钥（本地填写，不上传 git）
├── api_keys.env.example     ← 密钥模板
├── docker.env               ← Docker 配置（Windows 用）
│
├── setup_mac.sh             ← Mac 首次安装
├── run_mac.sh               ← Mac 启动脚本
├── requirements-mac.txt     ← Mac 依赖（mlx-whisper）
│
├── setup.sh / setup.bat     ← Linux / Windows 首次安装
├── start.sh / start.bat     ← Linux / Windows 启动
├── stop.sh / stop.bat       ← Linux / Windows 停止
├── requirements.txt         ← Docker 依赖（WhisperX）
│
├── app/
│   ├── main.py              ← Gradio UI 入口
│   └── pipeline.py          ← 按钮路径辅助函数
├── agent/
│   ├── intent_parser.py     ← AI 解析自定义需求
│   ├── orchestrator.py      ← 流水线编排
│   └── rule_engine.py       ← 规则执行引擎
├── processing/
│   ├── transcriber.py       ← 转录（mlx-whisper / WhisperX 自动切换）
│   ├── subtitle.py          ← SRT 字幕生成
│   └── ffmpeg_utils.py      ← 视频剪辑工具
├── models/                  ← Pydantic 数据模型
├── config/
│   └── settings.py          ← 配置管理
│
└── output/                  ← 导出文件保存位置
```
