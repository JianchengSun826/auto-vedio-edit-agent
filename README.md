# 视频自动剪辑 Agent

AI 驱动的视频自动剪辑工具。输入自然语言需求，自动转录视频、提取候选片段，经人工审核后导出到多个社交媒体平台。

---

## 目录

- [功能概览](#功能概览)
- [第一步：环境准备](#第一步环境准备)
- [第二步：启动后台服务](#第二步启动后台服务)
- [第三步：启动 UI](#第三步启动-ui)
- [第四步：实际使用](#第四步实际使用)
- [批量处理（Python API）](#批量处理python-api)
- [三种剪辑模式](#三种剪辑模式)
- [常见问题](#常见问题)
- [项目结构](#项目结构)
- [未来扩展](#未来扩展)

---

## 功能概览

- **自然语言需求输入**：用中文描述剪辑要求，无需学习任何剪辑软件
- **本地语音转录**：使用 faster-whisper 在本地运行，零 STT 费用
- **AI 意图解析**：调用 Claude Haiku 将需求转换为结构化剪辑计划
- **人工审核**：在 Gradio UI 中逐条勾选要保留的片段
- **多平台导出**：抖音（9:16/60s）、B站、YouTube、微信视频号，自动转码和分割

---

## 第一步：环境准备

> 只需做一次。

### 1.1 安装系统依赖

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# 验证安装
ffmpeg -version
```

### 1.2 安装 Python 依赖

```bash
git clone <仓库地址>
cd auto-vedio-edit-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1.3 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填入以下内容：

```dotenv
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx    # 必填，前往 console.anthropic.com 获取
WHISPER_MODEL=medium                  # 可选：tiny / base / small / medium / large-v3
WHISPER_DEVICE=auto                   # 有 NVIDIA GPU 填 cuda，否则填 cpu
```

**Whisper 模型选择参考：**

| 模型 | 内存占用 | 速度 | 准确度 | 推荐场景 |
|------|---------|------|--------|---------|
| tiny | ~1 GB | 极快 | 一般 | 快速测试 |
| small | ~2 GB | 快 | 较好 | 日常使用 |
| medium | ~5 GB | 中等 | 好 | 推荐默认 |
| large-v3 | ~10 GB | 慢 | 最佳 | 精度要求高 |

---

## 第二步：启动后台服务

Celery worker 负责异步处理转录和导出任务，避免长时间操作阻塞 UI。

### 方式 A：Docker（推荐）

```bash
# 启动 Valkey 消息队列 + Celery worker
docker compose up -d

# 查看 worker 是否就绪
docker compose logs -f worker
```

看到以下输出表示启动成功：

```
[tasks.transcribe] .ok
[tasks.export] .ok
ready.
```

### 方式 B：本地手动启动

```bash
# 终端 1：启动 Valkey
docker run -d -p 6379:6379 valkey/valkey:7

# 终端 2：启动 Celery worker
source .venv/bin/activate
celery -A tasks.celery_app worker --loglevel=info
```

---

## 第三步：启动 UI

```bash
source .venv/bin/activate
python app/main.py
```

浏览器打开 **http://127.0.0.1:7860**

---

## 第四步：实际使用

### 4.1 上传视频

点击界面左侧"上传视频"，支持 mp4、mov、avi 等常见格式。

### 4.2 输入剪辑需求

用自然语言描述你想要什么，例如：

```
提取所有提到"竞品"和"价格"的片段，每段前后各保留5秒
```

```
截取视频的第2分钟到第8分钟，导出到YouTube
```

```
把这个采访剪成3条抖音短视频，每条不超过60秒，聚焦产品优势部分
```

```
删除所有超过1秒的停顿，保留说话内容，导出到B站
```

### 4.3 点击"开始分析"

系统依次执行以下步骤（状态栏实时更新）：

1. **转录**：faster-whisper 将视频音频转为带时间戳的文字（10 分钟视频约需 1~3 分钟）
2. **AI 解析**：Claude Haiku 将你的需求转换为结构化的剪辑计划
3. **规则匹配**：根据剪辑计划在转录结果中找出符合条件的候选片段

完成后状态栏显示：

```
找到 5 个候选片段 | 模式: highlight_extraction
```

### 4.4 审核候选片段

界面会出现一张可编辑的表格，每行是一个候选片段：

| 序号 | 时间范围 | 内容预览 | 置信度 | 包含 |
|------|---------|---------|------|------|
| 1 | 2.3s - 18.5s | 竞品的价格策略非常激进... | 1.00 | ☑ |
| 2 | 45.0s - 62.1s | 我们的产品比竞品便宜30%... | 1.00 | ☑ |
| 3 | 120.5s - 135.0s | 关于价格方面，用户反馈... | 1.00 | ☑ |

- 取消勾选不想要的片段
- 右侧可预览原始视频，对照时间戳确认内容

### 4.5 选择导出平台并导出

勾选目标平台（可多选）：

| 平台 | 画面比例 | 时长限制 | 备注 |
|------|---------|---------|------|
| 抖音 | 9:16 竖屏 | 60 秒/条 | 超长自动分割 |
| 微信视频号 | 9:16 竖屏 | 10 分钟 | 超长自动分割 |
| B站 | 16:9 横屏 | 无限制 | 流复制，速度极快 |
| YouTube | 16:9 横屏 | 无限制 | 流复制，速度极快 |

点击"批准并导出"，完成后显示所有输出文件的路径。

### 4.6 获取导出文件

导出文件保存在项目根目录的 `output/` 文件夹：

```
output/
  interview_douyin_<片段id>.mp4
  interview_douyin_<片段id>_part1.mp4    # 超长片段自动分割
  interview_douyin_<片段id>_part2.mp4
  interview_youtube_<片段id>.mp4
```

---

## 批量处理（Python API）

如需处理多个视频，可跳过 UI 直接调用 Python API：

```python
from pathlib import Path
from agent.orchestrator import Orchestrator
from processing.exporter import Exporter
from models.edit_plan import OutputFormat, Platform

orch = Orchestrator()
exporter = Exporter(output_dir="./output")

videos = list(Path("./input").glob("*.mp4"))
for video in videos:
    result = orch.run(
        video_path=video,
        user_instruction="提取所有提到竞品的片段，前后各保留5秒"
    )
    output_files = exporter.export(
        src=video,
        candidates=result.candidates,
        formats=[
            OutputFormat(platform=Platform.DOUYIN),
            OutputFormat(platform=Platform.YOUTUBE),
        ]
    )
    print(f"{video.name}: 找到 {len(result.candidates)} 个片段，导出 {len(output_files)} 个文件")
```

---

## 三种剪辑模式

Claude Haiku 会根据你的需求自动选择合适的模式：

| 模式 | 适用场景 | 示例指令 |
|------|---------|---------|
| **精华提取** | 从长视频中找关键片段 | `提取所有提到竞品价格的片段` |
| **素材拼接** | 按顺序组合多个片段 | `截取第1分钟开场、第5-8分钟产品演示、最后30秒结尾` |
| **社媒生产** | 生成适合发布的短视频 | `把采访剪成3条抖音，每条不超过60秒` |

---

## 常见问题

**Q：转录速度很慢怎么办？**

在 `.env` 中换小模型：`WHISPER_MODEL=small`；有 NVIDIA GPU 则设 `WHISPER_DEVICE=cuda`，速度可提升 5~10 倍。

**Q：找到的片段不准确怎么办？**

在需求中加入更具体的关键词，或指定时间范围辅助定位：

```
提取第5到第30分钟内提到"价格"、"优惠"、"折扣"的片段，每段保留前3秒后5秒
```

**Q：视频超过 2 小时怎么办？**

系统自动分块处理，无需手动切割。每块 30 分钟，处理后时间戳自动合并到原始时间轴。

**Q：导出的竖屏视频画面被裁切了？**

这是预期行为。原始横屏视频转为 9:16 竖屏时，系统做中心裁切保留画面中央内容。如果主体不在中央，建议源视频拍摄时就使用竖屏。

**Q：没有启动 Celery worker 可以直接用 UI 吗？**

可以。当前 UI 直接同步调用 Orchestrator，Celery 用于未来的进度推送和并发扩展。没有 Celery 时 UI 功能完整，只是转录期间页面会短暂无响应。

---

## 项目结构

```
auto-vedio-edit-agent/
├── app/
│   └── main.py              # Gradio UI 入口
├── agent/
│   ├── intent_parser.py     # LLM 意图解析（Claude Haiku）
│   ├── orchestrator.py      # 流水线编排器
│   └── rule_engine.py       # 规则执行引擎
├── processing/
│   ├── transcriber.py       # 语音转录（faster-whisper）
│   ├── ffmpeg_utils.py      # FFmpeg 工具函数
│   └── exporter.py          # 多平台导出
├── storage/
│   ├── base.py              # 存储抽象接口
│   ├── local.py             # 本地文件系统实现
│   ├── s3.py                # S3 存储（v1 占位）
│   └── factory.py           # 存储后端工厂
├── models/
│   └── edit_plan.py         # Pydantic 数据模型
├── tasks/
│   ├── celery_app.py        # Celery 应用配置
│   └── celery_tasks.py      # 异步任务定义
├── config/
│   └── settings.py          # 配置管理（Pydantic-Settings）
├── tests/                   # 测试套件（33 个测试）
├── docker-compose.yml       # Valkey + worker 容器编排
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 未来扩展

- **S3 云存储**：实现 `storage/s3.py`，将 `.env` 中 `STORAGE_BACKEND` 改为 `s3` 即可切换，其他代码零改动
- **视觉 AI**：在转录与规则引擎之间插入场景检测模块，支持按画面内容筛选片段
- **字幕嵌入**：导出时将转录文字烧录为字幕
- **多语言支持**：faster-whisper 原生支持 99 种语言，修改转录参数即可
