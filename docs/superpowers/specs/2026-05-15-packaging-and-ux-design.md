# 设计文档：打包、用户体验改进与说话人分离

**日期**：2026-05-15  
**状态**：待实现

---

## 目标

1. 让无编程经验的用户能在任何平台（macOS / Windows / Linux）通过 Docker 一键安装并使用
2. 将 API Key 和用户剪辑偏好分离到独立的、易于修改的文件中
3. 集成 WhisperX 实现多说话人识别，支持按说话人过滤提取片段

---

## 核心变更概览

| 变更 | 原状态 | 新状态 |
|------|--------|--------|
| 运行方式 | 手动建 venv + 分别启动 worker 和 UI | Docker Compose 一键全起 |
| API Key 位置 | 混在 `.env` 里 | 独立 `api_keys.env` 文件 |
| 非敏感配置 | 混在 `.env` 里 | 独立 `docker.env` 文件 |
| 用户偏好 | 无 | `USER_PREFERENCES.md`，注入 LLM 提示词 |
| 安装引导 | 无 | `setup.sh` / `setup.bat` 交互式向导 |
| 转录引擎 | faster-whisper（无说话人区分） | WhisperX（转录 + 词级对齐 + 说话人分离） |
| 说话人分离 | 不支持 | pyannote/speaker-diarization-3.1，本地运行 |
| Python 兼容性 | 需要 3.10+（`str \| None` 语法） | 本地支持 3.9+（`from __future__ import annotations`） |

---

## 一、文件结构

### 新增文件

```
api_keys.env.example          ← git 提交，密钥模板
api_keys.env                  ← git 忽略，用户填写
docker.env                    ← git 提交，非敏感 Docker 配置
USER_PREFERENCES.example.md  ← git 提交，偏好文件模板
USER_PREFERENCES.md           ← git 忽略，用户个人偏好
setup.sh                      ← macOS/Linux 安装向导
setup.bat                     ← Windows 安装向导
start.sh / start.bat          ← 重新启动服务
stop.sh / stop.bat            ← 停止服务
```

### 更新 .gitignore

新增忽略项：
```
api_keys.env
USER_PREFERENCES.md
output/
data/
tmp/
```

---

## 二、配置文件设计

### api_keys.env.example

```dotenv
# ============================================================
# API 密钥配置 — 这里是唯一需要填写的地方
# 修改后运行 ./start.sh 重启服务生效
# ============================================================

# Anthropic API Key（必填）
# 获取地址：https://console.anthropic.com → API Keys → Create Key
ANTHROPIC_API_KEY=your_key_here

# HuggingFace Token（说话人分离功能需要，不填则跳过分离）
# 第一步：https://huggingface.co/settings/tokens → 创建 Read token
# 第二步：访问以下两个页面，点击 "Agree and access repository" 接受授权：
#   https://huggingface.co/pyannote/speaker-diarization-3.1
#   https://huggingface.co/pyannote/segmentation-3.0
HF_TOKEN=
```

### docker.env

```dotenv
WHISPER_MODEL=medium
WHISPER_DEVICE=auto
STORAGE_BACKEND=local
LOCAL_STORAGE_ROOT=./data
TEMP_DIR=./tmp
OUTPUT_DIR=./output
CELERY_BROKER_URL=redis://valkey:6379/0
CELERY_RESULT_BACKEND=redis://valkey:6379/0
ENABLE_DIARIZATION=true
```

注意：
- `CELERY_BROKER_URL` 在 Docker 内使用服务名 `valkey`，不是 `localhost`
- `ENABLE_DIARIZATION=true` 时若 `HF_TOKEN` 为空，系统自动降级为无分离模式，不报错

---

## 三、Docker Compose 重构

当前 docker-compose 只包含 `valkey` + `worker`，UI 需手动启动。新版将 UI 也纳入编排。

### 三个服务

**valkey**：消息队列，仅内部访问，不暴露端口到宿主机。

**worker**：Celery 异步任务，读取 `api_keys.env` + `docker.env`，挂载 `data/`、`output/`、`tmp/` 目录。

**ui**：Gradio 应用，暴露 `7860:7860`，读取相同 env 文件，依赖 `valkey` 启动。所有服务设置 `restart: unless-stopped`。

### Celery Broker URL

docker-compose 内的服务间通信使用服务名，而非 `localhost`。`docker.env` 中配置为 `redis://valkey:6379/0`。

### Dockerfile 补充

当前 Dockerfile 缺少端口声明，补充：
```dockerfile
EXPOSE 7860
```

---

## 四、Setup 脚本逻辑

### setup.sh（macOS / Linux）

执行顺序：

1. 检测 `docker` 命令是否存在，否则打印下载链接并退出
2. 检测 Docker daemon 是否运行（`docker info`），否则提示启动 Docker Desktop 并退出
3. 如果 `api_keys.env` 不存在，从 `api_keys.env.example` 复制
4. 读取当前 `ANTHROPIC_API_KEY` 值，若为空或等于 `your_key_here`，交互提示用户粘贴 Key，用 `sed` 写入 `api_keys.env`
5. 运行 `docker compose up --build -d`
6. 打印成功信息，告知访问地址和后续命令

### setup.bat（Windows）

逻辑与 `setup.sh` 一致，使用 `where docker`、`docker info`、`findstr`、`powershell -Command "(Get-Content ...) -replace ..."` 实现相同步骤。

### start.sh / start.bat

```bash
docker compose up -d
```

不重新构建，适合日常启动。

### stop.sh / stop.bat

```bash
docker compose down
```

---

## 五、USER_PREFERENCES.md 注入机制

### 读取时机

`IntentParser.__init__` 时读取，作为实例变量缓存（避免每次 `parse()` 重复 IO）。

### 读取逻辑

```python
def _load_preferences(self) -> str:
    path = Path("USER_PREFERENCES.md")
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""
```

### 注入位置

拼接到 `SYSTEM_PROMPT` 开头，格式：

```
[用户剪辑偏好]
{preferences_content}

---

{原 SYSTEM_PROMPT}
```

若 `preferences_content` 为空，则直接使用原 `SYSTEM_PROMPT`，不附加任何内容。

### 不影响范围

偏好文件只影响 `IntentParser` 的 LLM 调用。规则引擎、转录、导出均不读取该文件。

---

## 六、Python 3.9 兼容性修复

在以下使用 `X | Y` union 语法的文件顶部添加 `from __future__ import annotations`：

- `processing/transcriber.py`
- `processing/exporter.py`
- `agent/rule_engine.py`
- `app/main.py`

此修改使类型注解在运行时作为字符串处理，Pydantic v2 可正确解析，测试可在本地 Python 3.9 运行。Docker 镜像继续使用 Python 3.11，不受影响。

---

## 七、WhisperX 说话人分离集成

### 依赖变更

`requirements.txt` 中：
- 移除 `faster-whisper`（whisperx 内部已包含）
- 新增 `whisperx>=3.1.0`
- 新增 `pyannote.audio>=3.1.0`

`Dockerfile` 需先安装 PyTorch CPU 版本（体积更小），再安装其余依赖：

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
```

docker-compose 新增模型缓存卷，避免每次重建容器重新下载模型（约 1.5 GB）：

```yaml
volumes:
  - hf_models:/root/.cache/huggingface
  - whisperx_models:/root/.cache/whisperx
```

### Segment 模型变更

`models/edit_plan.py` 中 `Segment` 新增可选字段：

```python
class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None  # 如 "SPEAKER_00"，无分离时为 None
```

### RuleType 新增 speaker_filter

```python
class RuleType(str, Enum):
    KEYWORD_MATCH = "keyword_match"
    TIME_RANGE = "time_range"
    SILENCE_CUT = "silence_cut"
    MIN_DURATION = "min_duration"
    SPEAKER_FILTER = "speaker_filter"   # 新增
```

`Rule` 模型新增字段：

```python
class Rule(BaseModel):
    ...
    speakers: list[str] = Field(default_factory=list)  # 用于 speaker_filter
```

### Transcriber 重写

用 WhisperX 三阶段流程替换原 faster-whisper 调用：

```
音频 → whisperx.transcribe（转录）
     → whisperx.align（词级时间戳对齐）
     → DiarizationPipeline（说话人分离，HF_TOKEN 存在时执行）
     → whisperx.assign_word_speakers（说话人标签合并到 segment）
     → 转换为 Segment 列表
```

`HF_TOKEN` 为空或 `ENABLE_DIARIZATION=false` 时，跳过分离步骤，`speaker` 字段为 `None`，其余功能不受影响。

### IntentParser 更新

**转录文本格式**（有说话人时）：

```
[12.3s – 18.5s] SPEAKER_00: 我们这款产品比竞品便宜 30%
[19.1s – 24.8s] SPEAKER_01: 那在续航方面呢？
```

**LLM schema 新增 speaker_filter 规则类型**：

```json
{
  "type": "speaker_filter",
  "speakers": ["SPEAKER_00"],
  "padding_before_sec": 2,
  "padding_after_sec": 3
}
```

System prompt 说明：仅当 transcript 中含有 SPEAKER_xx 标签时才可使用 speaker_filter，否则使用 keyword_match。

### UI 变更

候选片段表格在有说话人数据时新增"说话人"列：

| 序号 | 说话人 | 时间范围 | 内容预览 | 置信度 | 包含 |
|------|--------|---------|---------|------|------|
| 1 | SPEAKER_00 | 12.3s – 18.5s | 我们这款产品比竞品... | 1.00 | ☑ |

无分离数据时，该列不显示（向后兼容）。

### Settings 变更

```python
class Settings(BaseSettings):
    ...
    hf_token: Optional[str] = None
    enable_diarization: bool = True
```

### setup.sh 变更

在填入 Anthropic Key 之后，新增 HF Token 引导步骤（可按回车跳过）：

```
[可选] 说话人分离功能需要 HuggingFace Token
  获取地址：https://huggingface.co/settings/tokens
  注意：还需在 HuggingFace 接受 pyannote 模型授权（详见 api_keys.env）
请粘贴 HF Token（直接回车跳过，说话人分离将不可用）：
```

---

## 八、不在本次范围内

- 字幕烧录（FFmpeg subtitles filter）
- S3 云存储实现
- 视觉 AI 场景检测
- USER_PREFERENCES.md 中结构化字段的程序化解析（当前全部作为自然语言注入 LLM）
- GPU 加速支持（Dockerfile 使用 CPU 版 torch；有 GPU 的用户需自行替换 torch 安装命令）
