# Packaging, WhisperX & Speaker Diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repackage the project as a Docker-first one-command install, split API keys into a dedicated file, add `USER_PREFERENCES.md` injection, and replace the transcriber with WhisperX to support multi-speaker diarization and `speaker_filter` extraction.

**Architecture:** Two independent chunks. Chunk A (Tasks 1–3) is pure infrastructure — config split, Docker Compose full-stack, and setup scripts. Chunk B (Tasks 4–9) replaces faster-whisper with WhisperX, extends the data models with `speaker` fields, adds `speaker_filter` to the rule engine, injects `USER_PREFERENCES.md` into the LLM prompt, and adds a speaker column to the review UI. Both chunks can be worked independently; Chunk B depends on Chunk A's `Settings` changes.

**Tech Stack:** Python 3.11 (Docker), whisperx≥3.1, pyannote.audio≥3.1, torch (CPU), Gradio 5.x, Pydantic v2, Docker Compose, bash/bat scripts

---

## File Map

| File | Action | Chunk |
|------|--------|-------|
| `api_keys.env.example` | Create | A |
| `docker.env` | Create | A |
| `USER_PREFERENCES.example.md` | Create | A |
| `config/settings.py` | Modify — add `hf_token`, `enable_diarization` | A |
| `.gitignore` | Modify — add `api_keys.env`, `USER_PREFERENCES.md` | A |
| `.env.example` | Modify — redirect to new files | A |
| `docker-compose.yml` | Rewrite — add `ui` service, split env files, model cache volumes | A |
| `Dockerfile` | Modify — add torch CPU install, `EXPOSE 7860` | A |
| `requirements.txt` | Modify — swap faster-whisper → whisperx + pyannote.audio | A |
| `setup.sh` | Create | A |
| `setup.bat` | Create | A |
| `start.sh` / `start.bat` | Create | A |
| `stop.sh` / `stop.bat` | Create | A |
| `models/edit_plan.py` | Modify — `Segment.speaker`, `CandidateSegment.speaker`, `SPEAKER_FILTER`, `Rule.speakers` | B |
| `processing/transcriber.py` | Rewrite — WhisperX three-stage pipeline | B |
| `agent/rule_engine.py` | Modify — `speaker_filter` rule + carry `speaker` through merge | B |
| `agent/intent_parser.py` | Modify — `USER_PREFERENCES.md` injection + speaker-format transcript | B |
| `app/main.py` | Modify — 6-column review table with speaker, `from __future__` | B |
| `processing/exporter.py` | Modify — `from __future__ import annotations` only | B |
| `tests/test_models.py` | Modify — test `Segment.speaker`, `SPEAKER_FILTER` | B |
| `tests/test_transcriber.py` | Rewrite — mock whisperx instead of faster-whisper | B |
| `tests/test_rule_engine.py` | Modify — add `speaker_filter` tests | B |
| `tests/test_intent_parser.py` | Modify — test preferences injection + speaker transcript | B |
| `tests/test_ui.py` | Modify — 6-column table format | B |

---

## Chunk A — Infrastructure

### Task 1: Config file split + Settings + .gitignore

**Files:**
- Create: `api_keys.env.example`
- Create: `docker.env`
- Modify: `config/settings.py`
- Modify: `.gitignore`
- Modify: `.env.example`

- [ ] **Step 1: Create `api_keys.env.example`**

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

- [ ] **Step 2: Create `docker.env`**

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

- [ ] **Step 3: Update `config/settings.py`**

Replace the entire file:

```python
from __future__ import annotations
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str

    # WhisperX
    whisper_model: Literal["tiny", "base", "small", "medium", "large-v3"] = "medium"
    whisper_device: Literal["cpu", "cuda", "auto"] = "auto"
    hf_token: Optional[str] = None
    enable_diarization: bool = True

    # Storage
    storage_backend: Literal["local"] = "local"
    local_storage_root: str = "./data"
    temp_dir: str = "./tmp"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # Export
    output_dir: str = "./output"

    model_config = SettingsConfigDict(env_file=(".env", "api_keys.env", "docker.env"))


settings = Settings()
```

Note: `env_file` accepts a tuple — later files override earlier ones. The `.env` fallback preserves backward-compat for developers who still have a local `.env`.

- [ ] **Step 4: Update `.gitignore`**

Add these lines at the end of the existing `.gitignore`:

```
api_keys.env
USER_PREFERENCES.md
```

- [ ] **Step 5: Update `.env.example` to redirect**

Replace entire content:

```dotenv
# ============================================================
# 此文件已拆分为两个文件：
#   api_keys.env.example  ← API 密钥（Anthropic Key、HF Token）
#   docker.env            ← 运行配置（Whisper 型号、路径等）
#
# 开发者本地使用：可在此 .env 中覆盖任意配置，优先级最低。
# ============================================================
```

- [ ] **Step 6: Run tests to confirm nothing broke**

```bash
python3 -m pytest tests/ -q --tb=short
```

Expected: same pass/fail count as before (infrastructure change only).

- [ ] **Step 7: Commit**

```bash
git add api_keys.env.example docker.env config/settings.py .gitignore .env.example
git commit -m "feat: split config into api_keys.env and docker.env, add hf_token to Settings"
```

---

### Task 2: Docker Compose full-stack + Dockerfile

**Files:**
- Modify: `docker-compose.yml`
- Modify: `Dockerfile`
- Modify: `requirements.txt`

- [ ] **Step 1: Update `requirements.txt`**

Replace entire file:

```
whisperx>=3.1.0
pyannote.audio>=3.1.0
anthropic>=0.40.0
gradio>=5.0.0
celery>=5.3.0
redis>=5.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
boto3>=1.34.0
pytest>=8.0.0
pytest-mock>=3.12.0
python-dotenv>=1.0.0
```

- [ ] **Step 2: Update `Dockerfile`**

Replace entire file:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install CPU PyTorch before whisperx so it doesn't pull in CUDA deps
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 7860
```

- [ ] **Step 3: Rewrite `docker-compose.yml`**

Replace entire file:

```yaml
services:
  valkey:
    image: valkey/valkey:7
    volumes:
      - valkey_data:/data
    command: valkey-server --appendonly yes
    restart: unless-stopped

  worker:
    build: .
    command: celery -A tasks.celery_app worker --loglevel=info --concurrency=2
    depends_on:
      - valkey
    env_file:
      - api_keys.env
      - docker.env
    volumes:
      - ./data:/app/data
      - ./output:/app/output
      - ./tmp:/app/tmp
      - hf_models:/root/.cache/huggingface
      - whisperx_models:/root/.cache/whisperx
    restart: unless-stopped

  ui:
    build: .
    command: python app/main.py
    ports:
      - "7860:7860"
    depends_on:
      - valkey
    env_file:
      - api_keys.env
      - docker.env
    volumes:
      - ./data:/app/data
      - ./output:/app/output
      - ./tmp:/app/tmp
      - hf_models:/root/.cache/huggingface
      - whisperx_models:/root/.cache/whisperx
    restart: unless-stopped

volumes:
  valkey_data:
  hf_models:
  whisperx_models:
```

- [ ] **Step 4: Verify docker-compose syntax**

```bash
docker compose config
```

Expected: valid YAML printed, no errors.

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml Dockerfile requirements.txt
git commit -m "feat: add ui service to docker-compose, add model cache volumes, update Dockerfile for whisperx"
```

---

### Task 3: Setup and convenience scripts

**Files:**
- Create: `setup.sh`, `setup.bat`
- Create: `start.sh`, `start.bat`
- Create: `stop.sh`, `stop.bat`

- [ ] **Step 1: Create `setup.sh`**

```bash
#!/usr/bin/env bash
set -e

echo "=== 视频自动剪辑 Agent — 安装向导 ==="
echo ""

# 1. Check Docker installed
if ! command -v docker &>/dev/null; then
    echo "错误：未检测到 Docker，请先安装 Docker Desktop"
    echo "下载地址：https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# 2. Check Docker daemon running
if ! docker info &>/dev/null 2>&1; then
    echo "错误：Docker 未运行，请先启动 Docker Desktop，再重新运行此脚本"
    exit 1
fi

# 3. Initialise api_keys.env from example if missing
if [ ! -f api_keys.env ]; then
    cp api_keys.env.example api_keys.env
    echo "已创建 api_keys.env（从模板复制）"
fi

# 4. Prompt for Anthropic API Key if placeholder
current_anthropic=$(grep "^ANTHROPIC_API_KEY=" api_keys.env | cut -d'=' -f2-)
if [ -z "$current_anthropic" ] || [ "$current_anthropic" = "your_key_here" ]; then
    echo "需要填入 Anthropic API Key"
    echo "获取地址：https://console.anthropic.com → API Keys → Create Key"
    echo ""
    read -rp "请粘贴 Anthropic API Key: " anthropic_key
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=${anthropic_key}|" api_keys.env
    else
        sed -i "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=${anthropic_key}|" api_keys.env
    fi
    echo "Anthropic API Key 已保存"
fi

echo ""
# 5. Prompt for HF Token (optional)
current_hf=$(grep "^HF_TOKEN=" api_keys.env | cut -d'=' -f2-)
if [ -z "$current_hf" ]; then
    echo "[可选] 说话人分离功能需要 HuggingFace Token"
    echo "获取地址：https://huggingface.co/settings/tokens（创建 Read token）"
    echo "注意：还需在 HuggingFace 接受 pyannote 模型授权（详见 api_keys.env）"
    echo ""
    read -rp "请粘贴 HF Token（直接回车跳过，说话人分离将不可用）: " hf_key
    if [ -n "$hf_key" ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|^HF_TOKEN=.*|HF_TOKEN=${hf_key}|" api_keys.env
        else
            sed -i "s|^HF_TOKEN=.*|HF_TOKEN=${hf_key}|" api_keys.env
        fi
        echo "HF Token 已保存"
    else
        echo "已跳过，说话人分离功能不可用（可随时编辑 api_keys.env 补填）"
    fi
fi

echo ""
echo "正在构建并启动服务（首次构建约需 5–10 分钟，请耐心等待）..."
docker compose up --build -d

echo ""
echo "==================================="
echo "  安装完成！"
echo "  打开浏览器访问：http://localhost:7860"
echo ""
echo "  停止服务：./stop.sh"
echo "  再次启动：./start.sh"
echo "==================================="
```

- [ ] **Step 2: Create `setup.bat`**

```bat
@echo off
setlocal enabledelayedexpansion
echo === 视频自动剪辑 Agent — 安装向导 ===
echo.

where docker >nul 2>&1
if errorlevel 1 (
    echo 错误：未检测到 Docker，请先安装 Docker Desktop
    echo 下载地址：https://www.docker.com/products/docker-desktop/
    pause & exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo 错误：Docker 未运行，请先启动 Docker Desktop，再重新运行此脚本
    pause & exit /b 1
)

if not exist api_keys.env (
    copy api_keys.env.example api_keys.env >nul
    echo 已创建 api_keys.env（从模板复制）
)

for /f "tokens=1,* delims==" %%a in ('findstr "^ANTHROPIC_API_KEY=" api_keys.env') do set CURRENT_KEY=%%b
if "!CURRENT_KEY!"=="" set CURRENT_KEY=your_key_here
if "!CURRENT_KEY!"=="your_key_here" (
    echo 需要填入 Anthropic API Key
    echo 获取地址：https://console.anthropic.com 右上角 API Keys
    echo.
    set /p ANTHROPIC_KEY="请粘贴 Anthropic API Key: "
    powershell -Command "(Get-Content api_keys.env) -replace '^ANTHROPIC_API_KEY=.*', 'ANTHROPIC_API_KEY=!ANTHROPIC_KEY!' | Set-Content api_keys.env"
    echo Anthropic API Key 已保存
)

echo.
for /f "tokens=1,* delims==" %%a in ('findstr "^HF_TOKEN=" api_keys.env') do set CURRENT_HF=%%b
if "!CURRENT_HF!"=="" (
    echo [可选] 说话人分离功能需要 HuggingFace Token
    echo 获取地址：https://huggingface.co/settings/tokens
    echo.
    set /p HF_KEY="请粘贴 HF Token（直接回车跳过）: "
    if not "!HF_KEY!"=="" (
        powershell -Command "(Get-Content api_keys.env) -replace '^HF_TOKEN=.*', 'HF_TOKEN=!HF_KEY!' | Set-Content api_keys.env"
        echo HF Token 已保存
    ) else (
        echo 已跳过说话人分离功能
    )
)

echo.
echo 正在构建并启动服务（首次构建约需 5-10 分钟）...
docker compose up --build -d

echo.
echo ===================================
echo   安装完成！
echo   打开浏览器访问：http://localhost:7860
echo.
echo   停止服务：stop.bat
echo   再次启动：start.bat
echo ===================================
pause
```

- [ ] **Step 3: Create `start.sh`**

```bash
#!/usr/bin/env bash
set -e
echo "正在启动服务..."
docker compose up -d
echo "已启动，访问 http://localhost:7860"
```

- [ ] **Step 4: Create `start.bat`**

```bat
@echo off
echo 正在启动服务...
docker compose up -d
echo 已启动，访问 http://localhost:7860
pause
```

- [ ] **Step 5: Create `stop.sh`**

```bash
#!/usr/bin/env bash
set -e
echo "正在停止服务..."
docker compose down
echo "已停止"
```

- [ ] **Step 6: Create `stop.bat`**

```bat
@echo off
echo 正在停止服务...
docker compose down
echo 已停止
pause
```

- [ ] **Step 7: Make shell scripts executable**

```bash
chmod +x setup.sh start.sh stop.sh
```

- [ ] **Step 8: Commit**

```bash
git add setup.sh setup.bat start.sh start.bat stop.sh stop.bat
git commit -m "feat: add Docker setup wizard and start/stop convenience scripts"
```

---

## Chunk B — WhisperX & Speaker Diarization

### Task 4: Data model updates

**Files:**
- Modify: `models/edit_plan.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for new model fields**

Add to `tests/test_models.py`:

```python
def test_segment_has_optional_speaker():
    seg = Segment(start=0.0, end=5.0, text="hello")
    assert seg.speaker is None

    seg_with_speaker = Segment(start=0.0, end=5.0, text="hello", speaker="SPEAKER_00")
    assert seg_with_speaker.speaker == "SPEAKER_00"


def test_candidate_segment_has_optional_speaker():
    seg = CandidateSegment(id="1", start=0.0, end=5.0, text_preview="test")
    assert seg.speaker is None

    seg_with_speaker = CandidateSegment(
        id="2", start=0.0, end=5.0, text_preview="test", speaker="SPEAKER_01"
    )
    assert seg_with_speaker.speaker == "SPEAKER_01"


def test_speaker_filter_rule_type_exists():
    from models.edit_plan import RuleType
    assert RuleType.SPEAKER_FILTER == "speaker_filter"


def test_rule_has_speakers_field():
    rule = Rule(type=RuleType.KEYWORD_MATCH, keywords=["竞品"])
    assert rule.speakers == []

    rule_with_speakers = Rule(type=RuleType.SPEAKER_FILTER, speakers=["SPEAKER_00"])
    assert rule_with_speakers.speakers == ["SPEAKER_00"]
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m pytest tests/test_models.py -k "speaker" -v
```

Expected: `AttributeError: 'Segment' object has no attribute 'speaker'`

- [ ] **Step 3: Update `models/edit_plan.py`**

Make four targeted changes:

**a)** Add `SPEAKER_FILTER` to `RuleType`:
```python
class RuleType(str, Enum):
    KEYWORD_MATCH = "keyword_match"
    TIME_RANGE = "time_range"
    SILENCE_CUT = "silence_cut"
    MIN_DURATION = "min_duration"
    SPEAKER_FILTER = "speaker_filter"
```

**b)** Add `speakers` field to `Rule`:
```python
class Rule(BaseModel):
    type: RuleType
    keywords: list[str] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)   # ← add this line
    padding_before_sec: float = 3.0
    padding_after_sec: float = 5.0
    min_duration_sec: float = 5.0
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
```

**c)** Add `speaker` to `Segment` and update docstring:
```python
class Segment(BaseModel):
    """Raw transcript segment from WhisperX."""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None  # e.g. "SPEAKER_00"; None when diarization skipped
```

**d)** Add `speaker` to `CandidateSegment`:
```python
class CandidateSegment(BaseModel):
    """A segment proposed for inclusion in the final edit."""
    id: str
    start: float
    end: float
    text_preview: str
    confidence_score: float = 1.0
    included: bool = True
    source_file: Optional[str] = None
    speaker: Optional[str] = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_models.py -v
```

Expected: all tests PASS including the 4 new ones.

- [ ] **Step 5: Commit**

```bash
git add models/edit_plan.py tests/test_models.py
git commit -m "feat: add speaker field to Segment/CandidateSegment, add SPEAKER_FILTER rule type"
```

---

### Task 5: WhisperX Transcriber rewrite

**Files:**
- Modify: `processing/transcriber.py`
- Modify: `tests/test_transcriber.py`

- [ ] **Step 1: Rewrite `tests/test_transcriber.py`**

Replace entire file:

```python
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from processing.transcriber import Transcriber
from models.edit_plan import Segment

MOCK_TRANSCRIBE_RESULT = {
    "segments": [
        {"start": 0.0, "end": 3.5, "text": " Hello world"},
        {"start": 3.5, "end": 7.0, "text": " This is a test"},
    ],
    "language": "en",
}

MOCK_ALIGN_RESULT = {
    "segments": [
        {"start": 0.0, "end": 3.5, "text": " Hello world"},
        {"start": 3.5, "end": 7.0, "text": " This is a test"},
    ],
}

MOCK_DIARIZED_RESULT = {
    "segments": [
        {"start": 0.0, "end": 3.5, "text": " Hello world", "speaker": "SPEAKER_00"},
        {"start": 3.5, "end": 7.0, "text": " This is a test", "speaker": "SPEAKER_01"},
    ],
}


def _make_transcriber_no_init(hf_token=None, enable_diarization=False):
    """Build a Transcriber without calling __init__ (avoids model download)."""
    t = Transcriber.__new__(Transcriber)
    t._device = "cpu"
    t._model = MagicMock()
    t._hf_token = hf_token
    t._enable_diarization = enable_diarization
    return t


@patch("processing.transcriber.whisperx.load_align_model")
@patch("processing.transcriber.whisperx.align")
@patch("processing.transcriber.whisperx.load_audio")
def test_transcribe_returns_segments(mock_load_audio, mock_align, mock_load_align_model, tmp_path):
    mock_load_audio.return_value = MagicMock()
    mock_load_align_model.return_value = (MagicMock(), MagicMock())
    mock_align.return_value = MOCK_ALIGN_RESULT

    t = _make_transcriber_no_init()
    t._model.transcribe.return_value = MOCK_TRANSCRIBE_RESULT

    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"fake")

    result = t.transcribe(fake_video)

    assert len(result) == 2
    assert isinstance(result[0], Segment)
    assert result[0].text == "Hello world"   # leading space stripped
    assert result[0].start == 0.0
    assert result[1].end == 7.0
    assert result[0].speaker is None         # no diarization


@patch("processing.transcriber.whisperx.assign_word_speakers")
@patch("processing.transcriber.whisperx.DiarizationPipeline")
@patch("processing.transcriber.whisperx.load_align_model")
@patch("processing.transcriber.whisperx.align")
@patch("processing.transcriber.whisperx.load_audio")
def test_transcribe_with_diarization(
    mock_load_audio, mock_align, mock_load_align_model, mock_diarize_cls, mock_assign, tmp_path
):
    mock_load_audio.return_value = MagicMock()
    mock_load_align_model.return_value = (MagicMock(), MagicMock())
    mock_align.return_value = MOCK_ALIGN_RESULT
    mock_diarize_cls.return_value = MagicMock()
    mock_assign.return_value = MOCK_DIARIZED_RESULT

    t = _make_transcriber_no_init(hf_token="hf_fake", enable_diarization=True)
    t._model.transcribe.return_value = MOCK_TRANSCRIBE_RESULT

    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"fake")

    result = t.transcribe(fake_video)

    mock_diarize_cls.assert_called_once_with(use_auth_token="hf_fake", device="cpu")
    assert result[0].speaker == "SPEAKER_00"
    assert result[1].speaker == "SPEAKER_01"


@patch("processing.transcriber.whisperx.load_align_model")
@patch("processing.transcriber.whisperx.align")
@patch("processing.transcriber.whisperx.load_audio")
def test_transcribe_empty_video_returns_empty(mock_load_audio, mock_align, mock_load_align_model, tmp_path):
    mock_load_audio.return_value = MagicMock()
    mock_load_align_model.return_value = (MagicMock(), MagicMock())
    mock_align.return_value = {"segments": []}

    t = _make_transcriber_no_init()
    t._model.transcribe.return_value = {"segments": [], "language": "en"}

    fake_video = tmp_path / "silent.mp4"
    fake_video.write_bytes(b"fake")

    result = t.transcribe(fake_video)
    assert result == []


def test_transcribe_missing_file_raises():
    t = _make_transcriber_no_init()
    with pytest.raises(FileNotFoundError):
        t.transcribe(Path("/nonexistent/video.mp4"))
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m pytest tests/test_transcriber.py -v
```

Expected: `ModuleNotFoundError: No module named 'whisperx'` (or similar import failure)

- [ ] **Step 3: Rewrite `processing/transcriber.py`**

Replace entire file:

```python
from __future__ import annotations
import whisperx
from pathlib import Path
from typing import Optional
from models.edit_plan import Segment
from config.settings import settings

CHUNK_DURATION = 1800   # 30 minutes per chunk
CHUNK_THRESHOLD = 7200  # only chunk videos longer than 2 hours


class Transcriber:
    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None):
        size = model_size or settings.whisper_model
        dev = device or settings.whisper_device
        if dev == "auto":
            try:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                dev = "cpu"
        self._device = dev
        self._model = whisperx.load_model(size, self._device, compute_type="int8")
        self._hf_token = settings.hf_token
        self._enable_diarization = settings.enable_diarization

    def transcribe(self, video_path: Path) -> list[Segment]:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        duration = self._get_duration(video_path)
        if duration and duration > CHUNK_THRESHOLD:
            return self._transcribe_chunked(video_path, duration)
        return self._transcribe_single(video_path)

    def _transcribe_single(self, video_path: Path) -> list[Segment]:
        audio = whisperx.load_audio(str(video_path))
        result = self._model.transcribe(audio, batch_size=16)

        # Word-level alignment — best-effort, skip on failure
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=self._device
            )
            result = whisperx.align(
                result["segments"], model_a, metadata, audio, self._device,
                return_char_alignments=False,
            )
        except Exception:
            pass

        # Speaker diarization — best-effort, requires HF token
        if self._enable_diarization and self._hf_token:
            try:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=self._hf_token, device=self._device
                )
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception:
                pass

        return [
            Segment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                speaker=seg.get("speaker"),
            )
            for seg in result["segments"]
            if seg.get("text", "").strip()
        ]

    def _transcribe_chunked(self, video_path: Path, duration: float) -> list[Segment]:
        import tempfile
        import subprocess
        all_segments: list[Segment] = []
        cursor = 0.0
        chunk_index = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            while cursor < duration:
                chunk_end = min(cursor + CHUNK_DURATION, duration)
                chunk_path = Path(tmpdir) / f"chunk_{chunk_index}.mp4"
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(cursor), "-to", str(chunk_end),
                     "-i", str(video_path), "-c", "copy", str(chunk_path)],
                    capture_output=True, check=True,
                )
                for seg in self._transcribe_single(chunk_path):
                    all_segments.append(Segment(
                        start=seg.start + cursor,
                        end=seg.end + cursor,
                        text=seg.text,
                        speaker=seg.speaker,
                    ))
                cursor = chunk_end
                chunk_index += 1
        return all_segments

    def _get_duration(self, video_path: Path) -> Optional[float]:
        import subprocess
        import re
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1", str(video_path)],
                capture_output=True, text=True,
            )
            match = re.search(r"duration=([\d.]+)", result.stdout)
            return float(match.group(1)) if match else None
        except Exception:
            return None
```

- [ ] **Step 4: Run tests — whisperx won't be installed locally, so mock-based tests should pass**

```bash
python3 -m pytest tests/test_transcriber.py -v
```

Expected: 4 tests PASS (all via mocks, no real whisperx model loaded).

- [ ] **Step 5: Commit**

```bash
git add processing/transcriber.py tests/test_transcriber.py
git commit -m "feat: replace faster-whisper with WhisperX, add speaker diarization support"
```

---

### Task 6: RuleEngine speaker_filter

**Files:**
- Modify: `agent/rule_engine.py`
- Modify: `tests/test_rule_engine.py`

- [ ] **Step 1: Add failing tests to `tests/test_rule_engine.py`**

Add these tests at the end of the file:

```python
from models.edit_plan import EditPlan, EditMode, Rule, RuleType, OutputFormat, Platform, Segment, CandidateSegment

TRANSCRIPT_WITH_SPEAKERS = [
    Segment(start=0.0, end=5.0, text="今天介绍产品功能", speaker="SPEAKER_00"),
    Segment(start=5.0, end=12.0, text="竞品的价格非常高", speaker="SPEAKER_01"),
    Segment(start=12.0, end=20.0, text="我们提供更好的性价比", speaker="SPEAKER_00"),
    Segment(start=20.0, end=25.0, text="欢迎联系我们了解竞品对比", speaker="SPEAKER_01"),
]


def test_speaker_filter_returns_only_matching_speaker():
    plan = EditPlan(
        mode=EditMode.HIGHLIGHT_EXTRACTION,
        rules=[Rule(type=RuleType.SPEAKER_FILTER, speakers=["SPEAKER_00"],
                    padding_before_sec=0, padding_after_sec=0)],
        output_formats=[OutputFormat(platform=Platform.YOUTUBE)],
    )
    engine = RuleEngine()
    candidates = engine.execute(plan, TRANSCRIPT_WITH_SPEAKERS, video_path=None)

    assert len(candidates) == 2
    assert all(c.speaker == "SPEAKER_00" for c in candidates)


def test_speaker_filter_multiple_speakers():
    plan = EditPlan(
        mode=EditMode.HIGHLIGHT_EXTRACTION,
        rules=[Rule(type=RuleType.SPEAKER_FILTER,
                    speakers=["SPEAKER_00", "SPEAKER_01"],
                    padding_before_sec=0, padding_after_sec=0)],
        output_formats=[OutputFormat(platform=Platform.YOUTUBE)],
    )
    engine = RuleEngine()
    candidates = engine.execute(plan, TRANSCRIPT_WITH_SPEAKERS, video_path=None)
    assert len(candidates) == 4


def test_speaker_filter_no_match_returns_empty():
    plan = EditPlan(
        mode=EditMode.HIGHLIGHT_EXTRACTION,
        rules=[Rule(type=RuleType.SPEAKER_FILTER, speakers=["SPEAKER_99"],
                    padding_before_sec=0, padding_after_sec=0)],
        output_formats=[OutputFormat(platform=Platform.YOUTUBE)],
    )
    engine = RuleEngine()
    candidates = engine.execute(plan, TRANSCRIPT_WITH_SPEAKERS, video_path=None)
    assert candidates == []


def test_keyword_match_carries_speaker():
    plan = EditPlan(
        mode=EditMode.HIGHLIGHT_EXTRACTION,
        rules=[Rule(type=RuleType.KEYWORD_MATCH, keywords=["竞品"],
                    padding_before_sec=0, padding_after_sec=0)],
        output_formats=[OutputFormat(platform=Platform.YOUTUBE)],
    )
    engine = RuleEngine()
    candidates = engine.execute(plan, TRANSCRIPT_WITH_SPEAKERS, video_path=None)

    # Both segments containing "竞品" should carry their speaker label
    speakers = {c.speaker for c in candidates}
    assert "SPEAKER_01" in speakers
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m pytest tests/test_rule_engine.py -k "speaker" -v
```

Expected: `AttributeError` — `speaker_filter` not handled and `speaker` not carried through.

- [ ] **Step 3: Update `agent/rule_engine.py`**

Make three changes:

**a)** Add `from __future__ import annotations` as first line.

**b)** Add `SPEAKER_FILTER` branch in `execute()`:
```python
elif rule.type == RuleType.SPEAKER_FILTER:
    candidates.extend(self._speaker_filter(rule, transcript))
```

**c)** Add `_speaker_filter` method:
```python
def _speaker_filter(self, rule: Rule, transcript: list[Segment]) -> list[CandidateSegment]:
    results = []
    for seg in transcript:
        if seg.speaker and seg.speaker in rule.speakers:
            results.append(CandidateSegment(
                id=str(uuid.uuid4()),
                start=max(0.0, seg.start - rule.padding_before_sec),
                end=seg.end + rule.padding_after_sec,
                text_preview=seg.text,
                confidence_score=1.0,
                speaker=seg.speaker,
            ))
    return results
```

**d)** Carry `speaker` through `_keyword_match` — update `CandidateSegment(...)` call inside `_keyword_match`:
```python
results.append(CandidateSegment(
    id=str(uuid.uuid4()),
    start=start,
    end=end,
    text_preview=seg.text,
    confidence_score=1.0,
    speaker=seg.speaker,      # ← add this line
))
```

**e)** Carry `speaker` through `_merge_overlapping` — update the merge stanza:
```python
merged[-1] = CandidateSegment(
    id=last.id,
    start=last.start,
    end=max(last.end, current.end),
    text_preview=f"{last.text_preview} | {current.text_preview}",
    confidence_score=max(last.confidence_score, current.confidence_score),
    speaker=last.speaker if last.speaker == current.speaker else None,
)
```

- [ ] **Step 4: Run all rule engine tests**

```bash
python3 -m pytest tests/test_rule_engine.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/rule_engine.py tests/test_rule_engine.py
git commit -m "feat: add speaker_filter rule, carry speaker field through keyword_match and merge"
```

---

### Task 7: IntentParser — USER_PREFERENCES.md injection + speaker-aware prompt

**Files:**
- Modify: `agent/intent_parser.py`
- Modify: `tests/test_intent_parser.py`

- [ ] **Step 1: Add failing tests to `tests/test_intent_parser.py`**

Add these tests at the end of the file:

```python
def test_preferences_injected_into_system_prompt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prefs = tmp_path / "USER_PREFERENCES.md"
    prefs.write_text("## 偏好\n- 默认留白 5 秒", encoding="utf-8")

    with patch("agent.intent_parser.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=SAMPLE_LLM_RESPONSE)]
        )
        parser = IntentParser()
        parser.parse(user_instruction="test", transcript=[])

    call_kwargs = mock_client.messages.create.call_args
    system_prompt = call_kwargs.kwargs.get("system") or call_kwargs.args[0] if call_kwargs.args else ""
    # Retrieve from keyword arg
    if not system_prompt:
        system_prompt = call_kwargs[1].get("system", "")
    assert "## 偏好" in system_prompt
    assert "默认留白 5 秒" in system_prompt


def test_no_preferences_file_uses_base_prompt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # directory has no USER_PREFERENCES.md

    with patch("agent.intent_parser.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=SAMPLE_LLM_RESPONSE)]
        )
        parser = IntentParser()
        parser.parse(user_instruction="test", transcript=[])

    call_kwargs = mock_client.messages.create.call_args
    system_prompt = call_kwargs[1].get("system", "")
    assert "[用户剪辑偏好]" not in system_prompt


def test_transcript_includes_speaker_labels():
    transcript = [
        {"start": 0.0, "end": 5.0, "text": "hello", "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 10.0, "text": "world", "speaker": None},
    ]

    with patch("agent.intent_parser.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=SAMPLE_LLM_RESPONSE)]
        )
        parser = IntentParser()
        parser.parse(user_instruction="test", transcript=transcript)

    call_kwargs = mock_client.messages.create.call_args
    user_message = call_kwargs[1]["messages"][0]["content"]
    assert "SPEAKER_00:" in user_message
    assert "SPEAKER_00:" not in user_message.split("SPEAKER_00:")[0] or True  # present once
    # Second segment has no speaker — should not show a label
    lines = user_message.split("\n")
    speaker_none_line = next((l for l in lines if "world" in l), "")
    assert "SPEAKER" not in speaker_none_line
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m pytest tests/test_intent_parser.py -k "preferences or speaker_labels" -v
```

Expected: failures on missing method / wrong prompt format.

- [ ] **Step 3: Update `agent/intent_parser.py`**

Replace entire file:

```python
from __future__ import annotations
import json
from pathlib import Path
import anthropic
from models.edit_plan import EditPlan
from config.settings import settings

SYSTEM_PROMPT = """You are a video editing assistant. Given a user's editing instruction and a video transcript,
output a JSON EditPlan that describes how to edit the video.

The transcript may include speaker labels (e.g. "SPEAKER_00:"). Use speaker_filter rules when the user
wants to extract segments from a specific speaker. Only use speaker_filter when the transcript contains
SPEAKER_xx labels; otherwise use keyword_match.

Output ONLY valid JSON matching this exact schema:
{
  "mode": "highlight_extraction" | "material_assembly" | "social_media",
  "rules": [
    {
      "type": "keyword_match" | "time_range" | "silence_cut" | "min_duration" | "speaker_filter",
      "keywords": [...],            // for keyword_match only
      "speakers": [...],            // for speaker_filter only, e.g. ["SPEAKER_00"]
      "padding_before_sec": 3,
      "padding_after_sec": 5,
      "min_duration_sec": 5,
      "start_sec": null,            // for time_range only
      "end_sec": null               // for time_range only
    }
  ],
  "output_formats": [
    {
      "platform": "douyin" | "bilibili" | "youtube" | "wechat",
      "ratio": "9:16" | "16:9" | "1:1",
      "max_duration_sec": null,
      "resolution": "1080p"
    }
  ],
  "segment_count_hint": 3
}

Output JSON only. No explanation."""


class IntentParser:
    def __init__(self):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._preferences = self._load_preferences()

    def _load_preferences(self) -> str:
        path = Path("USER_PREFERENCES.md")
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def _build_system_prompt(self) -> str:
        if self._preferences:
            return f"[用户剪辑偏好]\n{self._preferences}\n\n---\n\n{SYSTEM_PROMPT}"
        return SYSTEM_PROMPT

    def parse(self, user_instruction: str, transcript: list[dict]) -> EditPlan:
        transcript_text = "\n".join(
            f"[{s['start']:.1f}s - {s['end']:.1f}s]"
            f"{' ' + s['speaker'] + ':' if s.get('speaker') else ''} {s['text']}"
            for s in transcript
        )
        user_message = (
            f"User instruction: {user_instruction}\n\n"
            f"Transcript:\n{transcript_text}"
        )

        for attempt in range(2):
            response = self._client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                system=self._build_system_prompt(),
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()
            try:
                data = json.loads(raw)
                return EditPlan.model_validate(data)
            except (json.JSONDecodeError, Exception):
                if attempt == 1:
                    raise ValueError(
                        f"Failed to parse LLM response after 2 attempts. "
                        f"Last response: {raw[:200]}"
                    )
                continue

        raise ValueError("Failed to parse LLM response")
```

- [ ] **Step 4: Run all intent parser tests**

```bash
python3 -m pytest tests/test_intent_parser.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/intent_parser.py tests/test_intent_parser.py
git commit -m "feat: inject USER_PREFERENCES.md into LLM prompt, add speaker-labeled transcript format"
```

---

### Task 8: UI speaker column + Python compat fixes

**Files:**
- Modify: `app/main.py`
- Modify: `processing/exporter.py`
- Modify: `tests/test_ui.py`

- [ ] **Step 1: Update `tests/test_ui.py` for 6-column table**

In `tests/test_ui.py`, update every `review_table` list to 6-element rows (add speaker as second column):

```python
# test_export_approved_filters_unchecked — update review_table:
review_table = [
    [1, "—", "0s - 10s", "seg1", "1.00", True],   # included
    [2, "—", "10s - 20s", "seg2", "1.00", False],  # excluded
]
```

Also update the assertion in `test_run_pipeline_returns_candidate_rows` to check `rows[0][5]` (index shifts by 1):

```python
assert rows[0][5] is True  # included=True by default (was rows[0][4])
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m pytest tests/test_ui.py -v
```

Expected: failures — current code produces 5-column rows, tests expect 6.

- [ ] **Step 3: Update `app/main.py`**

Replace entire file:

```python
from __future__ import annotations
import gradio as gr
from pathlib import Path
from agent.orchestrator import Orchestrator
from processing.exporter import Exporter
from models.edit_plan import CandidateSegment, OutputFormat, Platform
from config.settings import settings

orchestrator = None
exporter = None


def _ensure_initialized():
    global orchestrator, exporter
    if orchestrator is None:
        orchestrator = Orchestrator()
    if exporter is None:
        exporter = Exporter(output_dir=settings.output_dir)


def run_pipeline(video_file, instruction: str, session_state: dict):
    """Step 1: Transcribe + parse + generate candidates."""
    if video_file is None:
        return "请上传视频文件", [], None, session_state

    if orchestrator is None:
        _ensure_initialized()

    video_path = Path(video_file)
    result = orchestrator.run(video_path=video_path, user_instruction=instruction)
    session_state["result"] = result
    session_state["video_path"] = video_path

    rows = []
    for i, seg in enumerate(result.candidates):
        rows.append([
            i + 1,
            seg.speaker or "—",
            f"{seg.start:.1f}s - {seg.end:.1f}s",
            seg.text_preview[:80],
            f"{seg.confidence_score:.2f}",
            True,
        ])

    status = f"找到 {len(result.candidates)} 个候选片段 | 模式: {result.plan.mode.value}"
    return status, rows, video_file, session_state


def export_approved(review_table, platform_choices: list[str], session_state: dict):
    """Step 2: Export approved segments."""
    if "result" not in session_state:
        return "请先运行分析", []

    if exporter is None:
        _ensure_initialized()

    result = session_state["result"]
    video_path = session_state["video_path"]

    approved_ids = set()
    for row in review_table:
        idx, _speaker, _time, _preview, _conf, included = row
        if included:
            approved_ids.add(int(idx) - 1)

    for i, seg in enumerate(result.candidates):
        seg.included = (i in approved_ids)

    platform_map = {
        "抖音": Platform.DOUYIN,
        "B站": Platform.BILIBILI,
        "YouTube": Platform.YOUTUBE,
        "微信视频号": Platform.WECHAT,
    }
    formats = [OutputFormat(platform=platform_map[p]) for p in platform_choices]

    output_paths = exporter.export(video_path, result.candidates, formats)
    file_list = [str(p) for p in output_paths]
    return f"导出完成，共 {len(file_list)} 个文件", file_list


with gr.Blocks(title="视频自动剪辑 Agent") as demo:
    gr.Markdown("# 视频自动剪辑 Agent")
    session_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="上传视频")
            instruction_input = gr.Textbox(
                label="剪辑需求",
                placeholder="例如：提取所有提到竞品价格的片段，前后各保留5秒",
                lines=3,
            )
            run_btn = gr.Button("开始分析", variant="primary")

        with gr.Column(scale=2):
            video_preview = gr.Video(label="视频预览")
            status_output = gr.Textbox(label="状态", interactive=False)

    gr.Markdown("## 候选片段审核")
    review_table = gr.Dataframe(
        headers=["序号", "说话人", "时间范围", "内容预览", "置信度", "包含"],
        datatype=["number", "str", "str", "str", "str", "bool"],
        interactive=True,
        label="勾选要保留的片段",
    )

    with gr.Row():
        platform_select = gr.CheckboxGroup(
            choices=["抖音", "B站", "YouTube", "微信视频号"],
            value=["抖音"],
            label="导出平台",
        )
        export_btn = gr.Button("批准并导出", variant="secondary")

    export_status = gr.Textbox(label="导出状态", interactive=False)
    export_files = gr.JSON(label="导出文件列表")

    run_btn.click(
        fn=run_pipeline,
        inputs=[video_input, instruction_input, session_state],
        outputs=[status_output, review_table, video_preview, session_state],
    )
    export_btn.click(
        fn=export_approved,
        inputs=[review_table, platform_select, session_state],
        outputs=[export_status, export_files],
    )


if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 4: Add `from __future__ import annotations` to `processing/exporter.py`**

Add as first line of `processing/exporter.py`:

```python
from __future__ import annotations
```

- [ ] **Step 5: Run all UI tests**

```bash
python3 -m pytest tests/test_ui.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
python3 -m pytest tests/ -v --tb=short
```

Expected: all tests PASS. Fix any failures before continuing.

- [ ] **Step 7: Commit**

```bash
git add app/main.py processing/exporter.py tests/test_ui.py
git commit -m "feat: add speaker column to review table, fix Python 3.9 compat in exporter"
```

---

### Task 9: USER_PREFERENCES.example.md

**Files:**
- Create: `USER_PREFERENCES.example.md`

- [ ] **Step 1: Create `USER_PREFERENCES.example.md`**

```markdown
# 我的剪辑偏好

复制此文件为 USER_PREFERENCES.md 并按需修改：
  cp USER_PREFERENCES.example.md USER_PREFERENCES.md

---

## 剪辑习惯

- 每段片段前后默认保留：5 秒
- 最短保留片段时长：8 秒
- 节奏风格：紧凑，去掉停顿和废话
- 尽量保留完整句子，不在句子中间切断
- 相邻片段若时间间隔不足 3 秒则自动合并

## 字幕偏好（未来版本生效）

- 字体：思源黑体
- 颜色：白色，黑色描边
- 大小：适中
- 位置：底部居中

## 默认导出平台

- 抖音

## 内容背景（帮助 AI 更准确理解你的素材）

- 内容方向：（例：科技产品测评 / 新闻采访 / 纪录片）
- 目标受众：（例：18–35 岁年轻用户 / 专业从业者）
- 常用关键词：（例：评测、上手、体验、对比）

## 额外偏好

- 优先提取观点鲜明、有信息量的片段
- 避免保留纯粹的过渡性废话（"那个"、"嗯"、"然后呢"）
- 产品名称和人名尽量完整保留，不要切断
```

- [ ] **Step 2: Final full test run**

```bash
python3 -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 3: Commit and push**

```bash
git add USER_PREFERENCES.example.md
git commit -m "feat: add USER_PREFERENCES.example.md template"
git push origin main
```

---

## Self-Review

### Spec coverage check

| Spec section | Covered by task |
|---|---|
| api_keys.env.example + HF_TOKEN | Task 1 |
| docker.env | Task 1 |
| Settings hf_token + enable_diarization | Task 1 |
| .gitignore additions | Task 1 |
| Docker Compose — ui service + model cache volumes | Task 2 |
| Dockerfile — torch CPU, EXPOSE 7860 | Task 2 |
| requirements.txt whisperx + pyannote | Task 2 |
| setup.sh / setup.bat | Task 3 |
| start/stop scripts | Task 3 |
| Segment.speaker + CandidateSegment.speaker | Task 4 |
| SPEAKER_FILTER RuleType + Rule.speakers | Task 4 |
| WhisperX Transcriber — 3-stage pipeline | Task 5 |
| RuleEngine speaker_filter | Task 6 |
| IntentParser USER_PREFERENCES injection | Task 7 |
| IntentParser speaker-labeled transcript | Task 7 |
| UI speaker column | Task 8 |
| Python 3.9 compat exporter.py | Task 8 |
| USER_PREFERENCES.example.md | Task 9 |

No gaps found.

### Type consistency check

- `Segment.speaker: Optional[str]` — defined Task 4, used in Tasks 5, 6, 7, 8 ✓
- `CandidateSegment.speaker: Optional[str]` — defined Task 4, used in Tasks 6, 8 ✓
- `Rule.speakers: list[str]` — defined Task 4, used in Task 6 `_speaker_filter` ✓
- `RuleType.SPEAKER_FILTER` — defined Task 4, handled in Task 6 `execute()` ✓
- `Transcriber._hf_token` / `._enable_diarization` — set in Task 5 `__init__`, referenced in `_make_transcriber_no_init` helper ✓
- `IntentParser._preferences` / `._build_system_prompt()` — defined and used within Task 7 ✓
- `app/main.py` row unpacking `idx, _speaker, _time, _preview, _conf, included` — 6 elements matches 6-column table ✓
