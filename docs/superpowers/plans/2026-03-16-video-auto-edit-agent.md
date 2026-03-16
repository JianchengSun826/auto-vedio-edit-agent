# Video Auto-Edit Agent Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an AI-powered video auto-editing agent that accepts natural language instructions, transcribes video with faster-whisper, parses intent with Claude Haiku, generates candidate edit plans, presents them for human review in a Gradio UI, and exports to multiple social media platforms via FFmpeg.

**Architecture:** Local-first pipeline — faster-whisper runs locally for zero STT cost, Claude Haiku parses user intent into a structured EditPlan, FFmpeg executes all video operations. A storage abstraction layer (StorageBackend ABC) keeps cloud storage (S3) addable without touching any other module. Celery + Valkey handles async processing of long-running transcription and export tasks.

**Tech Stack:** Python 3.11+, faster-whisper, anthropic SDK, FFmpeg (subprocess), Gradio 5.x, Celery 5.x, Valkey (Redis-compatible), Pydantic v2, pytest, docker-compose

---

## Chunk 1: Project Setup, Config, and Data Models

### Task 1: Initialize project structure and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `config/settings.py`
- Create: `config/__init__.py`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create requirements.txt**

```
faster-whisper>=1.0.0,<2.0.0
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

- [ ] **Step 2: Create config/settings.py**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str

    # Whisper
    whisper_model: Literal["tiny", "base", "small", "medium", "large-v3"] = "medium"
    whisper_device: Literal["cpu", "cuda", "auto"] = "auto"

    # Storage
    storage_backend: Literal["local"] = "local"
    local_storage_root: str = "./data"
    temp_dir: str = "./tmp"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # Export
    output_dir: str = "./output"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
```

- [ ] **Step 3: Create .env.example**

```
ANTHROPIC_API_KEY=your_key_here
WHISPER_MODEL=medium
WHISPER_DEVICE=auto
STORAGE_BACKEND=local
LOCAL_STORAGE_ROOT=./data
TEMP_DIR=./tmp
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
OUTPUT_DIR=./output
```

- [ ] **Step 4: Create .gitignore**

```
.env
__pycache__/
*.pyc
.pytest_cache/
tmp/
output/
data/
*.egg-info/
dist/
.venv/
```

- [ ] **Step 5: Create required __init__.py files and directories**

```bash
mkdir -p app agent processing storage models tasks config tests
touch app/__init__.py agent/__init__.py processing/__init__.py
touch storage/__init__.py models/__init__.py tasks/__init__.py
touch config/__init__.py tests/__init__.py
```

- [ ] **Step 6: Install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: All packages install without errors.

- [ ] **Step 7: Create tests/conftest.py** (prevents Settings ValidationError across all tests)

```python
# tests/conftest.py
import os
import pytest

@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Ensure required env vars are set for all tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
```

- [ ] **Step 8: Commit**

```bash
git add requirements.txt config/ .env.example .gitignore app/__init__.py agent/__init__.py processing/__init__.py storage/__init__.py models/__init__.py tasks/__init__.py tests/__init__.py tests/conftest.py
git commit -m "feat: initialize project structure and config"
```

---

### Task 2: Define Pydantic data models

**Files:**
- Create: `models/edit_plan.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_models.py
import pytest
from models.edit_plan import (
    EditMode, RuleType, Platform, Rule, OutputFormat,
    EditPlan, Segment, CandidateSegment
)


def test_edit_plan_valid():
    plan = EditPlan(
        mode=EditMode.HIGHLIGHT_EXTRACTION,
        rules=[Rule(type=RuleType.KEYWORD_MATCH, keywords=["竞品"], padding_before_sec=3, padding_after_sec=5)],
        output_formats=[OutputFormat(platform=Platform.DOUYIN)],
    )
    assert plan.mode == EditMode.HIGHLIGHT_EXTRACTION
    assert len(plan.rules) == 1
    assert plan.rules[0].keywords == ["竞品"]


def test_segment_ordering():
    s1 = Segment(start=10.0, end=20.0, text="hello")
    s2 = Segment(start=5.0, end=8.0, text="world")
    assert sorted([s1, s2], key=lambda s: s.start)[0] == s2


def test_candidate_segment_defaults():
    seg = CandidateSegment(id="1", start=0.0, end=5.0, text_preview="test")
    assert seg.confidence_score == 1.0
    assert seg.included is True


def test_output_format_douyin_defaults():
    fmt = OutputFormat(platform=Platform.DOUYIN)
    assert fmt.ratio == "9:16"
    assert fmt.max_duration_sec == 60
    assert fmt.resolution == "1080p"


def test_output_format_youtube_defaults():
    fmt = OutputFormat(platform=Platform.YOUTUBE)
    assert fmt.ratio == "16:9"
    assert fmt.max_duration_sec is None
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'models.edit_plan'`

- [ ] **Step 3: Implement models/edit_plan.py**

```python
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class EditMode(str, Enum):
    HIGHLIGHT_EXTRACTION = "highlight_extraction"
    MATERIAL_ASSEMBLY = "material_assembly"
    SOCIAL_MEDIA = "social_media"


class RuleType(str, Enum):
    KEYWORD_MATCH = "keyword_match"
    TIME_RANGE = "time_range"
    SILENCE_CUT = "silence_cut"
    MIN_DURATION = "min_duration"


class Platform(str, Enum):
    DOUYIN = "douyin"
    BILIBILI = "bilibili"
    YOUTUBE = "youtube"
    WECHAT = "wechat"


# Platform defaults
_PLATFORM_DEFAULTS = {
    Platform.DOUYIN:   {"ratio": "9:16",  "max_duration_sec": 60,   "resolution": "1080p"},
    Platform.WECHAT:   {"ratio": "9:16",  "max_duration_sec": 600,  "resolution": "1080p"},
    Platform.BILIBILI: {"ratio": "16:9",  "max_duration_sec": None, "resolution": "1080p"},
    Platform.YOUTUBE:  {"ratio": "16:9",  "max_duration_sec": None, "resolution": "1080p"},
}


class Rule(BaseModel):
    type: RuleType
    keywords: list[str] = Field(default_factory=list)
    padding_before_sec: float = 3.0
    padding_after_sec: float = 5.0
    min_duration_sec: float = 5.0
    start_sec: Optional[float] = None   # for time_range
    end_sec: Optional[float] = None     # for time_range


class OutputFormat(BaseModel):
    platform: Platform
    ratio: str = "16:9"
    max_duration_sec: Optional[int] = None
    resolution: str = "1080p"

    @classmethod
    def model_validate(cls, obj, **kwargs):
        # Apply platform defaults BEFORE validation so explicit user values win
        if isinstance(obj, dict) and "platform" in obj:
            try:
                platform = Platform(obj["platform"])
                defaults = _PLATFORM_DEFAULTS.get(platform, {})
                merged = {**defaults, **obj}  # user values override defaults
                return super().model_validate(merged, **kwargs)
            except ValueError:
                pass
        return super().model_validate(obj, **kwargs)

    def __init__(self, **data):
        platform = data.get("platform")
        if platform is not None:
            try:
                p = Platform(platform)
                defaults = _PLATFORM_DEFAULTS.get(p, {})
                for k, v in defaults.items():
                    data.setdefault(k, v)  # only set if not explicitly provided
            except ValueError:
                pass
        super().__init__(**data)


class EditPlan(BaseModel):
    mode: EditMode
    rules: list[Rule]
    output_formats: list[OutputFormat]
    segment_count_hint: int = 3


class Segment(BaseModel):
    """Raw transcript segment from faster-whisper."""
    start: float
    end: float
    text: str


class CandidateSegment(BaseModel):
    """A segment proposed for inclusion in the final edit."""
    id: str
    start: float
    end: float
    text_preview: str
    confidence_score: float = 1.0
    included: bool = True
    source_file: Optional[str] = None   # for material assembly mode
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_models.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add models/edit_plan.py tests/test_models.py
git commit -m "feat: add Pydantic data models for EditPlan and segments"
```

---

### Task 3: Implement storage abstraction layer

**Files:**
- Create: `storage/base.py`
- Create: `storage/local.py`
- Create: `storage/s3.py` (stub only)
- Create: `storage/factory.py`
- Create: `tests/test_storage.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_storage.py
import pytest
from pathlib import Path
from storage.local import LocalStorage
from storage.factory import get_storage_backend


def test_local_storage_read_existing_file(tmp_path):
    # Create a test file
    test_file = tmp_path / "video.mp4"
    test_file.write_bytes(b"fake video data")

    storage = LocalStorage(root=str(tmp_path))
    result = storage.read(str(test_file))

    assert result == test_file
    assert result.exists()


def test_local_storage_write(tmp_path):
    storage = LocalStorage(root=str(tmp_path))
    src = tmp_path / "source.mp4"
    src.write_bytes(b"data")

    storage.write(src, "output/result.mp4")

    assert (tmp_path / "output" / "result.mp4").exists()


def test_local_storage_list(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.mp4").write_bytes(b"")
    (tmp_path / "c.txt").write_bytes(b"")

    storage = LocalStorage(root=str(tmp_path))
    results = storage.list("")

    assert len(results) == 3


def test_local_storage_read_missing_file(tmp_path):
    storage = LocalStorage(root=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        storage.read("/nonexistent/path/video.mp4")


def test_factory_returns_local(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    backend = get_storage_backend()
    assert isinstance(backend, LocalStorage)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_storage.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement storage/base.py**

```python
from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    @abstractmethod
    def read(self, path: str) -> Path:
        """Return a local Path to the file (downloads if remote)."""

    @abstractmethod
    def write(self, local_path: Path, dest: str) -> None:
        """Write local_path to dest (uploads if remote)."""

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """List files under prefix."""
```

- [ ] **Step 4: Implement storage/local.py**

```python
import shutil
from pathlib import Path
from storage.base import StorageBackend


class LocalStorage(StorageBackend):
    def __init__(self, root: str = "./data"):
        self.root = Path(root)

    def read(self, path: str) -> Path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return p

    def write(self, local_path: Path, dest: str) -> None:
        dest_path = self.root / dest
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest_path)

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix if prefix else self.root
        if not base.exists():
            return []
        return [str(p) for p in base.rglob("*") if p.is_file()]
```

- [ ] **Step 5: Implement storage/s3.py (stub)**

```python
from pathlib import Path
from storage.base import StorageBackend


class S3Storage(StorageBackend):
    """
    Stub — not implemented in v1.
    To implement: pip install boto3, configure AWS credentials,
    implement download-to-temp on read(), upload on write().
    Change STORAGE_BACKEND=s3 in settings to activate.
    """

    def __init__(self, bucket: str, region: str = "us-east-1"):
        raise NotImplementedError(
            "S3Storage is not implemented in v1. "
            "Set STORAGE_BACKEND=local in your .env file."
        )

    def read(self, path: str) -> Path:
        raise NotImplementedError

    def write(self, local_path: Path, dest: str) -> None:
        raise NotImplementedError

    def list(self, prefix: str) -> list[str]:
        raise NotImplementedError
```

- [ ] **Step 6: Implement storage/factory.py**

```python
from storage.base import StorageBackend
from storage.local import LocalStorage
from config.settings import settings


def get_storage_backend() -> StorageBackend:
    if settings.storage_backend == "local":
        return LocalStorage(root=settings.local_storage_root)
    raise ValueError(
        f"Unknown storage backend: {settings.storage_backend}. "
        "Only 'local' is supported in v1."
    )
```

- [ ] **Step 7: Run tests — verify they pass**

```bash
pytest tests/test_storage.py -v
```

Expected: 5 tests PASS

- [ ] **Step 8: Commit**

```bash
git add storage/ tests/test_storage.py
git commit -m "feat: add storage abstraction layer with LocalStorage"
```

---

## Chunk 2: STT Transcription and FFmpeg Utilities

### Task 4: Implement STT transcription module

**Files:**
- Create: `processing/transcriber.py`
- Create: `tests/test_transcriber.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_transcriber.py
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from processing.transcriber import Transcriber
from models.edit_plan import Segment


def make_mock_segment(start, end, text):
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    return seg


@patch("processing.transcriber.WhisperModel")
def test_transcribe_returns_segments(mock_model_cls, tmp_path):
    # Arrange
    mock_model = MagicMock()
    mock_model_cls.return_value = mock_model
    mock_segments = [
        make_mock_segment(0.0, 3.5, " Hello world"),
        make_mock_segment(3.5, 7.0, " This is a test"),
    ]
    mock_model.transcribe.return_value = (iter(mock_segments), MagicMock())

    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"fake")

    transcriber = Transcriber(model_size="tiny")
    result = transcriber.transcribe(fake_video)

    assert len(result) == 2
    assert isinstance(result[0], Segment)
    assert result[0].text == "Hello world"  # leading space stripped
    assert result[0].start == 0.0
    assert result[1].end == 7.0


@patch("processing.transcriber.WhisperModel")
def test_transcribe_empty_video_returns_empty(mock_model_cls, tmp_path):
    mock_model = MagicMock()
    mock_model_cls.return_value = mock_model
    mock_model.transcribe.return_value = (iter([]), MagicMock())

    fake_video = tmp_path / "silent.mp4"
    fake_video.write_bytes(b"fake")

    transcriber = Transcriber(model_size="tiny")
    result = transcriber.transcribe(fake_video)

    assert result == []


def test_transcribe_missing_file_raises():
    transcriber = Transcriber.__new__(Transcriber)  # skip __init__
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe(Path("/nonexistent/video.mp4"))
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_transcriber.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement processing/transcriber.py**

```python
from pathlib import Path
from faster_whisper import WhisperModel
from models.edit_plan import Segment
from config.settings import settings

# Chunk size and threshold for splitting long videos
CHUNK_DURATION = 1800   # 30 minutes per chunk
CHUNK_THRESHOLD = 7200  # only chunk videos longer than 2 hours


class Transcriber:
    def __init__(self, model_size: str | None = None, device: str | None = None):
        size = model_size or settings.whisper_model
        dev = device or settings.whisper_device
        self._model = WhisperModel(size, device=dev, compute_type="int8")

    def transcribe(self, video_path: Path) -> list[Segment]:
        """Transcribe a video file. Returns list of Segment objects."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        duration = self._get_duration(video_path)
        if duration and duration > CHUNK_THRESHOLD:
            return self._transcribe_chunked(video_path, duration)

        return self._transcribe_single(video_path)

    def _transcribe_single(self, video_path: Path) -> list[Segment]:
        segments_iter, _ = self._model.transcribe(
            str(video_path),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        return [
            Segment(start=seg.start, end=seg.end, text=seg.text.strip())
            for seg in segments_iter
            if seg.text.strip()
        ]

    def _transcribe_chunked(self, video_path: Path, duration: float) -> list[Segment]:
        """Split video into 30-min chunks, transcribe each, merge with offset."""
        import tempfile, subprocess
        all_segments = []
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
                segments = self._transcribe_single(chunk_path)
                # Offset timestamps back to original video position
                for seg in segments:
                    all_segments.append(Segment(
                        start=seg.start + cursor,
                        end=seg.end + cursor,
                        text=seg.text,
                    ))
                cursor = chunk_end
                chunk_index += 1
        return all_segments

    def _get_duration(self, video_path: Path) -> float | None:
        import subprocess, re
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

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_transcriber.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add processing/transcriber.py tests/test_transcriber.py
git commit -m "feat: add STT transcription module with faster-whisper"
```

---

### Task 5: Implement FFmpeg utilities

**Files:**
- Create: `processing/ffmpeg_utils.py`
- Create: `tests/test_ffmpeg_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ffmpeg_utils.py
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from processing.ffmpeg_utils import (
    cut_segment, concat_segments, detect_silence, get_video_duration
)


@patch("processing.ffmpeg_utils.subprocess.run")
def test_cut_segment_calls_ffmpeg(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0, stderr="")
    src = tmp_path / "input.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"

    cut_segment(src, out, start=10.0, end=30.0)

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    # -ss must appear before -i (fast input seek)
    assert cmd.index("-ss") < cmd.index("-i")
    assert "10.0" in cmd
    # -t (duration) must appear after -i
    assert "-t" in cmd
    assert "20.0" in cmd   # duration = 30.0 - 10.0
    assert "-c" in cmd
    assert "copy" in cmd


@patch("processing.ffmpeg_utils.subprocess.run")
def test_cut_segment_raises_on_ffmpeg_error(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=1, stderr="Error: codec not found")
    src = tmp_path / "input.mp4"
    src.write_bytes(b"fake")

    with pytest.raises(RuntimeError, match="FFmpeg error"):
        cut_segment(src, tmp_path / "out.mp4", start=0.0, end=5.0)


@patch("processing.ffmpeg_utils.subprocess.run")
def test_detect_silence_parses_output(mock_run, tmp_path):
    stderr_output = (
        "[silencedetect] silence_start: 2.5\n"
        "[silencedetect] silence_end: 5.0 | silence_duration: 2.5\n"
        "[silencedetect] silence_start: 10.0\n"
        "[silencedetect] silence_end: 12.0 | silence_duration: 2.0\n"
    )
    mock_run.return_value = MagicMock(returncode=0, stderr=stderr_output)
    src = tmp_path / "video.mp4"
    src.write_bytes(b"fake")

    silences = detect_silence(src)

    assert len(silences) == 2
    assert silences[0] == (2.5, 5.0)
    assert silences[1] == (10.0, 12.0)


@patch("processing.ffmpeg_utils.subprocess.run")
def test_get_video_duration(mock_run, tmp_path):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="duration=125.4\n",
        stderr=""
    )
    src = tmp_path / "video.mp4"
    src.write_bytes(b"fake")

    duration = get_video_duration(src)
    assert duration == pytest.approx(125.4)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_ffmpeg_utils.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement processing/ffmpeg_utils.py**

```python
import re
import subprocess
from pathlib import Path


def _run(cmd: list[str], capture_stderr: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{result.stderr}")
    return result


def cut_segment(src: Path, dest: Path, start: float, end: float) -> None:
    """Cut a segment from src using stream copy (lossless).
    -ss before -i for fast seek; -to after -i is relative to the seek point.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    _run([
        "ffmpeg", "-y",
        "-ss", str(start),   # fast input seek
        "-i", str(src),
        "-t", str(duration), # duration (not absolute end) after -i
        "-c", "copy",
        str(dest),
    ])


def concat_segments(segment_files: list[Path], dest: Path) -> None:
    """Concatenate pre-cut segment files using FFmpeg concat demuxer."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    list_file = dest.parent / "concat_list.txt"
    list_file.write_text(
        "\n".join(f"file '{f.resolve()}'" for f in segment_files)
    )
    _run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(dest),
    ])
    list_file.unlink()


def detect_silence(src: Path, noise_db: float = -40, min_duration: float = 1.0) -> list[tuple[float, float]]:
    """Return list of (start, end) silence intervals in seconds."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(src),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg silencedetect error:\n{result.stderr}")
    starts = re.findall(r"silence_start: ([\d.]+)", result.stderr)
    ends = re.findall(r"silence_end: ([\d.]+)", result.stderr)
    return [(float(s), float(e)) for s, e in zip(starts, ends)]


def transcode_for_platform(src: Path, dest: Path, width: int, height: int) -> None:
    """Re-encode video with center crop to target aspect ratio and resolution."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # scale2ref + crop to center-crop to target ratio, then scale to target resolution
    vf = (
        f"scale='if(gt(a,{width}/{height}),{height}*a,-2)':'if(gt(a,{width}/{height}),-2,{width}/a)',"
        f"crop={width}:{height},"
        f"scale={width}:{height}"
    )
    _run([
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        str(dest),
    ])


def get_video_duration(src: Path) -> float:
    """Return video duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1",
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    match = re.search(r"duration=([\d.]+)", result.stdout)
    if not match:
        raise ValueError(f"Could not parse duration from: {result.stdout}")
    return float(match.group(1))
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_ffmpeg_utils.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add processing/ffmpeg_utils.py tests/test_ffmpeg_utils.py
git commit -m "feat: add FFmpeg utilities for cut, concat, silence detection"
```

---

## Chunk 3: LLM Intent Parser and Rule Engine

### Task 6: Implement LLM intent parser

**Files:**
- Create: `agent/intent_parser.py`
- Create: `tests/test_intent_parser.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_intent_parser.py
import pytest
from unittest.mock import MagicMock, patch
from models.edit_plan import EditPlan, EditMode, RuleType, Platform
from agent.intent_parser import IntentParser


SAMPLE_TRANSCRIPT = [
    {"start": 0.0, "end": 5.0, "text": "今天我们聊聊竞品的价格策略"},
    {"start": 5.0, "end": 10.0, "text": "我们的产品比竞品便宜30%"},
]

SAMPLE_LLM_RESPONSE = '''{
  "mode": "highlight_extraction",
  "rules": [
    {
      "type": "keyword_match",
      "keywords": ["竞品", "价格"],
      "padding_before_sec": 3,
      "padding_after_sec": 5,
      "min_duration_sec": 5
    }
  ],
  "output_formats": [
    {
      "platform": "douyin",
      "ratio": "9:16",
      "max_duration_sec": 60,
      "resolution": "1080p"
    }
  ],
  "segment_count_hint": 3
}'''


@patch("agent.intent_parser.anthropic.Anthropic")
def test_parse_returns_edit_plan(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=SAMPLE_LLM_RESPONSE)]
    )

    parser = IntentParser()
    plan = parser.parse(
        user_instruction="提取所有提到竞品价格的片段",
        transcript=SAMPLE_TRANSCRIPT,
    )

    assert isinstance(plan, EditPlan)
    assert plan.mode == EditMode.HIGHLIGHT_EXTRACTION
    assert plan.rules[0].type == RuleType.KEYWORD_MATCH
    assert "竞品" in plan.rules[0].keywords
    assert plan.output_formats[0].platform == Platform.DOUYIN


@patch("agent.intent_parser.anthropic.Anthropic")
def test_parse_retries_on_invalid_json(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.side_effect = [
        MagicMock(content=[MagicMock(text="not json at all")]),
        MagicMock(content=[MagicMock(text=SAMPLE_LLM_RESPONSE)]),
    ]

    parser = IntentParser()
    plan = parser.parse(
        user_instruction="提取竞品片段",
        transcript=SAMPLE_TRANSCRIPT,
    )

    assert isinstance(plan, EditPlan)
    assert mock_client.messages.create.call_count == 2


@patch("agent.intent_parser.anthropic.Anthropic")
def test_parse_raises_after_two_failures(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="invalid json")]
    )

    parser = IntentParser()
    with pytest.raises(ValueError, match="Failed to parse"):
        parser.parse(user_instruction="test", transcript=[])
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_intent_parser.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement agent/intent_parser.py**

```python
import json
import anthropic
from models.edit_plan import EditPlan
from config.settings import settings

SYSTEM_PROMPT = """You are a video editing assistant. Given a user's editing instruction and a video transcript,
output a JSON EditPlan that describes how to edit the video.

Output ONLY valid JSON matching this exact schema:
{
  "mode": "highlight_extraction" | "material_assembly" | "social_media",
  "rules": [
    {
      "type": "keyword_match" | "time_range" | "silence_cut" | "min_duration",
      "keywords": [...],          // for keyword_match only
      "padding_before_sec": 3,
      "padding_after_sec": 5,
      "min_duration_sec": 5,
      "start_sec": null,          // for time_range only
      "end_sec": null             // for time_range only
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

    def parse(self, user_instruction: str, transcript: list[dict]) -> EditPlan:
        transcript_text = "\n".join(
            f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}"
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
                system=SYSTEM_PROMPT,
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

        raise ValueError("Failed to parse LLM response")  # unreachable but satisfies type checker
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_intent_parser.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent/intent_parser.py tests/test_intent_parser.py
git commit -m "feat: add LLM intent parser using Claude Haiku"
```

---

### Task 7: Implement rule execution engine

**Files:**
- Create: `agent/rule_engine.py`
- Create: `tests/test_rule_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rule_engine.py
import pytest
from unittest.mock import patch
from pathlib import Path
from models.edit_plan import EditPlan, EditMode, Rule, RuleType, OutputFormat, Platform, Segment, CandidateSegment
from agent.rule_engine import RuleEngine


TRANSCRIPT = [
    Segment(start=0.0, end=5.0, text="今天介绍产品功能"),
    Segment(start=5.0, end=12.0, text="竞品的价格非常高"),
    Segment(start=12.0, end=20.0, text="我们提供更好的性价比"),
    Segment(start=20.0, end=25.0, text="欢迎联系我们了解竞品对比"),
]

PLAN_KEYWORD = EditPlan(
    mode=EditMode.HIGHLIGHT_EXTRACTION,
    rules=[Rule(type=RuleType.KEYWORD_MATCH, keywords=["竞品"], padding_before_sec=1, padding_after_sec=2)],
    output_formats=[OutputFormat(platform=Platform.DOUYIN)],
)


def test_keyword_match_finds_segments():
    engine = RuleEngine()
    candidates = engine.execute(PLAN_KEYWORD, TRANSCRIPT, video_path=None)

    assert len(candidates) >= 2
    texts = [c.text_preview for c in candidates]
    assert any("竞品" in t for t in texts)


def test_keyword_match_applies_padding():
    engine = RuleEngine()
    candidates = engine.execute(PLAN_KEYWORD, TRANSCRIPT, video_path=None)

    # Segment at 5-12s should get padding: start max(0, 5-1)=4, end min(duration, 12+2)=14
    keyword_seg = next(c for c in candidates if "价格" in c.text_preview or "竞品" in c.text_preview)
    assert keyword_seg.start <= 5.0   # padding applied before
    assert keyword_seg.end >= 12.0    # padding applied after


def test_min_duration_filters_short_segments():
    plan = EditPlan(
        mode=EditMode.HIGHLIGHT_EXTRACTION,
        rules=[
            Rule(type=RuleType.KEYWORD_MATCH, keywords=["竞品"], padding_before_sec=0, padding_after_sec=0),
            Rule(type=RuleType.MIN_DURATION, min_duration_sec=10),
        ],
        output_formats=[OutputFormat(platform=Platform.DOUYIN)],
    )
    engine = RuleEngine()
    candidates = engine.execute(plan, TRANSCRIPT, video_path=None)

    for c in candidates:
        assert (c.end - c.start) >= 10


def test_time_range_extracts_correct_segment():
    plan = EditPlan(
        mode=EditMode.HIGHLIGHT_EXTRACTION,
        rules=[Rule(type=RuleType.TIME_RANGE, start_sec=5.0, end_sec=15.0)],
        output_formats=[OutputFormat(platform=Platform.YOUTUBE)],
    )
    engine = RuleEngine()
    candidates = engine.execute(plan, TRANSCRIPT, video_path=None)

    assert len(candidates) == 1
    assert candidates[0].start == 5.0
    assert candidates[0].end == 15.0
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_rule_engine.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement agent/rule_engine.py**

```python
from pathlib import Path
from models.edit_plan import EditPlan, RuleType, Rule, Segment, CandidateSegment
import uuid


class RuleEngine:
    def execute(
        self,
        plan: EditPlan,
        transcript: list[Segment],
        video_path: Path | None,
        video_duration: float | None = None,
    ) -> list[CandidateSegment]:
        """Execute EditPlan rules against transcript. Returns candidate segments."""
        candidates: list[CandidateSegment] = []

        for rule in plan.rules:
            if rule.type == RuleType.KEYWORD_MATCH:
                candidates.extend(self._keyword_match(rule, transcript, video_duration))
            elif rule.type == RuleType.TIME_RANGE:
                candidates.extend(self._time_range(rule))
            elif rule.type == RuleType.MIN_DURATION:
                candidates = self._filter_min_duration(rule, candidates)
            elif rule.type == RuleType.SILENCE_CUT:
                if video_path is not None:
                    candidates = self._apply_silence_cut(rule, candidates, video_path)

        # Deduplicate overlapping segments
        candidates = self._merge_overlapping(candidates)
        return candidates

    def _keyword_match(
        self, rule: Rule, transcript: list[Segment], video_duration: float | None
    ) -> list[CandidateSegment]:
        results = []
        for seg in transcript:
            if any(kw in seg.text for kw in rule.keywords):
                start = max(0.0, seg.start - rule.padding_before_sec)
                end = seg.end + rule.padding_after_sec
                if video_duration:
                    end = min(end, video_duration)
                results.append(CandidateSegment(
                    id=str(uuid.uuid4()),
                    start=start,
                    end=end,
                    text_preview=seg.text,
                    confidence_score=1.0,
                ))
        return results

    def _time_range(self, rule: Rule) -> list[CandidateSegment]:
        if rule.start_sec is None or rule.end_sec is None:
            return []
        return [CandidateSegment(
            id=str(uuid.uuid4()),
            start=rule.start_sec,
            end=rule.end_sec,
            text_preview=f"[{rule.start_sec:.1f}s - {rule.end_sec:.1f}s]",
            confidence_score=1.0,
        )]

    def _apply_silence_cut(
        self, rule: Rule, candidates: list[CandidateSegment], video_path: Path
    ) -> list[CandidateSegment]:
        """Remove silence intervals from candidate segments."""
        from processing.ffmpeg_utils import detect_silence
        try:
            silences = detect_silence(video_path)
        except RuntimeError:
            return candidates  # if detection fails, return unchanged

        result = []
        for seg in candidates:
            # Split segment around silence intervals
            cursor = seg.start
            for s_start, s_end in silences:
                if s_end <= seg.start or s_start >= seg.end:
                    continue  # silence outside segment
                if cursor < s_start:
                    result.append(CandidateSegment(
                        id=str(uuid.uuid4()),
                        start=cursor,
                        end=s_start,
                        text_preview=seg.text_preview,
                        confidence_score=seg.confidence_score,
                    ))
                cursor = max(cursor, s_end)
            if cursor < seg.end:
                result.append(CandidateSegment(
                    id=str(uuid.uuid4()),
                    start=cursor,
                    end=seg.end,
                    text_preview=seg.text_preview,
                    confidence_score=seg.confidence_score,
                ))
        return result if result else candidates

    def _filter_min_duration(self, rule: Rule, candidates: list[CandidateSegment]) -> list[CandidateSegment]:
        return [c for c in candidates if (c.end - c.start) >= rule.min_duration_sec]

    def _merge_overlapping(self, candidates: list[CandidateSegment]) -> list[CandidateSegment]:
        if not candidates:
            return []
        sorted_segs = sorted(candidates, key=lambda c: c.start)
        merged = [sorted_segs[0]]
        for current in sorted_segs[1:]:
            last = merged[-1]
            if current.start <= last.end:
                # Merge: extend end, combine text
                merged[-1] = CandidateSegment(
                    id=last.id,
                    start=last.start,
                    end=max(last.end, current.end),
                    text_preview=f"{last.text_preview} | {current.text_preview}",
                    confidence_score=max(last.confidence_score, current.confidence_score),
                )
            else:
                merged.append(current)
        return merged
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_rule_engine.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent/rule_engine.py tests/test_rule_engine.py
git commit -m "feat: add rule execution engine for EditPlan"
```

---

## Chunk 4: Export Module and Async Tasks

### Task 8: Implement multi-platform export module

**Files:**
- Create: `processing/exporter.py`
- Create: `tests/test_exporter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_exporter.py
import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from models.edit_plan import OutputFormat, Platform, CandidateSegment
from processing.exporter import Exporter, PlatformSpec


def make_candidate(id, start, end):
    return CandidateSegment(id=id, start=start, end=end, text_preview="test")


def test_platform_spec_douyin():
    spec = PlatformSpec.for_platform(Platform.DOUYIN)
    assert spec.ratio == "9:16"
    assert spec.max_duration_sec == 60
    assert spec.width == 1080
    assert spec.height == 1920


def test_platform_spec_youtube():
    spec = PlatformSpec.for_platform(Platform.YOUTUBE)
    assert spec.ratio == "16:9"
    assert spec.max_duration_sec is None
    assert spec.width == 1920
    assert spec.height == 1080


@patch("processing.exporter.transcode_for_platform")
@patch("processing.exporter.cut_segment")
def test_export_single_segment_youtube(mock_cut, mock_transcode, tmp_path):
    src = tmp_path / "input.mp4"
    src.write_bytes(b"fake")
    candidates = [make_candidate("1", 10.0, 40.0)]
    fmt = OutputFormat(platform=Platform.YOUTUBE)

    exporter = Exporter(output_dir=tmp_path)
    results = exporter.export(src, candidates, [fmt])

    assert len(results) == 1
    mock_cut.assert_called_once()
    mock_transcode.assert_not_called()  # YouTube uses stream copy (same ratio assumed)


@patch("processing.exporter.transcode_for_platform")
@patch("processing.exporter.cut_segment")
def test_export_douyin_auto_splits_long_segment(mock_cut, mock_transcode, tmp_path):
    src = tmp_path / "input.mp4"
    src.write_bytes(b"fake")
    # 3-minute segment should split into 3 x 60s clips for Douyin
    candidates = [make_candidate("1", 0.0, 180.0)]
    fmt = OutputFormat(platform=Platform.DOUYIN)

    exporter = Exporter(output_dir=tmp_path)
    results = exporter.export(src, candidates, [fmt])

    assert len(results) == 3
    assert mock_cut.call_count == 3
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_exporter.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement processing/exporter.py**

```python
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from models.edit_plan import OutputFormat, Platform, CandidateSegment
from processing.ffmpeg_utils import cut_segment, transcode_for_platform


@dataclass
class PlatformSpec:
    ratio: str
    width: int
    height: int
    max_duration_sec: Optional[int]
    codec: str = "libx264"

    @classmethod
    def for_platform(cls, platform: Platform) -> "PlatformSpec":
        specs = {
            Platform.DOUYIN:   cls("9:16",  1080, 1920, 60),
            Platform.WECHAT:   cls("9:16",  1080, 1920, 600),
            Platform.BILIBILI: cls("16:9",  1920, 1080, None),
            Platform.YOUTUBE:  cls("16:9",  1920, 1080, None),
        }
        return specs[platform]


class Exporter:
    def __init__(self, output_dir: Path | str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        src: Path,
        candidates: list[CandidateSegment],
        formats: list[OutputFormat],
    ) -> list[Path]:
        """Export approved candidate segments for all requested platforms."""
        results = []
        included = [c for c in candidates if c.included]

        for fmt in formats:
            spec = PlatformSpec.for_platform(fmt.platform)
            needs_transcode = fmt.ratio != "16:9"  # source assumed 16:9; transcode for vertical
            for candidate in included:
                parts = self._split_if_needed(candidate.start, candidate.end, spec)
                for i, (part_start, part_end) in enumerate(parts):
                    suffix = f"_part{i+1}" if len(parts) > 1 else ""
                    out_name = f"{src.stem}_{fmt.platform.value}_{candidate.id}{suffix}.mp4"
                    cut_path = self.output_dir / f"_tmp_{out_name}"
                    out_path = self.output_dir / out_name
                    cut_segment(src, cut_path, start=part_start, end=part_end)
                    if needs_transcode:
                        transcode_for_platform(cut_path, out_path, spec.width, spec.height)
                        cut_path.unlink(missing_ok=True)
                    else:
                        cut_path.rename(out_path)
                    results.append(out_path)

        return results

    def _split_if_needed(
        self, start: float, end: float, spec: PlatformSpec
    ) -> list[tuple[float, float]]:
        """Split segment into chunks if it exceeds platform max duration."""
        if spec.max_duration_sec is None:
            return [(start, end)]

        parts = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + spec.max_duration_sec, end)
            parts.append((cursor, chunk_end))
            cursor = chunk_end
        return parts
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_exporter.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add processing/exporter.py tests/test_exporter.py
git commit -m "feat: add multi-platform export module with auto-split"
```

---

### Task 9: Set up Celery async tasks and docker-compose

**Files:**
- Create: `tasks/celery_app.py`
- Create: `tasks/celery_tasks.py`
- Create: `docker-compose.yml`

- [ ] **Step 1: Create tasks/celery_app.py**

```python
from celery import Celery
from config.settings import settings

app = Celery(
    "video_edit_agent",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_max_tasks_per_child=10,
    task_track_started=True,
)
```

- [ ] **Step 2: Create tasks/celery_tasks.py**

```python
from pathlib import Path
from tasks.celery_app import app
from processing.transcriber import Transcriber
from processing.exporter import Exporter
from models.edit_plan import CandidateSegment, OutputFormat


@app.task(bind=True, name="tasks.transcribe")
def transcribe_video(self, video_path: str) -> list[dict]:
    """Transcribe a video file. Returns list of segment dicts."""
    self.update_state(state="PROGRESS", meta={"status": "Transcribing..."})
    transcriber = Transcriber()
    segments = transcriber.transcribe(Path(video_path))
    return [s.model_dump() for s in segments]


@app.task(bind=True, name="tasks.export")
def export_video(
    self,
    video_path: str,
    candidates: list[dict],
    formats: list[dict],
    output_dir: str,
) -> list[str]:
    """Export approved candidates for all platforms. Returns list of output paths."""
    self.update_state(state="PROGRESS", meta={"status": "Exporting..."})
    exporter = Exporter(output_dir=output_dir)
    candidate_objs = [CandidateSegment(**c) for c in candidates]
    format_objs = [OutputFormat(**f) for f in formats]
    results = exporter.export(Path(video_path), candidate_objs, format_objs)
    return [str(p) for p in results]
```

- [ ] **Step 3: Create docker-compose.yml**

```yaml
services:
  valkey:
    image: valkey/valkey:7
    ports:
      - "6379:6379"
    volumes:
      - valkey_data:/data
    command: valkey-server --appendonly yes

  worker:
    build: .
    command: celery -A tasks.celery_app worker --loglevel=info --concurrency=2
    depends_on:
      - valkey
    environment:
      - CELERY_BROKER_URL=redis://valkey:6379/0
      - CELERY_RESULT_BACKEND=redis://valkey:6379/0
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./output:/app/output
      - ./tmp:/app/tmp

volumes:
  valkey_data:
```

- [ ] **Step 4: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
```

- [ ] **Step 5: Verify docker-compose syntax**

```bash
docker compose config
```

Expected: Valid YAML printed, no errors.

- [ ] **Step 6: Commit**

```bash
git add tasks/ docker-compose.yml Dockerfile
git commit -m "feat: add Celery async tasks and docker-compose with Valkey"
```

---

## Chunk 5: Orchestrator and Gradio UI

### Task 10: Implement agent orchestrator

**Files:**
- Create: `agent/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orchestrator.py
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from models.edit_plan import EditPlan, EditMode, Rule, RuleType, OutputFormat, Platform, Segment
from agent.orchestrator import Orchestrator


FAKE_TRANSCRIPT = [
    Segment(start=0.0, end=5.0, text="竞品价格很高"),
    Segment(start=5.0, end=10.0, text="我们更便宜"),
]

FAKE_PLAN = EditPlan(
    mode=EditMode.HIGHLIGHT_EXTRACTION,
    rules=[Rule(type=RuleType.KEYWORD_MATCH, keywords=["竞品"])],
    output_formats=[OutputFormat(platform=Platform.DOUYIN)],
)


@patch("agent.orchestrator.IntentParser")
@patch("agent.orchestrator.Transcriber")
def test_orchestrator_run_returns_candidates(mock_transcriber_cls, mock_parser_cls, tmp_path):
    # Arrange
    mock_transcriber = MagicMock()
    mock_transcriber_cls.return_value = mock_transcriber
    mock_transcriber.transcribe.return_value = FAKE_TRANSCRIPT

    mock_parser = MagicMock()
    mock_parser_cls.return_value = mock_parser
    mock_parser.parse.return_value = FAKE_PLAN

    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")

    orch = Orchestrator()
    result = orch.run(video_path=video, user_instruction="提取竞品片段")

    assert result.plan is not None
    assert result.candidates is not None
    assert len(result.candidates) > 0
    assert result.transcript == FAKE_TRANSCRIPT


@patch("agent.orchestrator.IntentParser")
@patch("agent.orchestrator.Transcriber")
def test_orchestrator_passes_transcript_to_parser(mock_transcriber_cls, mock_parser_cls, tmp_path):
    mock_transcriber = MagicMock()
    mock_transcriber_cls.return_value = mock_transcriber
    mock_transcriber.transcribe.return_value = FAKE_TRANSCRIPT

    mock_parser = MagicMock()
    mock_parser_cls.return_value = mock_parser
    mock_parser.parse.return_value = FAKE_PLAN

    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")

    orch = Orchestrator()
    orch.run(video_path=video, user_instruction="找竞品")

    call_kwargs = mock_parser.parse.call_args
    assert call_kwargs.kwargs["transcript"] is not None or call_kwargs.args[1] is not None
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_orchestrator.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement agent/orchestrator.py**

```python
from dataclasses import dataclass
from pathlib import Path
from models.edit_plan import EditPlan, Segment, CandidateSegment
from processing.transcriber import Transcriber
from agent.intent_parser import IntentParser
from agent.rule_engine import RuleEngine
from processing.ffmpeg_utils import get_video_duration


@dataclass
class OrchestrationResult:
    transcript: list[Segment]
    plan: EditPlan
    candidates: list[CandidateSegment]


class Orchestrator:
    def __init__(self):
        self._transcriber = Transcriber()
        self._parser = IntentParser()
        self._engine = RuleEngine()

    def run(self, video_path: Path, user_instruction: str) -> OrchestrationResult:
        """Full pipeline: transcribe → parse intent → execute rules → return candidates."""
        # Step 1: Transcribe
        transcript = self._transcriber.transcribe(video_path)

        # Step 2: Get video duration for padding bounds
        try:
            duration = get_video_duration(video_path)
        except Exception:
            duration = None

        # Step 3: Parse intent
        transcript_dicts = [s.model_dump() for s in transcript]
        plan = self._parser.parse(
            user_instruction=user_instruction,
            transcript=transcript_dicts,
        )

        # Step 4: Execute rules
        candidates = self._engine.execute(plan, transcript, video_path, duration)

        return OrchestrationResult(
            transcript=transcript,
            plan=plan,
            candidates=candidates,
        )
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_orchestrator.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add agent orchestrator tying together pipeline"
```

---

### Task 11: Build Gradio review UI

**Files:**
- Create: `app/main.py`
- Create: `tests/test_ui.py`

- [ ] **Step 1: Write failing UI tests**

```python
# tests/test_ui.py
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from models.edit_plan import EditPlan, EditMode, Rule, RuleType, OutputFormat, Platform, Segment, CandidateSegment


def make_result(candidates):
    plan = EditPlan(
        mode=EditMode.HIGHLIGHT_EXTRACTION,
        rules=[Rule(type=RuleType.KEYWORD_MATCH, keywords=["test"])],
        output_formats=[OutputFormat(platform=Platform.YOUTUBE)],
    )
    from agent.orchestrator import OrchestrationResult
    return OrchestrationResult(transcript=[], plan=plan, candidates=candidates)


@patch("app.main.orchestrator")
def test_run_pipeline_returns_candidate_rows(mock_orch, tmp_path):
    from app.main import run_pipeline
    candidates = [
        CandidateSegment(id="1", start=0.0, end=10.0, text_preview="hello world"),
        CandidateSegment(id="2", start=20.0, end=30.0, text_preview="test segment"),
    ]
    mock_orch.run.return_value = make_result(candidates)
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")

    status, rows, _, state = run_pipeline(str(video), "test instruction", {})

    assert "2 个候选片段" in status
    assert len(rows) == 2
    assert rows[0][4] is True  # included=True by default
    assert "result" in state


@patch("app.main.orchestrator")
def test_run_pipeline_no_video_returns_error(mock_orch):
    from app.main import run_pipeline
    status, rows, _, state = run_pipeline(None, "test", {})
    assert "请上传" in status
    assert rows == []
    mock_orch.run.assert_not_called()


@patch("app.main.exporter")
def test_export_approved_filters_unchecked(mock_exporter, tmp_path):
    from app.main import export_approved
    from agent.orchestrator import OrchestrationResult
    candidates = [
        CandidateSegment(id="1", start=0.0, end=10.0, text_preview="seg1"),
        CandidateSegment(id="2", start=10.0, end=20.0, text_preview="seg2"),
    ]
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    state = {
        "result": make_result(candidates),
        "video_path": video,
    }
    mock_exporter.export.return_value = [Path("out.mp4")]

    review_table = [
        [1, "0s - 10s", "seg1", "1.00", True],   # included
        [2, "10s - 20s", "seg2", "1.00", False],  # excluded
    ]
    status, files = export_approved(review_table, ["YouTube"], state)

    assert "导出完成" in status
    call_candidates = mock_exporter.export.call_args[0][1]
    included = [c for c in call_candidates if c.included]
    excluded = [c for c in call_candidates if not c.included]
    assert len(included) == 1
    assert len(excluded) == 1
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_ui.py -v
```

Expected: `ModuleNotFoundError` (app/main.py not yet created)

- [ ] **Step 3: Implement app/main.py**

```python
import gradio as gr
from pathlib import Path
from agent.orchestrator import Orchestrator
from processing.exporter import Exporter
from models.edit_plan import CandidateSegment, OutputFormat, Platform
from config.settings import settings

orchestrator = Orchestrator()
exporter = Exporter(output_dir=settings.output_dir)


def run_pipeline(video_file, instruction: str, session_state: dict):
    """Step 1: Transcribe + parse + generate candidates."""
    if video_file is None:
        return "请上传视频文件", [], None, session_state

    video_path = Path(video_file)
    result = orchestrator.run(video_path=video_path, user_instruction=instruction)
    session_state["result"] = result
    session_state["video_path"] = video_path

    rows = []
    for i, seg in enumerate(result.candidates):
        rows.append([
            i + 1,
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

    result = session_state["result"]
    video_path = session_state["video_path"]

    approved_ids = set()
    for row in review_table:
        idx, _, _, _, included = row
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
    session_state = gr.State({})   # per-session isolation

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
        headers=["序号", "时间范围", "内容预览", "置信度", "包含"],
        datatype=["number", "str", "str", "str", "bool"],
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

- [ ] **Step 4: Run UI tests — verify they pass**

```bash
pytest tests/test_ui.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Run the UI locally to verify it starts**

```bash
python app/main.py
```

Expected: Gradio starts on `http://127.0.0.1:7860` with no import errors.

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_ui.py
git commit -m "feat: add Gradio review UI with pipeline integration"
```

---

### Task 12: Full test suite and README

**Files:**
- Create: `README.md`
- Modify: `tests/` (run all tests)

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests PASS. Fix any failures before proceeding.

- [ ] **Step 2: Create README.md**

```markdown
# 视频自动剪辑 Agent

AI 驱动的视频自动剪辑工具，支持自然语言需求输入，经人工审核后导出多平台视频。

## 快速开始

### 1. 安装依赖

\`\`\`bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
\`\`\`

确保系统已安装 FFmpeg：
\`\`\`bash
# macOS
brew install ffmpeg
# Ubuntu
sudo apt install ffmpeg
\`\`\`

### 2. 配置环境变量

\`\`\`bash
cp .env.example .env
# 编辑 .env，填入 ANTHROPIC_API_KEY
\`\`\`

### 3. 启动 Valkey（Redis）和 Celery worker

\`\`\`bash
docker compose up -d valkey
celery -A tasks.celery_app worker --loglevel=info
\`\`\`

### 4. 启动 UI

\`\`\`bash
python app/main.py
\`\`\`

打开 http://127.0.0.1:7860

## 使用方法

1. 上传本地视频文件
2. 用自然语言描述剪辑需求（如：`提取所有提到竞品的片段，每段保留前后5秒`）
3. 点击"开始分析"，等待转录和 AI 解析
4. 在候选片段列表中勾选要保留的片段
5. 选择导出平台，点击"批准并导出"

## 三种剪辑模式

| 模式 | 示例指令 |
|------|---------|
| 精华提取 | `提取所有提到竞品价格的片段` |
| 素材拼接 | `按脚本拼接：开场30秒用第一个视频，产品演示用第二个视频2-4分钟` |
| 社媒生产 | `把采访剪成3条抖音短视频，每条不超过60秒` |

## 未来扩展

- S3 云存储：实现 `storage/s3.py`，修改 `.env` 中 `STORAGE_BACKEND=s3`
- 视觉 AI：在转录与规则引擎之间插入场景检测模块
```

- [ ] **Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: add README with setup and usage instructions"
git push origin main
```

---

## Summary

| Chunk | Tasks | Key Output |
|-------|-------|-----------|
| 1 | 1-3 | Project setup, Pydantic models, Storage layer |
| 2 | 4-5 | STT transcription, FFmpeg utilities |
| 3 | 6-7 | LLM intent parser, Rule engine |
| 4 | 8-9 | Export module, Celery + Valkey |
| 5 | 10-12 | Orchestrator, Gradio UI, README |

**Run all tests at any time:**
```bash
pytest tests/ -v
```
