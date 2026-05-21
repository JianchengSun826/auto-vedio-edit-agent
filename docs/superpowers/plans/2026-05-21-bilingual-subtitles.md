# 双语字幕并行转录 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对同一视频并行运行两次 Whisper（`language="zh"` 和 `task="translate"`），在候选片段表格中同时展示中文/英文字幕预览，字幕导出同时生成 `_zh.srt` 和 `_en.srt`。

**Architecture:** `Transcriber.transcribe_bilingual()` 用 `ThreadPoolExecutor(max_workers=2)` 并行运行两次转录，按索引合并为含 `text_zh`/`text_en` 的 `Segment` 列表。`CandidateSegment` 同步扩展，`subtitle.py` 增加 `lang` 参数，`main.py` 改用双文件下载。

**Tech Stack:** Python threading (`concurrent.futures`), mlx_whisper (`language`/`task` kwargs via `**decode_options`), pydantic, gradio

---

> **Note:** 以下 8 个测试在修改前就已失败（pre-existing），与本次改动无关，无需修复：
> `test_transcriber.py` 3 个（lazy import 导致 mock path 错误）、`test_pipeline.py` 1 个、`test_ui.py` 4 个。

---

## File Map

| 文件 | 操作 | 职责 |
|------|------|------|
| `models/edit_plan.py` | 修改 | `Segment` 加 `text_zh`/`text_en`；`CandidateSegment` 加 `text_preview_zh`/`text_preview_en` |
| `agent/rule_engine.py` | 修改 | 构造 `CandidateSegment` 时填入新字段 |
| `processing/subtitle.py` | 修改 | `segments_to_srt` 加 `lang` 参数 |
| `processing/transcriber.py` | 修改 | `language`/`task` 沿调用链透传；新增 `transcribe_bilingual()` |
| `app/pipeline.py` | 修改 | `candidates_to_rows` 拆成双列 |
| `app/main.py` | 修改 | 调用 `transcribe_bilingual`；表格双列；导出双 SRT |
| `tests/test_transcriber.py` | 修改 | 新增 `transcribe_bilingual` 测试 |
| `tests/test_subtitle.py` | 新建 | `segments_to_srt` lang 参数测试 |

---

## Task 1: 扩展数据模型

**Files:**
- Modify: `models/edit_plan.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: 写失败测试**

打开 `tests/test_models.py`，在文件末尾添加：

```python
def test_segment_bilingual_fields():
    seg = Segment(start=0.0, end=3.0, text="你好", text_zh="你好", text_en="Hello")
    assert seg.text_zh == "你好"
    assert seg.text_en == "Hello"

def test_segment_bilingual_defaults_none():
    seg = Segment(start=0.0, end=3.0, text="hello")
    assert seg.text_zh is None
    assert seg.text_en is None

def test_candidate_segment_bilingual_fields():
    from models.edit_plan import CandidateSegment
    c = CandidateSegment(
        id="1", start=0.0, end=3.0, text_preview="hello",
        text_preview_zh="你好", text_preview_en="Hello",
    )
    assert c.text_preview_zh == "你好"
    assert c.text_preview_en == "Hello"

def test_candidate_segment_bilingual_defaults_none():
    from models.edit_plan import CandidateSegment
    c = CandidateSegment(id="1", start=0.0, end=3.0, text_preview="hello")
    assert c.text_preview_zh is None
    assert c.text_preview_en is None
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /Users/jianchengsun/media-dev/auto-vedio-edit-agent
source .venv/bin/activate
python -m pytest tests/test_models.py::test_segment_bilingual_fields tests/test_models.py::test_candidate_segment_bilingual_fields -v
```

期望：FAILED（字段不存在）

- [ ] **Step 3: 实现**

打开 `models/edit_plan.py`，将 `Segment` 改为：

```python
class Segment(BaseModel):
    """Raw transcript segment from WhisperX."""
    start: float
    end: float
    text: str
    text_zh: Optional[str] = None
    text_en: Optional[str] = None
    speaker: Optional[str] = None
```

将 `CandidateSegment` 改为：

```python
class CandidateSegment(BaseModel):
    """A segment proposed for inclusion in the final edit."""
    id: str
    start: float
    end: float
    text_preview: str
    text_preview_zh: Optional[str] = None
    text_preview_en: Optional[str] = None
    confidence_score: float = 1.0
    included: bool = True
    source_file: Optional[str] = None
    speaker: Optional[str] = None
```

- [ ] **Step 4: 运行确认通过**

```bash
python -m pytest tests/test_models.py -v
```

期望：全部 PASS（原有测试 + 新增 4 个）

- [ ] **Step 5: 提交**

```bash
git add models/edit_plan.py tests/test_models.py
git commit -m "feat: add text_zh/text_en to Segment and CandidateSegment"
```

---

## Task 2: 更新 RuleEngine 填入双语字段

**Files:**
- Modify: `agent/rule_engine.py`
- Test: `tests/test_rule_engine.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_rule_engine.py` 末尾添加：

```python
TRANSCRIPT_BILINGUAL = [
    Segment(start=0.0, end=5.0, text="竞品分析", text_zh="竞品分析", text_en="Competitor analysis"),
    Segment(start=5.0, end=12.0, text="价格对比", text_zh="价格对比", text_en="Price comparison"),
]

def test_keyword_match_preserves_bilingual_preview():
    engine = RuleEngine()
    plan = EditPlan(
        mode=EditMode.HIGHLIGHT_EXTRACTION,
        rules=[Rule(type=RuleType.KEYWORD_MATCH, keywords=["竞品"], padding_before_sec=0, padding_after_sec=0)],
        output_formats=[],
    )
    candidates = engine.execute(plan, TRANSCRIPT_BILINGUAL, video_path=None)
    assert len(candidates) == 1
    assert candidates[0].text_preview_zh == "竞品分析"
    assert candidates[0].text_preview_en == "Competitor analysis"
```

- [ ] **Step 2: 运行确认失败**

```bash
python -m pytest tests/test_rule_engine.py::test_keyword_match_preserves_bilingual_preview -v
```

期望：FAILED（`text_preview_zh` 为 None）

- [ ] **Step 3: 实现**

打开 `agent/rule_engine.py`，更新 `_keyword_match` 和 `_speaker_filter` 中的 `CandidateSegment` 构造：

`_keyword_match`（第 45-52 行）改为：
```python
results.append(CandidateSegment(
    id=str(uuid.uuid4()),
    start=start,
    end=end,
    text_preview=seg.text,
    text_preview_zh=seg.text_zh,
    text_preview_en=seg.text_en,
    confidence_score=1.0,
    speaker=seg.speaker,
))
```

`_speaker_filter`（第 59-65 行）改为：
```python
results.append(CandidateSegment(
    id=str(uuid.uuid4()),
    start=max(0.0, seg.start - rule.padding_before_sec),
    end=seg.end + rule.padding_after_sec,
    text_preview=seg.text,
    text_preview_zh=seg.text_zh,
    text_preview_en=seg.text_en,
    confidence_score=1.0,
    speaker=seg.speaker,
))
```

- [ ] **Step 4: 运行确认通过**

```bash
python -m pytest tests/test_rule_engine.py -v
```

期望：全部 PASS

- [ ] **Step 5: 提交**

```bash
git add agent/rule_engine.py tests/test_rule_engine.py
git commit -m "feat: propagate text_zh/text_en into CandidateSegment"
```

---

## Task 3: subtitle.py 双语导出

**Files:**
- Modify: `processing/subtitle.py`
- Create: `tests/test_subtitle.py`

- [ ] **Step 1: 写失败测试**

新建 `tests/test_subtitle.py`：

```python
from __future__ import annotations
from pathlib import Path
from models.edit_plan import Segment
from processing.subtitle import segments_to_srt


SEGS = [
    Segment(start=0.0, end=3.0, text="你好", text_zh="你好", text_en="Hello"),
    Segment(start=3.0, end=6.0, text="再见", text_zh="再见", text_en="Goodbye"),
]


def test_srt_zh_uses_text_zh(tmp_path):
    out = tmp_path / "test_zh.srt"
    segments_to_srt(SEGS, out, lang="zh")
    content = out.read_text(encoding="utf-8")
    assert "你好" in content
    assert "再见" in content
    assert "Hello" not in content


def test_srt_en_uses_text_en(tmp_path):
    out = tmp_path / "test_en.srt"
    segments_to_srt(SEGS, out, lang="en")
    content = out.read_text(encoding="utf-8")
    assert "Hello" in content
    assert "Goodbye" in content
    assert "你好" not in content


def test_srt_fallback_to_text_when_lang_field_none(tmp_path):
    segs = [Segment(start=0.0, end=3.0, text="fallback text")]
    out = tmp_path / "fallback.srt"
    segments_to_srt(segs, out, lang="zh")
    content = out.read_text(encoding="utf-8")
    assert "fallback text" in content


def test_srt_default_lang_is_zh(tmp_path):
    out = tmp_path / "default.srt"
    segments_to_srt(SEGS, out)
    content = out.read_text(encoding="utf-8")
    assert "你好" in content
```

- [ ] **Step 2: 运行确认失败**

```bash
python -m pytest tests/test_subtitle.py -v
```

期望：FAILED（`segments_to_srt` 不接受 `lang` 参数）

- [ ] **Step 3: 实现**

将 `processing/subtitle.py` 替换为：

```python
from __future__ import annotations
from pathlib import Path
from typing import Literal
from models.edit_plan import Segment


def segments_to_srt(
    transcript: list[Segment],
    output_path: Path,
    lang: Literal["zh", "en"] = "zh",
) -> Path:
    """Write transcript segments to an SRT subtitle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for i, seg in enumerate(transcript, 1):
        start = _fmt(seg.start)
        end = _fmt(seg.end)
        if lang == "zh":
            text = (seg.text_zh or seg.text).strip()
        else:
            text = (seg.text_en or seg.text).strip()
        blocks.append(f"{i}\n{start} --> {end}\n{text}")
    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return output_path


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
```

- [ ] **Step 4: 运行确认通过**

```bash
python -m pytest tests/test_subtitle.py -v
```

期望：4 个全部 PASS

- [ ] **Step 5: 提交**

```bash
git add processing/subtitle.py tests/test_subtitle.py
git commit -m "feat: add lang parameter to segments_to_srt for bilingual export"
```

---

## Task 4: Transcriber 参数透传

**Files:**
- Modify: `processing/transcriber.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_transcriber.py` 末尾添加：

```python
@patch("processing.transcriber.whisperx.load_align_model")
@patch("processing.transcriber.whisperx.align")
@patch("processing.transcriber.whisperx.load_audio")
def test_transcribe_passes_language_and_task(mock_load_audio, mock_align, mock_load_align_model, tmp_path):
    mock_load_audio.return_value = MagicMock()
    mock_load_align_model.return_value = (MagicMock(), MagicMock())
    mock_align.return_value = MOCK_ALIGN_RESULT

    t = _make_transcriber_no_init()
    t._backend = "whisperx"
    t._model.transcribe.return_value = MOCK_TRANSCRIBE_RESULT

    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"fake")

    t.transcribe(fake_video, language="zh", task="transcribe")

    call_kwargs = t._model.transcribe.call_args
    assert call_kwargs.kwargs.get("language") == "zh" or "zh" in str(call_kwargs)
```

- [ ] **Step 2: 运行确认失败**

```bash
python -m pytest tests/test_transcriber.py::test_transcribe_passes_language_and_task -v
```

期望：FAILED（`transcribe()` 不接受 `language`/`task` 参数）

- [ ] **Step 3: 实现**

打开 `processing/transcriber.py`，按如下修改各方法签名和调用：

**`transcribe` 方法（第 64 行）**，改为：
```python
def transcribe(self, video_path: Path, diarize: bool = False,
               language: Optional[str] = None, task: str = "transcribe") -> list[Segment]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    duration = self._get_duration(video_path)
    if duration and duration > CHUNK_THRESHOLD:
        return self._transcribe_chunked(video_path, duration, diarize=diarize,
                                        language=language, task=task)
    return self._transcribe_single(video_path, diarize=diarize, language=language, task=task)
```

**`_transcribe_single` 方法（第 73 行）**，改为：
```python
def _transcribe_single(self, video_path: Path, diarize: bool = False,
                       language: Optional[str] = None, task: str = "transcribe") -> list[Segment]:
    if self._backend == "mlx":
        return self._transcribe_single_mlx(video_path, diarize=diarize, language=language, task=task)
    return self._transcribe_single_whisperx(video_path, diarize=diarize, language=language, task=task)
```

**`_transcribe_single_mlx` 方法（第 78 行）**，改签名并在 `mlx_whisper.transcribe` 调用中添加两个参数：
```python
def _transcribe_single_mlx(self, video_path: Path, diarize: bool = False,
                            language: Optional[str] = None, task: str = "transcribe") -> list[Segment]:
```

在 `mlx_whisper.transcribe(...)` 调用（第 99 行）添加两行：
```python
result = mlx_whisper.transcribe(
    audio_path,
    path_or_hf_repo=mlx_model,
    word_timestamps=False,
    verbose=False,
    no_speech_threshold=1.0,
    logprob_threshold=None,
    condition_on_previous_text=False,
    language=language,
    task=task,
)
```

**`_transcribe_single_whisperx` 方法（第 172 行）**，改签名并更新 transcribe 调用：
```python
def _transcribe_single_whisperx(self, video_path: Path, diarize: bool = False,
                                 language: Optional[str] = None, task: str = "transcribe") -> list[Segment]:
    import whisperx
    audio = whisperx.load_audio(str(video_path))
    result = self._model.transcribe(audio, batch_size=16, language=language, task=task)
    # ... 其余不变
```

**`_transcribe_chunked` 方法（第 211 行）**，改签名并透传参数：
```python
def _transcribe_chunked(self, video_path: Path, duration: float, diarize: bool = False,
                        language: Optional[str] = None, task: str = "transcribe") -> list[Segment]:
```

将内部的 `self._transcribe_single(chunk_path, diarize=diarize)` 改为：
```python
for seg in self._transcribe_single(chunk_path, diarize=diarize, language=language, task=task):
```

- [ ] **Step 4: 运行确认通过**

```bash
python -m pytest tests/test_transcriber.py::test_transcribe_passes_language_and_task -v
```

期望：PASS

- [ ] **Step 5: 提交**

```bash
git add processing/transcriber.py tests/test_transcriber.py
git commit -m "feat: thread language and task params through Transcriber call chain"
```

---

## Task 5: 新增 transcribe_bilingual()

**Files:**
- Modify: `processing/transcriber.py`
- Test: `tests/test_transcriber.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_transcriber.py` 末尾添加：

```python
def test_transcribe_bilingual_merges_zh_and_en():
    t = _make_transcriber_no_init()
    t._backend = "whisperx"

    zh_segs = [
        Segment(start=0.0, end=3.5, text="你好世界"),
        Segment(start=3.5, end=7.0, text="这是测试"),
    ]
    en_segs = [
        Segment(start=0.0, end=3.5, text="Hello world"),
        Segment(start=3.5, end=7.0, text="This is a test"),
    ]

    call_count = {"n": 0}
    def fake_transcribe(video_path, diarize=False, language=None, task="transcribe"):
        call_count["n"] += 1
        return zh_segs if language == "zh" else en_segs

    t.transcribe = fake_transcribe

    fake_video = Path("/tmp/fake.mp4")
    # Patch exists check
    with patch.object(Path, "exists", return_value=True):
        result = t.transcribe_bilingual(fake_video)

    assert len(result) == 2
    assert result[0].text_zh == "你好世界"
    assert result[0].text_en == "Hello world"
    assert result[1].text_zh == "这是测试"
    assert result[1].text_en == "This is a test"
    assert result[0].text == "你好世界"


def test_transcribe_bilingual_handles_count_mismatch():
    t = _make_transcriber_no_init()
    t._backend = "whisperx"

    zh_segs = [Segment(start=0.0, end=4.0, text="片段一")]
    en_segs = [
        Segment(start=0.0, end=2.0, text="Part one"),
        Segment(start=2.0, end=4.0, text="Part two"),
    ]

    def fake_transcribe(video_path, diarize=False, language=None, task="transcribe"):
        return zh_segs if language == "zh" else en_segs

    t.transcribe = fake_transcribe

    with patch.object(Path, "exists", return_value=True):
        result = t.transcribe_bilingual(Path("/tmp/fake.mp4"))

    assert len(result) == 1
    assert result[0].text_zh == "片段一"
    assert result[0].text_en is not None
```

- [ ] **Step 2: 运行确认失败**

```bash
python -m pytest tests/test_transcriber.py::test_transcribe_bilingual_merges_zh_and_en tests/test_transcriber.py::test_transcribe_bilingual_handles_count_mismatch -v
```

期望：FAILED（`transcribe_bilingual` 不存在）

- [ ] **Step 3: 实现**

在 `processing/transcriber.py` 末尾（`_get_duration` 之前）添加模块级函数和方法：

在 `Transcriber` 类内部，`_get_duration` 方法之前，添加：

```python
def transcribe_bilingual(self, video_path: Path, diarize: bool = False) -> list[Segment]:
    """Run two Whisper passes in parallel and merge into bilingual segments."""
    from concurrent.futures import ThreadPoolExecutor
    import logging
    _log = logging.getLogger(__name__)

    _zh: dict = {}
    _en: dict = {}

    def _run_zh():
        try:
            _zh["segs"] = self.transcribe(video_path, diarize=diarize,
                                           language="zh", task="transcribe")
        except Exception as exc:
            _log.warning("中文转录失败: %s", exc)
            _zh["segs"] = []

    def _run_en():
        try:
            _en["segs"] = self.transcribe(video_path, diarize=diarize,
                                           language=None, task="translate")
        except Exception as exc:
            _log.warning("英文转录失败: %s", exc)
            _en["segs"] = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        fzh = pool.submit(_run_zh)
        fen = pool.submit(_run_en)
        fzh.result()
        fen.result()

    return _merge_bilingual(_zh["segs"], _en["segs"])
```

在 `Transcriber` 类**外部**，文件末尾添加：

```python
def _merge_bilingual(zh_segs: list[Segment], en_segs: list[Segment]) -> list[Segment]:
    """Merge zh and en segment lists into bilingual Segment objects."""
    if not zh_segs and not en_segs:
        return []

    # Happy path: same count → index merge
    if len(zh_segs) == len(en_segs):
        return [
            Segment(
                start=zh.start, end=zh.end,
                text=zh.text,
                text_zh=zh.text,
                text_en=en.text,
                speaker=zh.speaker,
            )
            for zh, en in zip(zh_segs, en_segs)
        ]

    # Fallback: use zh as base, match en by midpoint proximity
    import logging
    logging.getLogger(__name__).warning(
        "Bilingual segment count mismatch zh=%d en=%d, using time-based matching",
        len(zh_segs), len(en_segs),
    )
    base = zh_segs or en_segs
    other = en_segs if zh_segs else []
    result = []
    for seg in base:
        mid = (seg.start + seg.end) / 2
        best = min(other, key=lambda s: abs((s.start + s.end) / 2 - mid), default=None)
        result.append(Segment(
            start=seg.start, end=seg.end,
            text=seg.text,
            text_zh=seg.text if zh_segs else None,
            text_en=best.text if best else None,
            speaker=seg.speaker,
        ))
    return result
```

- [ ] **Step 4: 运行确认通过**

```bash
python -m pytest tests/test_transcriber.py::test_transcribe_bilingual_merges_zh_and_en tests/test_transcriber.py::test_transcribe_bilingual_handles_count_mismatch -v
```

期望：2 个 PASS

- [ ] **Step 5: 提交**

```bash
git add processing/transcriber.py tests/test_transcriber.py
git commit -m "feat: add transcribe_bilingual with parallel zh/en passes"
```

---

## Task 6: pipeline.py 双列预览

**Files:**
- Modify: `app/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_pipeline.py` 末尾添加：

```python
from models.edit_plan import CandidateSegment

def test_candidates_to_rows_has_two_text_columns():
    from app.pipeline import candidates_to_rows
    candidates = [
        CandidateSegment(
            id="1", start=0.0, end=5.0,
            text_preview="text", text_preview_zh="中文字幕", text_preview_en="English subtitle",
            confidence_score=0.9,
        )
    ]
    rows = candidates_to_rows(candidates)
    assert len(rows) == 1
    row = rows[0]
    # columns: 序号, 说话人, 时间范围, 中文字幕, 英文字幕, 置信度, 包含
    assert len(row) == 7
    assert row[3] == "中文字幕"
    assert row[4] == "English subtitle"

def test_candidates_to_rows_truncates_long_text():
    from app.pipeline import candidates_to_rows
    long_zh = "中" * 100
    long_en = "A" * 100
    candidates = [
        CandidateSegment(
            id="1", start=0.0, end=5.0,
            text_preview="x", text_preview_zh=long_zh, text_preview_en=long_en,
            confidence_score=1.0,
        )
    ]
    rows = candidates_to_rows(candidates)
    assert len(rows[0][3]) <= 81  # 77 chars + "…" + some margin
    assert rows[0][3].endswith("…")
```

- [ ] **Step 2: 运行确认失败**

```bash
python -m pytest tests/test_pipeline.py::test_candidates_to_rows_has_two_text_columns -v
```

期望：FAILED（行有 6 列而非 7 列）

- [ ] **Step 3: 实现**

打开 `app/pipeline.py`，将 `candidates_to_rows` 改为：

```python
def candidates_to_rows(candidates: list[CandidateSegment]) -> list[list]:
    """Convert CandidateSegment list to Gradio Dataframe rows."""
    def _preview(text: str | None) -> str:
        if not text:
            return "—"
        return (text[:77] + "…") if len(text) > 80 else text

    return [
        [
            i + 1,
            seg.speaker or "—",
            f"{seg.start:.1f}s – {seg.end:.1f}s",
            _preview(seg.text_preview_zh),
            _preview(seg.text_preview_en),
            f"{seg.confidence_score:.2f}",
            True,
        ]
        for i, seg in enumerate(candidates)
    ]
```

- [ ] **Step 4: 运行确认通过**

```bash
python -m pytest tests/test_pipeline.py::test_candidates_to_rows_has_two_text_columns tests/test_pipeline.py::test_candidates_to_rows_truncates_long_text -v
```

期望：PASS

- [ ] **Step 5: 提交**

```bash
git add app/pipeline.py tests/test_pipeline.py
git commit -m "feat: split content preview into zh/en columns in candidates_to_rows"
```

---

## Task 7: main.py 主流程更新

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: 更新转录调用**

打开 `app/main.py`，在 `_do_transcribe` 函数内（约第 103 行），将：
```python
_result["transcript"] = _transcriber.transcribe(
    video_path, diarize="speaker" in selected
)
```
改为：
```python
_result["transcript"] = _transcriber.transcribe_bilingual(
    video_path, diarize="speaker" in selected
)
```

- [ ] **Step 2: 更新进度状态文案**

将约第 134 行（心跳 status）中的 `正在转录音频` 改为 `正在双语转录（zh + en）`：

```python
status = f"正在双语转录（zh + en）{_dots[_tick % 3]}  ⏱ {_elapsed}\n\n{recent}" if recent else f"正在双语转录（zh + en）{_dots[_tick % 3]}  ⏱ {_elapsed}"
```

- [ ] **Step 3: 更新字幕导出为双文件**

找到约第 149-154 行的字幕生成代码：
```python
if "subtitle" in selected:
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    srt_file = output_dir / f"{video_path.stem}.srt"
    segments_to_srt(transcript, srt_file)
    srt_path_str = str(srt_file)
```

替换为：
```python
srt_paths: list[str] = []
if "subtitle" in selected:
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    srt_zh = output_dir / f"{video_path.stem}_zh.srt"
    srt_en = output_dir / f"{video_path.stem}_en.srt"
    segments_to_srt(transcript, srt_zh, lang="zh")
    segments_to_srt(transcript, srt_en, lang="en")
    srt_paths = [str(srt_zh), str(srt_en)]
```

- [ ] **Step 4: 全局替换 srt_path_str 为 srt_paths**

在 `run_pipeline` 函数中，将所有 `srt_path_str` 变量改为 `srt_paths`，并将所有：
```python
gr.update(visible=True, value=srt_path_str)
```
改为：
```python
gr.update(visible=True, value=srt_paths)
```

将所有：
```python
if srt_path_str else _no_srt
```
改为：
```python
if srt_paths else _no_srt
```

将 state 中：
```python
new_state["srt_path"] = srt_path_str
```
改为：
```python
new_state["srt_path"] = srt_paths
```

- [ ] **Step 5: 更新 confirm_speaker 中的 srt 读取**

找到 `confirm_speaker` 函数（约第 272-276 行）：
```python
srt_update = (
    gr.update(visible=True, value=state["srt_path"])
    if "srt_path" in state and Path(state["srt_path"]).exists()
    else _no_srt
)
```

替换为：
```python
_srt_paths = state.get("srt_path", [])
srt_update = (
    gr.update(visible=True, value=_srt_paths)
    if _srt_paths and all(Path(p).exists() for p in _srt_paths)
    else _no_srt
)
```

- [ ] **Step 6: 更新表格 headers 和 datatype**

找到 `review_table = gr.Dataframe(...)` 的定义（约第 425 行），改为：
```python
review_table = gr.Dataframe(
    headers=["序号", "说话人", "时间范围", "中文字幕", "英文字幕", "置信度", "包含"],
    datatype=["number", "str", "str", "str", "str", "str", "bool"],
    interactive=True,
    label="勾选要保留的片段",
)
```

- [ ] **Step 7: 更新 srt_download 为多文件**

找到 `srt_download = gr.File(...)` 的定义（约第 431 行），改为：
```python
srt_download = gr.File(
    label="📄 字幕文件（SRT）",
    file_count="multiple",
    visible=False,
)
```

- [ ] **Step 8: 更新 export_raw 中的列索引**

`export_raw` 函数读取 `row[5]` 作为"包含"checkbox（约第 341 行）。现在表格新增了一列（英文字幕在第 4 列），checkbox 移到第 6 列（索引 6）：

```python
if row[6]:
    approved_indices.add(int(row[0]) - 1)
```

- [ ] **Step 9: 验证启动无报错**

```bash
cd /Users/jianchengsun/media-dev/auto-vedio-edit-agent
source .venv/bin/activate
python -c "from app.main import demo; print('OK')"
```

期望：打印 `OK`，无 import 错误

- [ ] **Step 10: 提交**

```bash
git add app/main.py
git commit -m "feat: wire bilingual transcription into UI, dual SRT export, two-column table"
```

---

## Task 8: 回归验证

- [ ] **Step 1: 运行全量测试（忽略已知失败）**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | grep -E "PASSED|FAILED|ERROR"
```

期望：
- 所有新增测试 PASSED
- pre-existing 失败数量不超过 8 个（与改动前一致）
- 无新增 FAILED

- [ ] **Step 2: 最终提交**

如果 Step 1 结果符合预期：

```bash
git add -A
git status  # 确认没有意外文件
git commit -m "test: bilingual subtitle regression check passes"
```
