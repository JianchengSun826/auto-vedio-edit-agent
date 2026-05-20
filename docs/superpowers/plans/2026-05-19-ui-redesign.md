# UI 重设计实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重写 Gradio UI，增加多选功能按钮、gr.Progress 进度条步骤条、按说话人导出多个片段，并对免费/付费路径做清晰区分。

**Architecture:** 新增 `app/pipeline.py` 封装可测试的纯逻辑（构建 EditPlan、转换表格行、渲染步骤条 HTML）；在 `agent/orchestrator.py` 增加 `transcribe_only()` 供前端分步调用；完整重写 `app/main.py` 的布局与事件绑定。

**Tech Stack:** Gradio 4.x, Pydantic, Python 3.9+

---

## 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `app/pipeline.py` | 纯逻辑：build_plan_from_buttons, candidates_to_rows, extract_speakers, step_html |
| 新建 | `tests/test_pipeline.py` | pipeline.py 的单元测试 |
| 修改 | `agent/orchestrator.py` | 增加 `transcribe_only()` 方法 |
| 重写 | `app/main.py` | 完整新布局 + 事件绑定 |

---

## Task 1: 创建 `app/pipeline.py`

**Files:**
- Create: `app/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: 写失败测试**

新建 `tests/test_pipeline.py`：

```python
import pytest
from models.edit_plan import (
    EditPlan, EditMode, Rule, RuleType, Segment, CandidateSegment
)


# ── build_plan_from_buttons ──────────────────────────────────────────────────

def test_build_plan_keyword_only():
    from app.pipeline import build_plan_from_buttons
    plan = build_plan_from_buttons(
        selected=["keyword"],
        keywords=["竞品", "价格"],
        keyword_before=3.0,
        keyword_after=5.0,
        time_start=None,
        time_end=None,
        speaker_ids=[],
    )
    assert plan.mode == EditMode.HIGHLIGHT_EXTRACTION
    assert len(plan.rules) == 1
    rule = plan.rules[0]
    assert rule.type == RuleType.KEYWORD_MATCH
    assert rule.keywords == ["竞品", "价格"]
    assert rule.padding_before_sec == 3.0
    assert rule.padding_after_sec == 5.0


def test_build_plan_silence_only():
    from app.pipeline import build_plan_from_buttons
    plan = build_plan_from_buttons(
        selected=["silence"],
        keywords=[],
        keyword_before=3.0,
        keyword_after=5.0,
        time_start=None,
        time_end=None,
        speaker_ids=[],
    )
    assert len(plan.rules) == 1
    assert plan.rules[0].type == RuleType.SILENCE_CUT


def test_build_plan_multi_select():
    from app.pipeline import build_plan_from_buttons
    plan = build_plan_from_buttons(
        selected=["keyword", "silence"],
        keywords=["竞品"],
        keyword_before=2.0,
        keyword_after=3.0,
        time_start=None,
        time_end=None,
        speaker_ids=[],
    )
    types = [r.type for r in plan.rules]
    assert RuleType.KEYWORD_MATCH in types
    assert RuleType.SILENCE_CUT in types


def test_build_plan_time_range():
    from app.pipeline import build_plan_from_buttons
    plan = build_plan_from_buttons(
        selected=["time"],
        keywords=[],
        keyword_before=3.0,
        keyword_after=5.0,
        time_start=10.0,
        time_end=60.0,
        speaker_ids=[],
    )
    rule = plan.rules[0]
    assert rule.type == RuleType.TIME_RANGE
    assert rule.start_sec == 10.0
    assert rule.end_sec == 60.0


def test_build_plan_speaker():
    from app.pipeline import build_plan_from_buttons
    plan = build_plan_from_buttons(
        selected=["speaker"],
        keywords=[],
        keyword_before=3.0,
        keyword_after=5.0,
        time_start=None,
        time_end=None,
        speaker_ids=["SPEAKER_00", "SPEAKER_01"],
    )
    rule = plan.rules[0]
    assert rule.type == RuleType.SPEAKER_FILTER
    assert rule.speakers == ["SPEAKER_00", "SPEAKER_01"]


def test_build_plan_keyword_empty_skipped():
    from app.pipeline import build_plan_from_buttons
    plan = build_plan_from_buttons(
        selected=["keyword"],
        keywords=[],  # empty keywords → rule skipped
        keyword_before=3.0,
        keyword_after=5.0,
        time_start=None,
        time_end=None,
        speaker_ids=[],
    )
    assert len(plan.rules) == 0


# ── candidates_to_rows ───────────────────────────────────────────────────────

def test_candidates_to_rows():
    from app.pipeline import candidates_to_rows
    import uuid
    candidates = [
        CandidateSegment(id=str(uuid.uuid4()), start=2.0, end=8.0,
                         text_preview="竞品价格很高", confidence_score=0.95, speaker="SPEAKER_00"),
        CandidateSegment(id=str(uuid.uuid4()), start=15.0, end=22.0,
                         text_preview="我们更好", confidence_score=0.80, speaker=None),
    ]
    rows = candidates_to_rows(candidates)
    assert len(rows) == 2
    assert rows[0][0] == 1          # 序号
    assert rows[0][1] == "SPEAKER_00"
    assert rows[0][2] == "2.0s – 8.0s"
    assert rows[0][3] == "竞品价格很高"
    assert rows[0][4] == "0.95"
    assert rows[0][5] is True       # 默认选中
    assert rows[1][1] == "—"        # 无说话人显示 —


# ── extract_speakers ─────────────────────────────────────────────────────────

def test_extract_speakers():
    from app.pipeline import extract_speakers
    transcript = [
        Segment(start=0.0, end=5.0, text="a", speaker="SPEAKER_01"),
        Segment(start=5.0, end=10.0, text="b", speaker="SPEAKER_00"),
        Segment(start=10.0, end=15.0, text="c", speaker="SPEAKER_01"),  # duplicate
        Segment(start=15.0, end=20.0, text="d", speaker=None),
    ]
    speakers = extract_speakers(transcript)
    assert speakers == ["SPEAKER_00", "SPEAKER_01"]  # sorted, deduplicated, no None


def test_extract_speakers_no_diarization():
    from app.pipeline import extract_speakers
    transcript = [Segment(start=0.0, end=5.0, text="a", speaker=None)]
    assert extract_speakers(transcript) == []
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /Users/jianchengsun/media-dev/auto-vedio-edit-agent
PYTHONPATH=. pytest tests/test_pipeline.py -v 2>&1 | head -30
```

预期：`ModuleNotFoundError: No module named 'app.pipeline'`

- [ ] **Step 3: 创建 `app/pipeline.py`**

```python
from __future__ import annotations
from models.edit_plan import (
    EditPlan, EditMode, Rule, RuleType, CandidateSegment, Segment,
)


def build_plan_from_buttons(
    selected: list[str],
    keywords: list[str],
    keyword_before: float,
    keyword_after: float,
    time_start: float | None,
    time_end: float | None,
    speaker_ids: list[str],
) -> EditPlan:
    """Construct EditPlan directly from button selections, no LLM needed."""
    rules: list[Rule] = []

    if "keyword" in selected and keywords:
        rules.append(Rule(
            type=RuleType.KEYWORD_MATCH,
            keywords=keywords,
            padding_before_sec=keyword_before,
            padding_after_sec=keyword_after,
        ))

    if "speaker" in selected and speaker_ids:
        rules.append(Rule(
            type=RuleType.SPEAKER_FILTER,
            speakers=speaker_ids,
            padding_before_sec=0.0,
            padding_after_sec=0.0,
        ))

    if "time" in selected and time_start is not None and time_end is not None:
        rules.append(Rule(
            type=RuleType.TIME_RANGE,
            start_sec=time_start,
            end_sec=time_end,
        ))

    if "silence" in selected:
        rules.append(Rule(type=RuleType.SILENCE_CUT))

    return EditPlan(
        mode=EditMode.HIGHLIGHT_EXTRACTION,
        rules=rules,
        output_formats=[],
        segment_count_hint=max(1, len(rules) * 3),
    )


def candidates_to_rows(candidates: list[CandidateSegment]) -> list[list]:
    """Convert CandidateSegment list to Gradio Dataframe rows."""
    return [
        [
            i + 1,
            seg.speaker or "—",
            f"{seg.start:.1f}s – {seg.end:.1f}s",
            seg.text_preview[:80],
            f"{seg.confidence_score:.2f}",
            True,
        ]
        for i, seg in enumerate(candidates)
    ]


def extract_speakers(transcript: list[Segment]) -> list[str]:
    """Return sorted unique speaker IDs from transcript, excluding None."""
    seen: set[str] = set()
    result: list[str] = []
    for seg in transcript:
        if seg.speaker and seg.speaker not in seen:
            seen.add(seg.speaker)
            result.append(seg.speaker)
    return sorted(result)


def step_html(done: set[int], active: int, skip_llm: bool = False) -> str:
    """Render 3-step progress indicator as HTML. Steps are 1-indexed."""
    steps = [(1, "① 音频转录"), (2, "② 解析意图"), (3, "③ 执行规则")]
    parts: list[str] = []
    for idx, (num, label) in enumerate(steps):
        if num in done:
            bg, color, suffix = "#2a6a2a", "#6fa", " ✓"
        elif num == active:
            bg, color, suffix = "#3a5a9a", "#fff", " ⟳"
        elif skip_llm and num == 2:
            bg, color, suffix = "#252545", "#444", " —"
        else:
            bg, color, suffix = "#252545", "#555", ""
        r = "4px 0 0 4px" if idx == 0 else ("0 4px 4px 0" if idx == 2 else "0")
        parts.append(
            f'<div style="flex:1;text-align:center;background:{bg};border-radius:{r};'
            f'padding:6px 4px;font-size:12px;color:{color}">{label}{suffix}</div>'
        )
        if idx < 2:
            parts.append(
                f'<div style="width:0;height:0;border-top:12px solid transparent;'
                f'border-bottom:12px solid transparent;border-left:8px solid {bg}"></div>'
            )
    inner = "".join(parts)
    return f'<div style="display:flex;align-items:center;margin:8px 0">{inner}</div>'
```

- [ ] **Step 4: 运行测试确认全部通过**

```bash
PYTHONPATH=. pytest tests/test_pipeline.py -v
```

预期：所有测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add app/pipeline.py tests/test_pipeline.py
git commit -m "feat: add app/pipeline.py with build_plan_from_buttons, candidates_to_rows, step_html"
```

---

## Task 2: 扩展 `agent/orchestrator.py`

**Files:**
- Modify: `agent/orchestrator.py`

- [ ] **Step 1: 在 `Orchestrator` 类末尾添加 `transcribe_only()` 方法**

在 `agent/orchestrator.py` 的 `run()` 方法之后追加：

```python
    def transcribe_only(self, video_path: Path) -> tuple[list[Segment], float | None]:
        """Transcribe video and return (transcript, duration). No LLM called."""
        transcript = self._transcriber.transcribe(video_path)
        try:
            duration = get_video_duration(video_path)
        except Exception:
            duration = None
        return transcript, duration
```

- [ ] **Step 2: 确认现有测试不受影响**

```bash
PYTHONPATH=. pytest tests/test_orchestrator.py -v
```

预期：全部 PASS（只添加了新方法，未改动现有逻辑）。

- [ ] **Step 3: Commit**

```bash
git add agent/orchestrator.py
git commit -m "feat: add Orchestrator.transcribe_only() for step-by-step pipeline"
```

---

## Task 3: 重写 `app/main.py`

**Files:**
- Rewrite: `app/main.py`

> 注意：此步骤完整替换现有 `app/main.py`。现有功能会短暂不可用直到本 Task 完成。

- [ ] **Step 1: 完整替换 `app/main.py`**

用以下内容完整替换 `app/main.py`：

```python
from __future__ import annotations
import gradio as gr
from pathlib import Path

from agent.orchestrator import Orchestrator
from agent.intent_parser import IntentParser
from agent.rule_engine import RuleEngine
from processing.exporter import Exporter
from models.edit_plan import CandidateSegment, OutputFormat, Platform, Segment
from config.settings import settings
from app.pipeline import (
    build_plan_from_buttons,
    candidates_to_rows,
    extract_speakers,
    step_html,
)

PLATFORM_MAP = {
    "抖音": Platform.DOUYIN,
    "B站": Platform.BILIBILI,
    "YouTube": Platform.YOUTUBE,
    "微信视频号": Platform.WECHAT,
}

PRICING_HTML = """
<div style="background:#182818;border:1px solid #2a4a2a;border-radius:6px;
            padding:10px 12px;font-size:12px;line-height:1.8;color:#aaa;margin-top:4px">
  <div style="color:#6fa;font-weight:bold">ℹ️ 使用 Claude AI · 产生少量费用</div>
  <div>10分钟视频 ≈ ¥0.03 &nbsp;·&nbsp; 1小时视频 ≈ ¥0.28</div>
  <div style="color:#777;margin-top:4px">
    适合：理解模糊描述（"找情绪激动的片段"）&nbsp;·&nbsp;
    口语时间（"截取前五分钟"）&nbsp;·&nbsp;
    去除口误和重复片段
  </div>
</div>
"""


def _toggle(feature: str, current: list[str]):
    new = current.copy()
    if feature in new:
        new.remove(feature)
    else:
        new.append(feature)
    n = len(new)
    label = f"🚀 开始分析（已选 {n} 个功能）" if n else "🚀 开始分析"
    return (
        new,
        gr.update(variant="primary" if "keyword" in new else "secondary"),
        gr.update(variant="primary" if "speaker" in new else "secondary"),
        gr.update(variant="primary" if "time" in new else "secondary"),
        gr.update(variant="primary" if "silence" in new else "secondary"),
        gr.update(visible="keyword" in new),
        gr.update(visible="time" in new),
        gr.update(value=label),
    )


def run_pipeline(
    video_file, selected: list[str],
    kw_text: str, kw_before: float, kw_after: float,
    t_start: float, t_end: float,
    instruction: str, state: dict,
    progress=gr.Progress(),
):
    """Phase-1 pipeline: transcribe → (LLM or button plan) → execute rules.
    For speaker mode, stops after transcription and returns speaker list.
    Yields 9-tuple: (progress_group, speaker_group, results_group,
                     step_bar, status, speaker_selector, results_header,
                     review_table, state)
    """
    EMPTY = (
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        "", "", gr.update(), "", [], state,
    )

    if video_file is None:
        yield (*EMPTY[:4], "⚠️ 请先上传视频", *EMPTY[5:])
        return
    if not selected and not instruction.strip():
        yield (*EMPTY[:4], "⚠️ 请选择功能或输入需求", *EMPTY[5:])
        return

    use_llm = bool(instruction.strip()) and not selected
    skip_llm = bool(selected)

    # Show progress area
    yield (
        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
        step_html(set(), 1, skip_llm), "准备开始…", gr.update(), "", [], state,
    )

    # Step 1: Transcribe
    progress(0.1, desc="正在转录音频…")
    orch = Orchestrator()
    video_path = Path(video_file)
    transcript, duration = orch.transcribe_only(video_path)

    yield (
        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
        step_html({1}, 1, skip_llm),
        f"✓ 转录完成，共 {len(transcript)} 个片段",
        gr.update(), "", [], state,
    )

    # Speaker mode: stop and show selector
    if "speaker" in selected:
        speakers = extract_speakers(transcript)
        new_state = {
            **state,
            "transcript": [s.model_dump() for s in transcript],
            "duration": duration,
            "video_path": str(video_path),
            "selected": selected,
            "kw_text": kw_text,
            "kw_before": kw_before,
            "kw_after": kw_after,
            "t_start": t_start,
            "t_end": t_end,
        }
        yield (
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
            step_html({1}, 2, False),
            f"✓ 检测到 {len(speakers)} 位说话人，请选择后点击「确认筛选」",
            gr.update(choices=speakers, value=[]),
            "", [], new_state,
        )
        return

    # Button path
    if selected:
        progress(0.7, desc="正在执行规则…")
        keywords = [k.strip() for k in kw_text.split(",") if k.strip()]
        plan = build_plan_from_buttons(
            selected=selected, keywords=keywords,
            keyword_before=kw_before, keyword_after=kw_after,
            time_start=t_start, time_end=t_end,
            speaker_ids=[],
        )
        yield (
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
            step_html({1}, 3, True), "正在执行规则…", gr.update(), "", [], state,
        )

    # LLM path
    else:
        progress(0.5, desc="正在解析意图…")
        yield (
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
            step_html({1}, 2, False), "正在调用 AI 解析意图…", gr.update(), "", [], state,
        )
        parser = IntentParser()
        plan = parser.parse(
            user_instruction=instruction,
            transcript=[s.model_dump() for s in transcript],
        )
        progress(0.8, desc="正在执行规则…")
        yield (
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
            step_html({1, 2}, 3, False), "正在执行规则…", gr.update(), "", [], state,
        )

    engine = RuleEngine()
    candidates = engine.execute(plan, transcript, video_path, duration)
    progress(1.0, desc="完成")

    new_state = {
        **state,
        "candidates": [c.model_dump() for c in candidates],
        "video_path": str(video_path),
    }
    yield (
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
        step_html({1, 2, 3} if use_llm else {1, 3}, 0, skip_llm),
        "完成",
        gr.update(),
        f"✅ 找到 **{len(candidates)}** 个候选片段",
        candidates_to_rows(candidates),
        new_state,
    )


def confirm_speaker(
    speaker_ids: list[str], state: dict,
    progress=gr.Progress(),
):
    """Phase-2: apply SPEAKER_FILTER after user picks speakers.
    Yields 8-tuple: (progress_group, speaker_group, results_group,
                     step_bar, status, results_header, review_table, state)
    """
    yield (
        gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
        step_html({1}, 3, True), "正在执行说话人筛选…", "", [], state,
    )
    progress(0.5, desc="正在执行规则…")

    transcript = [Segment(**s) for s in state["transcript"]]
    video_path = Path(state["video_path"])
    duration = state.get("duration")

    plan = build_plan_from_buttons(
        selected=state["selected"],
        keywords=[k.strip() for k in state.get("kw_text", "").split(",") if k.strip()],
        keyword_before=state.get("kw_before", 3.0),
        keyword_after=state.get("kw_after", 5.0),
        time_start=state.get("t_start"),
        time_end=state.get("t_end"),
        speaker_ids=speaker_ids,
    )

    engine = RuleEngine()
    candidates = engine.execute(plan, transcript, video_path, duration)
    progress(1.0, desc="完成")

    new_state = {
        **state,
        "candidates": [c.model_dump() for c in candidates],
    }
    yield (
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
        step_html({1, 3}, 0, True),
        "完成",
        f"✅ 找到 **{len(candidates)}** 个候选片段",
        candidates_to_rows(candidates),
        new_state,
    )


def export_approved(review_table, platform_choices: list[str], state: dict):
    if "candidates" not in state:
        return gr.update(visible=False)

    candidates = [CandidateSegment(**c) for c in state["candidates"]]
    video_path = Path(state["video_path"])

    approved = {int(row[0]) - 1 for row in review_table if row[5]}
    for i, seg in enumerate(candidates):
        seg.included = (i in approved)

    formats = [OutputFormat(platform=PLATFORM_MAP[p]) for p in platform_choices]
    exporter = Exporter(output_dir=settings.output_dir)
    paths = exporter.export(video_path, candidates, formats)
    return gr.update(visible=True, value=[str(p) for p in paths])


# ── UI Layout ────────────────────────────────────────────────────────────────

with gr.Blocks(title="视频自动剪辑 Agent") as demo:
    session_state = gr.State({})
    selected_features = gr.State([])

    gr.Markdown("# 🎬 视频自动剪辑 Agent")

    # ── Region 1: upload + controls (always visible) ──────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="上传视频")

        with gr.Column(scale=2):
            gr.Markdown("**选择功能（可多选，免费）：**")
            with gr.Row():
                btn_keyword = gr.Button("🔍 关键词提取", variant="secondary", size="sm")
                btn_speaker = gr.Button("🎙 按说话人剪辑", variant="secondary", size="sm")
            with gr.Row():
                btn_time = gr.Button("⏱ 截取时间段", variant="secondary", size="sm")
                btn_silence = gr.Button("✂️ 去除静音", variant="secondary", size="sm")

            with gr.Group(visible=False) as keyword_params_group:
                kw_input = gr.Textbox(
                    label="关键词（逗号分隔）",
                    placeholder="竞品, 价格, 方案",
                )
                with gr.Row():
                    kw_before = gr.Number(label="片段前留（秒）", value=3, minimum=0, maximum=60, step=1)
                    kw_after = gr.Number(label="片段后留（秒）", value=5, minimum=0, maximum=60, step=1)

            with gr.Group(visible=False) as time_params_group:
                with gr.Row():
                    t_start = gr.Number(label="起始时间（秒）", value=0, minimum=0)
                    t_end = gr.Number(label="结束时间（秒）", value=60, minimum=0)

            gr.Markdown("---")
            instruction_input = gr.Textbox(
                label="自定义需求（AI 解析）",
                placeholder="例如：找情绪激动的片段，或截取前五分钟…",
                lines=2,
            )
            gr.HTML(PRICING_HTML)
            start_btn = gr.Button("🚀 开始分析", variant="primary")

    # ── Region 2: progress (appears after start) ──────────────────────────────
    with gr.Group(visible=False) as progress_group:
        step_bar = gr.HTML()
        progress_status = gr.Textbox(label="进度", interactive=False)

    # Speaker selector (shown only in speaker mode after transcription)
    with gr.Group(visible=False) as speaker_group:
        gr.Markdown("### 请选择要保留的说话人：")
        speaker_selector = gr.CheckboxGroup(label="说话人", choices=[])
        confirm_speaker_btn = gr.Button("确认筛选", variant="primary")

    # ── Region 3: results (appears after processing) ──────────────────────────
    with gr.Group(visible=False) as results_group:
        results_header = gr.Markdown()
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
        export_files = gr.File(
            label="下载导出文件",
            file_count="multiple",
            visible=False,
        )

    # ── Event wiring ──────────────────────────────────────────────────────────

    _toggle_outputs = [
        selected_features,
        btn_keyword, btn_speaker, btn_time, btn_silence,
        keyword_params_group, time_params_group,
        start_btn,
    ]

    btn_keyword.click(fn=lambda s: _toggle("keyword", s),
                      inputs=[selected_features], outputs=_toggle_outputs)
    btn_speaker.click(fn=lambda s: _toggle("speaker", s),
                      inputs=[selected_features], outputs=_toggle_outputs)
    btn_time.click(fn=lambda s: _toggle("time", s),
                   inputs=[selected_features], outputs=_toggle_outputs)
    btn_silence.click(fn=lambda s: _toggle("silence", s),
                      inputs=[selected_features], outputs=_toggle_outputs)

    _pipeline_outputs = [
        progress_group, speaker_group, results_group,
        step_bar, progress_status,
        speaker_selector,
        results_header, review_table,
        session_state,
    ]

    start_btn.click(
        fn=run_pipeline,
        inputs=[
            video_input, selected_features,
            kw_input, kw_before, kw_after,
            t_start, t_end,
            instruction_input, session_state,
        ],
        outputs=_pipeline_outputs,
    )

    confirm_speaker_btn.click(
        fn=confirm_speaker,
        inputs=[speaker_selector, session_state],
        outputs=[
            progress_group, speaker_group, results_group,
            step_bar, progress_status,
            results_header, review_table,
            session_state,
        ],
    )

    export_btn.click(
        fn=export_approved,
        inputs=[review_table, platform_select, session_state],
        outputs=[export_files],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
```

- [ ] **Step 2: 运行已有 UI 测试确认不报导入错误**

```bash
PYTHONPATH=. pytest tests/test_ui.py -v 2>&1 | head -30
```

预期：无 `ImportError`，测试运行（失败也可以，主要确认模块可导入）。

- [ ] **Step 3: 运行全部测试**

```bash
PYTHONPATH=. pytest tests/ -v --ignore=tests/test_transcriber.py 2>&1 | tail -20
```

（跳过 `test_transcriber.py`，它需要真实模型文件）

预期：`test_pipeline.py` 全绿；其余已有测试保持通过。

- [ ] **Step 4: 本地启动确认 UI 可访问**

```bash
PYTHONPATH=. python app/main.py &
sleep 3
curl -s http://localhost:7860 | grep -o "<title>[^<]*</title>"
```

预期：`<title>视频自动剪辑 Agent</title>`

- [ ] **Step 5: Commit**

```bash
git add app/main.py
git commit -m "feat: rewrite UI with multi-select buttons, progress bar, speaker two-phase flow"
```

---

## 自查结果

- **规范覆盖**：
  - ✅ 功能按钮多选（`_toggle` + `selected_features` State）
  - ✅ 按钮免费路径（`build_plan_from_buttons`，跳过 LLM）
  - ✅ 说话人两阶段流程（phase1 停止 → 用户选择 → `confirm_speaker`）
  - ✅ 进度条（`gr.Progress` + `step_html` HTML）
  - ✅ 付费说明常驻信息框（`PRICING_HTML`）
  - ✅ 结果返回多个独立片段（不拼接，进入 Dataframe）
  - ✅ 导出用 `gr.File`（支持逐个下载）
  - ✅ 去除静音返回多片段（由 `RuleEngine._apply_silence_cut` 已实现分段逻辑）
- **无占位符**：所有步骤含完整代码
- **类型一致**：`build_plan_from_buttons` 在 Task 1 定义，Task 3 导入使用，签名一致
