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

    # button path always takes precedence over text instruction when both are provided
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
        step_html({1}, 2, skip_llm),
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
    if "transcript" not in state:
        yield (gr.update(), gr.update(), gr.update(), "", "⚠️ 请先上传视频并运行分析", "", [], state)
        return

    yield (
        gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
        step_html({1}, 3, True), "正在执行说话人筛选…", "", [], state,
    )

    if not speaker_ids:
        yield (gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
               step_html({1}, 2, False), "⚠️ 请至少选择一位说话人", "", [], state)
        return

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
