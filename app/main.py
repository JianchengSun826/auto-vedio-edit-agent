from __future__ import annotations
import gradio as gr
from pathlib import Path

from agent.intent_parser import IntentParser
from agent.rule_engine import RuleEngine
from processing.transcriber import Transcriber
from processing.ffmpeg_utils import cut_segment, get_video_duration
from processing.subtitle import segments_to_srt
from models.edit_plan import CandidateSegment, Segment
from config.settings import settings
from app.pipeline import (
    build_plan_from_buttons,
    candidates_to_rows,
    extract_speakers,
    step_html,
)

# Load WhisperX model once at startup — shared across all requests
_transcriber = Transcriber()


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
        gr.update(variant="primary" if "subtitle" in new else "secondary"),
        gr.update(variant="primary" if "silence" in new else "secondary"),
        gr.update(visible="keyword" in new),
        gr.update(value=label),
    )


def run_pipeline(
    video_file, selected: list[str],
    kw_text: str, kw_before: float, kw_after: float,
    instruction: str, state: dict,
    progress=gr.Progress(track_tqdm=True),
):
    """Phase-1 pipeline: transcribe → (LLM or button plan) → execute rules.
    For speaker mode, stops after transcription and returns speaker list.
    Yields 10-tuple: (progress_group, speaker_group, results_group,
                      step_bar, status, speaker_selector, results_header,
                      review_table, srt_download, state)
    """
    _no_srt = gr.update(visible=False, value=None)

    def _idle(*extra):
        return (
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            "", "", gr.update(), "", [], _no_srt, state,
        )

    if video_file is None:
        yield _idle(); return  # noqa: E702 – keep compact
    if not selected and not instruction.strip():
        yield (
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            "", "⚠️ 请选择功能或输入需求", gr.update(), "", [], _no_srt, state,
        )
        return

    use_llm = bool(instruction.strip()) and not selected
    skip_llm = bool(selected)

    # Show progress area
    yield (
        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
        step_html(set(), 1, skip_llm), "准备开始…", gr.update(), "", [], _no_srt, state,
    )

    # Step 1: Transcribe in a background thread so we can yield heartbeats and
    # keep the WebSocket alive during long videos (otherwise Gradio disconnects).
    import threading
    progress(0.1)
    video_path = Path(video_file)
    _result: dict = {}

    def _do_transcribe():
        try:
            _result["transcript"] = _transcriber.transcribe(
                video_path, diarize="speaker" in selected
            )
        except Exception as exc:
            _result["error"] = exc

    t = threading.Thread(target=_do_transcribe, daemon=True)
    t.start()
    while t.is_alive():
        t.join(timeout=5)
        if t.is_alive():
            yield (
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                step_html(set(), 1, skip_llm), "正在转录音频…", gr.update(), "", [], _no_srt, state,
            )

    if "error" in _result:
        raise _result["error"]
    transcript = _result["transcript"]

    try:
        duration = get_video_duration(video_path)
    except Exception:
        duration = None

    yield (
        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
        step_html({1}, 2, skip_llm),
        f"✓ 转录完成，共 {len(transcript)} 个片段",
        gr.update(), "", [], _no_srt, state,
    )

    # Generate SRT if subtitle extraction requested
    srt_path_str: str | None = None
    if "subtitle" in selected:
        output_dir = Path(settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        srt_file = output_dir / f"{video_path.stem}.srt"
        segments_to_srt(transcript, srt_file)
        srt_path_str = str(srt_file)

    # Speaker mode: stop and show selector
    if "speaker" in selected:
        speakers = extract_speakers(transcript)
        if len(speakers) == 0:
            no_token = not settings.hf_token
            msg = (
                "⚠️ 未检测到说话人。说话人分离需要配置 HuggingFace Token，"
                "请在 api_keys.env 中填写 HF_TOKEN 并重启服务。详见 README 安装说明。"
                if no_token else
                "⚠️ 未检测到说话人。pyannote 分析返回空结果，请确认视频有清晰的多人对话音频。"
            )
            yield (
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                step_html({1}, 2, False), msg, gr.update(), "", [], _no_srt, state,
            )
            return
        srt_update = gr.update(visible=True, value=srt_path_str) if srt_path_str else _no_srt
        new_state = {
            **state,
            "transcript": [s.model_dump() for s in transcript],
            "duration": duration,
            "video_path": str(video_path),
            "selected": selected,
            "kw_text": kw_text,
            "kw_before": kw_before,
            "kw_after": kw_after,
        }
        if srt_path_str:
            new_state["srt_path"] = srt_path_str
        yield (
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
            step_html({1}, 2, False),
            f"✓ 检测到 {len(speakers)} 位说话人，请选择后点击「确认筛选」",
            gr.update(choices=speakers, value=[]),
            "", [], srt_update, new_state,
        )
        return

    # Button path (non-speaker, non-subtitle features)
    non_meta = [f for f in selected if f not in ("subtitle",)]
    if non_meta:
        progress(0.7)
        keywords = [k.strip() for k in (kw_text or "").split(",") if k.strip()]
        plan = build_plan_from_buttons(
            selected=non_meta, keywords=keywords,
            keyword_before=kw_before, keyword_after=kw_after,
            time_start=None, time_end=None,
            speaker_ids=[],
        )
        yield (
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
            step_html({1}, 3, True), "正在执行规则…", gr.update(), "", [], _no_srt, state,
        )
    elif "subtitle" in selected:
        plan = build_plan_from_buttons(
            selected=[], keywords=[], keyword_before=0, keyword_after=0,
            time_start=None, time_end=None, speaker_ids=[],
        )

    # LLM path
    else:
        progress(0.5)
        yield (
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
            step_html({1}, 2, False), "正在调用 AI 解析意图…", gr.update(), "", [], _no_srt, state,
        )
        parser = IntentParser()
        plan = parser.parse(
            user_instruction=instruction,
            transcript=[s.model_dump() for s in transcript],
        )
        progress(0.8)
        yield (
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
            step_html({1, 2}, 3, False), "正在执行规则…", gr.update(), "", [], _no_srt, state,
        )

    engine = RuleEngine()
    candidates = engine.execute(plan, transcript, video_path, duration)
    progress(1.0)

    new_state = {
        **state,
        "candidates": [c.model_dump() for c in candidates],
        "video_path": str(video_path),
    }
    if srt_path_str:
        new_state["srt_path"] = srt_path_str

    srt_update = gr.update(visible=True, value=srt_path_str) if srt_path_str else _no_srt
    n_clips = len(candidates)
    if n_clips == 0 and "keyword" in selected and not use_llm:
        header = "⚠️ 未找到匹配关键词的片段。可以尝试换用「自定义需求」输入框，用 AI 理解模糊描述。"
    else:
        header = f"✅ 找到 **{n_clips}** 个候选片段"
    yield (
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
        step_html({1, 2, 3} if use_llm else {1, 3}, 0, skip_llm),
        "完成",
        gr.update(),
        header,
        candidates_to_rows(candidates),
        srt_update,
        new_state,
    )


def confirm_speaker(
    speaker_ids: list[str], state: dict,
    progress=gr.Progress(track_tqdm=True),
):
    """Phase-2: apply SPEAKER_FILTER after user picks speakers.
    Yields 9-tuple: (progress_group, speaker_group, results_group,
                     step_bar, status, results_header, review_table, srt_download, state)
    """
    _no_srt = gr.update(visible=False, value=None)
    srt_update = (
        gr.update(visible=True, value=state["srt_path"])
        if "srt_path" in state and Path(state["srt_path"]).exists()
        else _no_srt
    )

    if "transcript" not in state:
        yield (gr.update(), gr.update(), gr.update(), "", "⚠️ 请先上传视频并运行分析", "", [], _no_srt, state)
        return

    yield (
        gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
        step_html({1}, 3, True), "正在执行说话人筛选…", "", [], _no_srt, state,
    )

    if not speaker_ids:
        yield (gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
               step_html({1}, 2, False), "⚠️ 请至少选择一位说话人", "", [], _no_srt, state)
        return

    progress(0.5)

    transcript = [Segment(**s) for s in state["transcript"]]
    video_path = Path(state["video_path"])
    duration = state.get("duration")

    plan = build_plan_from_buttons(
        selected=[f for f in state["selected"] if f != "subtitle"],
        keywords=[k.strip() for k in (state.get("kw_text") or "").split(",") if k.strip()],
        keyword_before=state.get("kw_before", 3.0),
        keyword_after=state.get("kw_after", 5.0),
        time_start=state.get("t_start"),
        time_end=state.get("t_end"),
        speaker_ids=speaker_ids,
    )

    engine = RuleEngine()
    candidates = engine.execute(plan, transcript, video_path, duration)
    progress(1.0)

    new_state = {**state, "candidates": [c.model_dump() for c in candidates]}
    yield (
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
        step_html({1, 3}, 0, True),
        "完成",
        f"✅ 找到 **{len(candidates)}** 个候选片段",
        candidates_to_rows(candidates),
        srt_update,
        new_state,
    )


def export_raw(review_table, state: dict):
    """Cut approved segments in parallel (ffmpeg stream copy) and return as downloads."""
    if "candidates" not in state or "video_path" not in state:
        return gr.update(visible=False)

    candidates = [CandidateSegment(**c) for c in state["candidates"]]
    video_path = Path(state["video_path"])
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gradio passes Dataframe as a pandas DataFrame — convert to list of rows
    import pandas as pd
    rows = review_table.values.tolist() if isinstance(review_table, pd.DataFrame) else list(review_table or [])

    approved_indices: set[int] = set()
    for row in rows:
        try:
            if row[5]:
                approved_indices.add(int(row[0]) - 1)
        except (TypeError, ValueError, IndexError):
            pass

    # Build list of (index, out_path, start, end) for approved clips
    jobs: list[tuple[int, Path, float, float]] = []
    for i, seg in enumerate(candidates):
        if i not in approved_indices:
            continue
        out_name = f"{video_path.stem}_clip{i + 1}_{int(seg.start)}s-{int(seg.end)}s.mp4"
        jobs.append((i, output_dir / out_name, seg.start, seg.end))

    # Cut clips in parallel — ffmpeg stream copy is I/O-bound, not CPU-bound
    from concurrent.futures import ThreadPoolExecutor, as_completed
    futures_map = {}
    with ThreadPoolExecutor(max_workers=min(4, len(jobs) or 1)) as pool:
        for idx, out_path, start, end in jobs:
            f = pool.submit(cut_segment, video_path, out_path, start, end)
            futures_map[f] = out_path
        for f in as_completed(futures_map):
            f.result()  # propagate any ffmpeg errors

    # Return paths sorted by original clip index
    paths = [str(out_path) for _, out_path, _, _ in jobs]

    if not paths:
        return gr.update(visible=False)
    return gr.update(visible=True, value=paths)


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
                btn_subtitle = gr.Button("📄 导出字幕（SRT）", variant="secondary", size="sm")
                btn_silence = gr.Button("✂️ 去除静音", variant="secondary", size="sm")

            with gr.Group(visible=False) as keyword_params_group:
                kw_input = gr.Textbox(
                    label="关键词（逗号分隔）",
                    placeholder="竞品, 价格, 方案",
                )
                with gr.Row():
                    kw_before = gr.Number(label="片段前留（秒）", value=3, minimum=0, maximum=60, step=1)
                    kw_after = gr.Number(label="片段后留（秒）", value=5, minimum=0, maximum=60, step=1)

            gr.Markdown("---")
            instruction_input = gr.Textbox(
                label="自定义需求（AI 解析，少量费用）",
                placeholder="例如：找情绪激动的片段，或截取前五分钟…",
                lines=2,
            )
            start_btn = gr.Button("🚀 开始分析", variant="primary")

    # ── Region 2: progress (appears after start) ──────────────────────────────
    with gr.Group(visible=False) as progress_group:
        step_bar = gr.HTML()
        progress_status = gr.Markdown()

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
        srt_download = gr.File(
            label="📄 字幕文件（SRT）",
            file_count="single",
            visible=False,
        )
        export_btn = gr.Button("⬇️ 下载选中片段", variant="primary")
        export_files = gr.File(
            label="视频片段下载",
            file_count="multiple",
            visible=False,
        )

    # ── Event wiring ──────────────────────────────────────────────────────────

    _toggle_outputs = [
        selected_features,
        btn_keyword, btn_speaker, btn_subtitle, btn_silence,
        keyword_params_group,
        start_btn,
    ]

    btn_keyword.click(fn=lambda s: _toggle("keyword", s),
                      inputs=[selected_features], outputs=_toggle_outputs)
    btn_speaker.click(fn=lambda s: _toggle("speaker", s),
                      inputs=[selected_features], outputs=_toggle_outputs)
    btn_subtitle.click(fn=lambda s: _toggle("subtitle", s),
                       inputs=[selected_features], outputs=_toggle_outputs)
    btn_silence.click(fn=lambda s: _toggle("silence", s),
                      inputs=[selected_features], outputs=_toggle_outputs)

    _pipeline_outputs = [
        progress_group, speaker_group, results_group,
        step_bar, progress_status,
        speaker_selector,
        results_header, review_table,
        srt_download,
        session_state,
    ]

    start_btn.click(
        fn=run_pipeline,
        inputs=[
            video_input, selected_features,
            kw_input, kw_before, kw_after,
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
            srt_download,
            session_state,
        ],
    )

    export_btn.click(
        fn=export_raw,
        inputs=[review_table, session_state],
        outputs=[export_files],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        allowed_paths=[str(Path(settings.output_dir).resolve())],
        max_file_size="10gb",
    )
