import gradio as gr
from pathlib import Path
from agent.orchestrator import Orchestrator
from processing.exporter import Exporter
from models.edit_plan import CandidateSegment, OutputFormat, Platform
from config.settings import settings

# Module-level singletons — set to None initially so they can be patched in
# tests before the first function call, and lazily initialized in production.
orchestrator = None  # type: ignore
exporter = None  # type: ignore


def _ensure_initialized():
    """Lazy-initialize singletons if not already set (e.g., patched in tests)."""
    global orchestrator, exporter
    if orchestrator is None:
        orchestrator = Orchestrator()
    if exporter is None:
        exporter = Exporter(output_dir=settings.output_dir)


def run_pipeline(video_file, instruction: str, session_state: dict):
    """Step 1: Transcribe + parse + generate candidates."""
    if video_file is None:
        return "请上传视频文件", [], None, session_state

    # Only initialize if not already set (tests patch before calling)
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

    # Only initialize if not already set (tests patch before calling)
    if exporter is None:
        _ensure_initialized()

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
