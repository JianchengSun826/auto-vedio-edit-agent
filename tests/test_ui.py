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
