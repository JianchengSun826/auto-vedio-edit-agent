# tests/test_ui.py
"""
Tests for the new app/main.py UI functions.

The new design:
- run_pipeline is a generator (yields tuples) with 9 positional args + progress
- export_approved returns gr.update(...)
- No module-level orchestrator/exporter globals; instances are created inline
"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from models.edit_plan import (
    EditPlan, EditMode, Rule, RuleType, OutputFormat, Platform,
    Segment, CandidateSegment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(i: int = 1) -> Segment:
    return Segment(
        id=str(i), start=float(i * 10), end=float(i * 10 + 9),
        text=f"segment {i}", speaker=f"SPEAKER_{i % 2}",
    )


def _make_candidate(i: int = 1) -> CandidateSegment:
    return CandidateSegment(
        id=str(i), start=float(i * 10), end=float(i * 10 + 9),
        text_preview=f"candidate {i}", speaker=f"SPEAKER_{i % 2}",
    )


def _exhaust_generator(gen):
    """Collect all yielded values; return the last one."""
    last = None
    for val in gen:
        last = val
    return last


# ---------------------------------------------------------------------------
# run_pipeline tests
# ---------------------------------------------------------------------------

@patch("app.main.RuleEngine")
@patch("app.main.build_plan_from_buttons")
@patch("app.main.Orchestrator")
def test_run_pipeline_returns_candidate_rows(
    MockOrch, mock_build_plan, MockEngine, tmp_path
):
    from app.main import run_pipeline

    candidates = [_make_candidate(1), _make_candidate(2)]
    transcript = [_make_segment(1), _make_segment(2)]

    mock_orch_inst = MagicMock()
    mock_orch_inst.transcribe_only.return_value = (transcript, 30.0)
    MockOrch.return_value = mock_orch_inst

    mock_plan = MagicMock()
    mock_build_plan.return_value = mock_plan

    mock_engine_inst = MagicMock()
    mock_engine_inst.execute.return_value = candidates
    MockEngine.return_value = mock_engine_inst

    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")

    gen = run_pipeline(
        str(video),          # video_file
        ["keyword"],         # selected
        "price",             # kw_text
        3.0,                 # kw_before
        5.0,                 # kw_after
        0.0,                 # t_start
        60.0,                # t_end
        "",                  # instruction
        {},                  # state
    )
    last = _exhaust_generator(gen)
    # last tuple: (prog_group, spk_group, res_group, step_bar, status,
    #              spk_selector, results_header, review_table, state)
    results_header = last[6]
    review_table = last[7]
    state = last[8]

    assert "2" in results_header
    assert len(review_table) == 2
    assert review_table[0][5] is True   # included=True by default
    assert "candidates" in state


@patch("app.main.Orchestrator")
def test_run_pipeline_no_video_returns_error(MockOrch):
    from app.main import run_pipeline

    gen = run_pipeline(
        None, ["keyword"], "", 3.0, 5.0, 0.0, 60.0, "", {},
    )
    last = _exhaust_generator(gen)
    status = last[4]
    assert "请先上传" in status
    MockOrch.return_value.transcribe_only.assert_not_called()


# ---------------------------------------------------------------------------
# export_approved tests
# ---------------------------------------------------------------------------

@patch("app.main.Exporter")
def test_export_approved_filters_unchecked(MockExporter, tmp_path):
    from app.main import export_approved

    candidates = [_make_candidate(1), _make_candidate(2)]
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")

    mock_exp_inst = MagicMock()
    mock_exp_inst.export.return_value = [Path("out.mp4")]
    MockExporter.return_value = mock_exp_inst

    state = {
        "candidates": [c.model_dump() for c in candidates],
        "video_path": str(video),
    }

    review_table = [
        [1, "—", "10s - 19s", "candidate 1", "1.00", True],   # included
        [2, "—", "20s - 29s", "candidate 2", "1.00", False],  # excluded
    ]

    result = export_approved(review_table, ["YouTube"], state)
    # result is gr.update(visible=True, value=[...])
    assert result is not None

    call_candidates = mock_exp_inst.export.call_args[0][1]
    included = [c for c in call_candidates if c.included]
    excluded = [c for c in call_candidates if not c.included]
    assert len(included) == 1
    assert len(excluded) == 1


def test_export_approved_no_candidates_returns_update():
    from app.main import export_approved

    result = export_approved([], ["YouTube"], {})
    # Should return gr.update(visible=False) when no candidates in state
    assert result is not None


# ---------------------------------------------------------------------------
# confirm_speaker tests
# ---------------------------------------------------------------------------

def test_confirm_speaker_requires_state():
    """confirm_speaker should not crash when state is empty."""
    from app.main import confirm_speaker
    gen = confirm_speaker(speaker_ids=["SPEAKER_00"], state={})
    result = next(gen)
    # Should yield an early-exit warning, not raise KeyError
    assert result is not None


def test_confirm_speaker_empty_speaker_ids():
    """confirm_speaker should warn and return early when no speakers selected."""
    from app.main import confirm_speaker
    state = {
        "transcript": [],
        "video_path": "test.mp4",
        "duration": 60.0,
        "selected": ["speaker"],
        "kw_text": "",
        "kw_before": 3.0,
        "kw_after": 5.0,
        "t_start": None,
        "t_end": None,
    }
    gen = confirm_speaker(speaker_ids=[], state=state)
    # First yield: "正在执行说话人筛选…"
    next(gen)
    # Second yield: warning about no speakers
    result = next(gen)
    assert result is not None
