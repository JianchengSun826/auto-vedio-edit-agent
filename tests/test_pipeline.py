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


# ── step_html ────────────────────────────────────────────────────────────────

def test_step_html_done_contains_checkmark():
    from app.pipeline import step_html
    html = step_html(done={1}, active=2, skip_llm=False)
    assert "✓" in html

def test_step_html_active_contains_spinner():
    from app.pipeline import step_html
    html = step_html(done={1}, active=2, skip_llm=False)
    assert "⟳" in html

def test_step_html_skip_llm_grays_step2():
    from app.pipeline import step_html
    # When skip_llm=True, step 2 should show the "—" suffix
    html = step_html(done={1}, active=3, skip_llm=True)
    assert " —" in html


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
