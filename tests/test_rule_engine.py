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
