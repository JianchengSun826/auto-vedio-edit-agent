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
