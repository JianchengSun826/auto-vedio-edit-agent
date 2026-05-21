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


def test_segment_has_optional_speaker():
    seg = Segment(start=0.0, end=5.0, text="hello")
    assert seg.speaker is None

    seg_with_speaker = Segment(start=0.0, end=5.0, text="hello", speaker="SPEAKER_00")
    assert seg_with_speaker.speaker == "SPEAKER_00"


def test_candidate_segment_has_optional_speaker():
    seg = CandidateSegment(id="1", start=0.0, end=5.0, text_preview="test")
    assert seg.speaker is None

    seg_with_speaker = CandidateSegment(
        id="2", start=0.0, end=5.0, text_preview="test", speaker="SPEAKER_01"
    )
    assert seg_with_speaker.speaker == "SPEAKER_01"


def test_speaker_filter_rule_type_exists():
    from models.edit_plan import RuleType
    assert RuleType.SPEAKER_FILTER == "speaker_filter"


def test_rule_has_speakers_field():
    rule = Rule(type=RuleType.KEYWORD_MATCH, keywords=["竞品"])
    assert rule.speakers == []

    rule_with_speakers = Rule(type=RuleType.SPEAKER_FILTER, speakers=["SPEAKER_00"])
    assert rule_with_speakers.speakers == ["SPEAKER_00"]


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
