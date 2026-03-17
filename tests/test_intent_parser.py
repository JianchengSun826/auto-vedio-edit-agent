# tests/test_intent_parser.py
import pytest
from unittest.mock import MagicMock, patch
from models.edit_plan import EditPlan, EditMode, RuleType, Platform
from agent.intent_parser import IntentParser


SAMPLE_TRANSCRIPT = [
    {"start": 0.0, "end": 5.0, "text": "今天我们聊聊竞品的价格策略"},
    {"start": 5.0, "end": 10.0, "text": "我们的产品比竞品便宜30%"},
]

SAMPLE_LLM_RESPONSE = '''{
  "mode": "highlight_extraction",
  "rules": [
    {
      "type": "keyword_match",
      "keywords": ["竞品", "价格"],
      "padding_before_sec": 3,
      "padding_after_sec": 5,
      "min_duration_sec": 5
    }
  ],
  "output_formats": [
    {
      "platform": "douyin",
      "ratio": "9:16",
      "max_duration_sec": 60,
      "resolution": "1080p"
    }
  ],
  "segment_count_hint": 3
}'''


@patch("agent.intent_parser.anthropic.Anthropic")
def test_parse_returns_edit_plan(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=SAMPLE_LLM_RESPONSE)]
    )

    parser = IntentParser()
    plan = parser.parse(
        user_instruction="提取所有提到竞品价格的片段",
        transcript=SAMPLE_TRANSCRIPT,
    )

    assert isinstance(plan, EditPlan)
    assert plan.mode == EditMode.HIGHLIGHT_EXTRACTION
    assert plan.rules[0].type == RuleType.KEYWORD_MATCH
    assert "竞品" in plan.rules[0].keywords
    assert plan.output_formats[0].platform == Platform.DOUYIN


@patch("agent.intent_parser.anthropic.Anthropic")
def test_parse_retries_on_invalid_json(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.side_effect = [
        MagicMock(content=[MagicMock(text="not json at all")]),
        MagicMock(content=[MagicMock(text=SAMPLE_LLM_RESPONSE)]),
    ]

    parser = IntentParser()
    plan = parser.parse(
        user_instruction="提取竞品片段",
        transcript=SAMPLE_TRANSCRIPT,
    )

    assert isinstance(plan, EditPlan)
    assert mock_client.messages.create.call_count == 2


@patch("agent.intent_parser.anthropic.Anthropic")
def test_parse_raises_after_two_failures(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="invalid json")]
    )

    parser = IntentParser()
    with pytest.raises(ValueError, match="Failed to parse"):
        parser.parse(user_instruction="test", transcript=[])
