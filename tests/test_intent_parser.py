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


def test_preferences_injected_into_system_prompt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prefs = tmp_path / "USER_PREFERENCES.md"
    prefs.write_text("## 偏好\n- 默认留白 5 秒", encoding="utf-8")

    with patch("agent.intent_parser.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=SAMPLE_LLM_RESPONSE)]
        )
        parser = IntentParser()
        parser.parse(user_instruction="test", transcript=[])

    call_kwargs = mock_client.messages.create.call_args
    system_prompt = call_kwargs.kwargs.get("system") or ""
    if not system_prompt:
        system_prompt = call_kwargs[1].get("system", "")
    assert "## 偏好" in system_prompt
    assert "默认留白 5 秒" in system_prompt


def test_no_preferences_file_uses_base_prompt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # directory has no USER_PREFERENCES.md

    with patch("agent.intent_parser.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=SAMPLE_LLM_RESPONSE)]
        )
        parser = IntentParser()
        parser.parse(user_instruction="test", transcript=[])

    call_kwargs = mock_client.messages.create.call_args
    system_prompt = call_kwargs[1].get("system", "")
    assert "[用户剪辑偏好]" not in system_prompt


def test_transcript_includes_speaker_labels():
    transcript = [
        {"start": 0.0, "end": 5.0, "text": "hello", "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 10.0, "text": "world", "speaker": None},
    ]

    with patch("agent.intent_parser.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=SAMPLE_LLM_RESPONSE)]
        )
        parser = IntentParser()
        parser.parse(user_instruction="test", transcript=transcript)

    call_kwargs = mock_client.messages.create.call_args
    user_message = call_kwargs[1]["messages"][0]["content"]
    assert "SPEAKER_00:" in user_message
    # Second segment has no speaker — should not show a label
    lines = user_message.split("\n")
    speaker_none_line = next((l for l in lines if "world" in l), "")
    assert "SPEAKER" not in speaker_none_line
