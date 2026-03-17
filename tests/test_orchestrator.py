import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from models.edit_plan import EditPlan, EditMode, Rule, RuleType, OutputFormat, Platform, Segment
from agent.orchestrator import Orchestrator


FAKE_TRANSCRIPT = [
    Segment(start=0.0, end=5.0, text="竞品价格很高"),
    Segment(start=5.0, end=10.0, text="我们更便宜"),
]

FAKE_PLAN = EditPlan(
    mode=EditMode.HIGHLIGHT_EXTRACTION,
    rules=[Rule(type=RuleType.KEYWORD_MATCH, keywords=["竞品"])],
    output_formats=[OutputFormat(platform=Platform.DOUYIN)],
)


@patch("agent.orchestrator.IntentParser")
@patch("agent.orchestrator.Transcriber")
def test_orchestrator_run_returns_candidates(mock_transcriber_cls, mock_parser_cls, tmp_path):
    # Arrange
    mock_transcriber = MagicMock()
    mock_transcriber_cls.return_value = mock_transcriber
    mock_transcriber.transcribe.return_value = FAKE_TRANSCRIPT

    mock_parser = MagicMock()
    mock_parser_cls.return_value = mock_parser
    mock_parser.parse.return_value = FAKE_PLAN

    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")

    orch = Orchestrator()
    result = orch.run(video_path=video, user_instruction="提取竞品片段")

    assert result.plan is not None
    assert result.candidates is not None
    assert len(result.candidates) > 0
    assert result.transcript == FAKE_TRANSCRIPT


@patch("agent.orchestrator.IntentParser")
@patch("agent.orchestrator.Transcriber")
def test_orchestrator_passes_transcript_to_parser(mock_transcriber_cls, mock_parser_cls, tmp_path):
    mock_transcriber = MagicMock()
    mock_transcriber_cls.return_value = mock_transcriber
    mock_transcriber.transcribe.return_value = FAKE_TRANSCRIPT

    mock_parser = MagicMock()
    mock_parser_cls.return_value = mock_parser
    mock_parser.parse.return_value = FAKE_PLAN

    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")

    orch = Orchestrator()
    orch.run(video_path=video, user_instruction="找竞品")

    call_kwargs = mock_parser.parse.call_args
    assert call_kwargs.kwargs["transcript"] is not None or call_kwargs.args[1] is not None
