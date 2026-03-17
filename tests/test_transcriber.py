import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from processing.transcriber import Transcriber
from models.edit_plan import Segment


def make_mock_segment(start, end, text):
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    return seg


@patch("processing.transcriber.WhisperModel")
def test_transcribe_returns_segments(mock_model_cls, tmp_path):
    mock_model = MagicMock()
    mock_model_cls.return_value = mock_model
    mock_segments = [
        make_mock_segment(0.0, 3.5, " Hello world"),
        make_mock_segment(3.5, 7.0, " This is a test"),
    ]
    mock_model.transcribe.return_value = (iter(mock_segments), MagicMock())

    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"fake")

    transcriber = Transcriber(model_size="tiny")
    result = transcriber.transcribe(fake_video)

    assert len(result) == 2
    assert isinstance(result[0], Segment)
    assert result[0].text == "Hello world"  # leading space stripped
    assert result[0].start == 0.0
    assert result[1].end == 7.0


@patch("processing.transcriber.WhisperModel")
def test_transcribe_empty_video_returns_empty(mock_model_cls, tmp_path):
    mock_model = MagicMock()
    mock_model_cls.return_value = mock_model
    mock_model.transcribe.return_value = (iter([]), MagicMock())

    fake_video = tmp_path / "silent.mp4"
    fake_video.write_bytes(b"fake")

    transcriber = Transcriber(model_size="tiny")
    result = transcriber.transcribe(fake_video)

    assert result == []


def test_transcribe_missing_file_raises():
    transcriber = Transcriber.__new__(Transcriber)  # skip __init__
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe(Path("/nonexistent/video.mp4"))
