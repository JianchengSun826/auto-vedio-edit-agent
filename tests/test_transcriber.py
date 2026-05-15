from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from processing.transcriber import Transcriber
from models.edit_plan import Segment

MOCK_TRANSCRIBE_RESULT = {
    "segments": [
        {"start": 0.0, "end": 3.5, "text": " Hello world"},
        {"start": 3.5, "end": 7.0, "text": " This is a test"},
    ],
    "language": "en",
}

MOCK_ALIGN_RESULT = {
    "segments": [
        {"start": 0.0, "end": 3.5, "text": " Hello world"},
        {"start": 3.5, "end": 7.0, "text": " This is a test"},
    ],
}

MOCK_DIARIZED_RESULT = {
    "segments": [
        {"start": 0.0, "end": 3.5, "text": " Hello world", "speaker": "SPEAKER_00"},
        {"start": 3.5, "end": 7.0, "text": " This is a test", "speaker": "SPEAKER_01"},
    ],
}


def _make_transcriber_no_init(hf_token=None, enable_diarization=False):
    """Build a Transcriber without calling __init__ (avoids model download)."""
    t = Transcriber.__new__(Transcriber)
    t._device = "cpu"
    t._model = MagicMock()
    t._hf_token = hf_token
    t._enable_diarization = enable_diarization
    return t


@patch("processing.transcriber.whisperx.load_align_model")
@patch("processing.transcriber.whisperx.align")
@patch("processing.transcriber.whisperx.load_audio")
def test_transcribe_returns_segments(mock_load_audio, mock_align, mock_load_align_model, tmp_path):
    mock_load_audio.return_value = MagicMock()
    mock_load_align_model.return_value = (MagicMock(), MagicMock())
    mock_align.return_value = MOCK_ALIGN_RESULT

    t = _make_transcriber_no_init()
    t._model.transcribe.return_value = MOCK_TRANSCRIBE_RESULT

    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"fake")

    result = t.transcribe(fake_video)

    assert len(result) == 2
    assert isinstance(result[0], Segment)
    assert result[0].text == "Hello world"   # leading space stripped
    assert result[0].start == 0.0
    assert result[1].end == 7.0
    assert result[0].speaker is None         # no diarization


@patch("processing.transcriber.whisperx.assign_word_speakers")
@patch("processing.transcriber.whisperx.DiarizationPipeline")
@patch("processing.transcriber.whisperx.load_align_model")
@patch("processing.transcriber.whisperx.align")
@patch("processing.transcriber.whisperx.load_audio")
def test_transcribe_with_diarization(
    mock_load_audio, mock_align, mock_load_align_model, mock_diarize_cls, mock_assign, tmp_path
):
    mock_load_audio.return_value = MagicMock()
    mock_load_align_model.return_value = (MagicMock(), MagicMock())
    mock_align.return_value = MOCK_ALIGN_RESULT
    mock_diarize_cls.return_value = MagicMock()
    mock_assign.return_value = MOCK_DIARIZED_RESULT

    t = _make_transcriber_no_init(hf_token="hf_fake", enable_diarization=True)
    t._model.transcribe.return_value = MOCK_TRANSCRIBE_RESULT

    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"fake")

    result = t.transcribe(fake_video)

    mock_diarize_cls.assert_called_once_with(use_auth_token="hf_fake", device="cpu")
    assert result[0].speaker == "SPEAKER_00"
    assert result[1].speaker == "SPEAKER_01"


@patch("processing.transcriber.whisperx.load_align_model")
@patch("processing.transcriber.whisperx.align")
@patch("processing.transcriber.whisperx.load_audio")
def test_transcribe_empty_video_returns_empty(mock_load_audio, mock_align, mock_load_align_model, tmp_path):
    mock_load_audio.return_value = MagicMock()
    mock_load_align_model.return_value = (MagicMock(), MagicMock())
    mock_align.return_value = {"segments": []}

    t = _make_transcriber_no_init()
    t._model.transcribe.return_value = {"segments": [], "language": "en"}

    fake_video = tmp_path / "silent.mp4"
    fake_video.write_bytes(b"fake")

    result = t.transcribe(fake_video)
    assert result == []


def test_transcribe_missing_file_raises():
    t = _make_transcriber_no_init()
    with pytest.raises(FileNotFoundError):
        t.transcribe(Path("/nonexistent/video.mp4"))
