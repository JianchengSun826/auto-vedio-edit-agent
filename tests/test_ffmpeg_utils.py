import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from processing.ffmpeg_utils import (
    cut_segment, concat_segments, detect_silence, get_video_duration
)


@patch("processing.ffmpeg_utils.subprocess.run")
def test_cut_segment_calls_ffmpeg(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0, stderr="")
    src = tmp_path / "input.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"

    cut_segment(src, out, start=10.0, end=30.0)

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    # -ss must appear before -i (fast input seek)
    assert cmd.index("-ss") < cmd.index("-i")
    assert "10.0" in cmd
    # -t (duration) must appear after -i
    assert "-t" in cmd
    assert "20.0" in cmd   # duration = 30.0 - 10.0
    assert "-c" in cmd
    assert "copy" in cmd


@patch("processing.ffmpeg_utils.subprocess.run")
def test_cut_segment_raises_on_ffmpeg_error(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=1, stderr="Error: codec not found")
    src = tmp_path / "input.mp4"
    src.write_bytes(b"fake")

    with pytest.raises(RuntimeError, match="FFmpeg error"):
        cut_segment(src, tmp_path / "out.mp4", start=0.0, end=5.0)


@patch("processing.ffmpeg_utils.subprocess.run")
def test_detect_silence_parses_output(mock_run, tmp_path):
    stderr_output = (
        "[silencedetect] silence_start: 2.5\n"
        "[silencedetect] silence_end: 5.0 | silence_duration: 2.5\n"
        "[silencedetect] silence_start: 10.0\n"
        "[silencedetect] silence_end: 12.0 | silence_duration: 2.0\n"
    )
    mock_run.return_value = MagicMock(returncode=0, stderr=stderr_output)
    src = tmp_path / "video.mp4"
    src.write_bytes(b"fake")

    silences = detect_silence(src)

    assert len(silences) == 2
    assert silences[0] == (2.5, 5.0)
    assert silences[1] == (10.0, 12.0)


@patch("processing.ffmpeg_utils.subprocess.run")
def test_get_video_duration(mock_run, tmp_path):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="duration=125.4\n",
        stderr=""
    )
    src = tmp_path / "video.mp4"
    src.write_bytes(b"fake")

    duration = get_video_duration(src)
    assert duration == pytest.approx(125.4)
