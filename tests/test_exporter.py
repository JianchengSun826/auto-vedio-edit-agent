import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from models.edit_plan import OutputFormat, Platform, CandidateSegment
from processing.exporter import Exporter, PlatformSpec


def make_candidate(id, start, end):
    return CandidateSegment(id=id, start=start, end=end, text_preview="test")


def test_platform_spec_douyin():
    spec = PlatformSpec.for_platform(Platform.DOUYIN)
    assert spec.ratio == "9:16"
    assert spec.max_duration_sec == 60
    assert spec.width == 1080
    assert spec.height == 1920


def test_platform_spec_youtube():
    spec = PlatformSpec.for_platform(Platform.YOUTUBE)
    assert spec.ratio == "16:9"
    assert spec.max_duration_sec is None
    assert spec.width == 1920
    assert spec.height == 1080


@patch("processing.exporter.transcode_for_platform")
@patch("processing.exporter.cut_segment")
def test_export_single_segment_youtube(mock_cut, mock_transcode, tmp_path):
    src = tmp_path / "input.mp4"
    src.write_bytes(b"fake")
    candidates = [make_candidate("1", 10.0, 40.0)]
    fmt = OutputFormat(platform=Platform.YOUTUBE)

    # Mock cut_segment to create the temp file
    def mock_cut_side_effect(src, dest, start, end):
        dest.write_bytes(b"cut_result")

    mock_cut.side_effect = mock_cut_side_effect

    exporter = Exporter(output_dir=tmp_path)
    results = exporter.export(src, candidates, [fmt])

    assert len(results) == 1
    mock_cut.assert_called_once()
    mock_transcode.assert_not_called()  # YouTube uses stream copy (same ratio assumed)


@patch("processing.exporter.transcode_for_platform")
@patch("processing.exporter.cut_segment")
def test_export_douyin_auto_splits_long_segment(mock_cut, mock_transcode, tmp_path):
    src = tmp_path / "input.mp4"
    src.write_bytes(b"fake")
    # 3-minute segment should split into 3 x 60s clips for Douyin
    candidates = [make_candidate("1", 0.0, 180.0)]
    fmt = OutputFormat(platform=Platform.DOUYIN)

    # Mock cut_segment to create the temp file
    def mock_cut_side_effect(src, dest, start, end):
        dest.write_bytes(b"cut_result")

    mock_cut.side_effect = mock_cut_side_effect

    exporter = Exporter(output_dir=tmp_path)
    results = exporter.export(src, candidates, [fmt])

    assert len(results) == 3
    assert mock_cut.call_count == 3
