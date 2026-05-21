from __future__ import annotations
from pathlib import Path
from models.edit_plan import Segment
from processing.subtitle import segments_to_srt


SEGS = [
    Segment(start=0.0, end=3.0, text="你好", text_zh="你好", text_en="Hello"),
    Segment(start=3.0, end=6.0, text="再见", text_zh="再见", text_en="Goodbye"),
]


def test_srt_zh_uses_text_zh(tmp_path):
    out = tmp_path / "test_zh.srt"
    segments_to_srt(SEGS, out, lang="zh")
    content = out.read_text(encoding="utf-8")
    assert "你好" in content
    assert "再见" in content
    assert "Hello" not in content


def test_srt_en_uses_text_en(tmp_path):
    out = tmp_path / "test_en.srt"
    segments_to_srt(SEGS, out, lang="en")
    content = out.read_text(encoding="utf-8")
    assert "Hello" in content
    assert "Goodbye" in content
    assert "你好" not in content


def test_srt_fallback_to_text_when_lang_field_none(tmp_path):
    segs = [Segment(start=0.0, end=3.0, text="fallback text")]
    out = tmp_path / "fallback.srt"
    segments_to_srt(segs, out, lang="zh")
    content = out.read_text(encoding="utf-8")
    assert "fallback text" in content


def test_srt_default_lang_is_zh(tmp_path):
    out = tmp_path / "default.srt"
    segments_to_srt(SEGS, out)
    content = out.read_text(encoding="utf-8")
    assert "你好" in content
