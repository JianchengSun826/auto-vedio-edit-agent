from __future__ import annotations
from pathlib import Path
from typing import Literal
from models.edit_plan import Segment


def segments_to_srt(
    transcript: list[Segment],
    output_path: Path,
    lang: Literal["zh", "en"] = "zh",
) -> Path:
    """Write transcript segments to an SRT subtitle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for i, seg in enumerate(transcript, 1):
        start = _fmt(seg.start)
        end = _fmt(seg.end)
        if lang == "zh":
            text = (seg.text_zh or seg.text).strip()
        else:
            text = (seg.text_en or seg.text).strip()
        blocks.append(f"{i}\n{start} --> {end}\n{text}")
    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return output_path


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
