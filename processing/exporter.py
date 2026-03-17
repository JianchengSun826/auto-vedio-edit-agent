import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from models.edit_plan import OutputFormat, Platform, CandidateSegment
from processing.ffmpeg_utils import cut_segment, transcode_for_platform


@dataclass
class PlatformSpec:
    ratio: str
    width: int
    height: int
    max_duration_sec: Optional[int]
    codec: str = "libx264"

    @classmethod
    def for_platform(cls, platform: Platform) -> "PlatformSpec":
        specs = {
            Platform.DOUYIN:   cls("9:16",  1080, 1920, 60),
            Platform.WECHAT:   cls("9:16",  1080, 1920, 600),
            Platform.BILIBILI: cls("16:9",  1920, 1080, None),
            Platform.YOUTUBE:  cls("16:9",  1920, 1080, None),
        }
        return specs[platform]


class Exporter:
    def __init__(self, output_dir: Path | str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        src: Path,
        candidates: list[CandidateSegment],
        formats: list[OutputFormat],
    ) -> list[Path]:
        """Export approved candidate segments for all requested platforms."""
        results = []
        included = [c for c in candidates if c.included]

        for fmt in formats:
            spec = PlatformSpec.for_platform(fmt.platform)
            needs_transcode = fmt.ratio != "16:9"  # source assumed 16:9; transcode for vertical
            for candidate in included:
                parts = self._split_if_needed(candidate.start, candidate.end, spec)
                for i, (part_start, part_end) in enumerate(parts):
                    suffix = f"_part{i+1}" if len(parts) > 1 else ""
                    out_name = f"{src.stem}_{fmt.platform.value}_{candidate.id}{suffix}.mp4"
                    cut_path = self.output_dir / f"_tmp_{out_name}"
                    out_path = self.output_dir / out_name
                    cut_segment(src, cut_path, start=part_start, end=part_end)
                    if needs_transcode:
                        transcode_for_platform(cut_path, out_path, spec.width, spec.height)
                        cut_path.unlink(missing_ok=True)
                    else:
                        cut_path.rename(out_path)
                    results.append(out_path)

        return results

    def _split_if_needed(
        self, start: float, end: float, spec: PlatformSpec
    ) -> list[tuple[float, float]]:
        """Split segment into chunks if it exceeds platform max duration."""
        if spec.max_duration_sec is None:
            return [(start, end)]

        parts = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + spec.max_duration_sec, end)
            parts.append((cursor, chunk_end))
            cursor = chunk_end
        return parts
