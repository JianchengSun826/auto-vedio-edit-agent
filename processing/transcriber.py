from pathlib import Path
from faster_whisper import WhisperModel
from models.edit_plan import Segment
from config.settings import settings

# Chunk size and threshold for splitting long videos
CHUNK_DURATION = 1800   # 30 minutes per chunk
CHUNK_THRESHOLD = 7200  # only chunk videos longer than 2 hours


class Transcriber:
    def __init__(self, model_size: str | None = None, device: str | None = None):
        size = model_size or settings.whisper_model
        dev = device or settings.whisper_device
        self._model = WhisperModel(size, device=dev, compute_type="int8")

    def transcribe(self, video_path: Path) -> list[Segment]:
        """Transcribe a video file. Returns list of Segment objects."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        duration = self._get_duration(video_path)
        if duration and duration > CHUNK_THRESHOLD:
            return self._transcribe_chunked(video_path, duration)

        return self._transcribe_single(video_path)

    def _transcribe_single(self, video_path: Path) -> list[Segment]:
        segments_iter, _ = self._model.transcribe(
            str(video_path),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        return [
            Segment(start=seg.start, end=seg.end, text=seg.text.strip())
            for seg in segments_iter
            if seg.text.strip()
        ]

    def _transcribe_chunked(self, video_path: Path, duration: float) -> list[Segment]:
        """Split video into 30-min chunks, transcribe each, merge with offset."""
        import tempfile, subprocess
        all_segments = []
        cursor = 0.0
        chunk_index = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            while cursor < duration:
                chunk_end = min(cursor + CHUNK_DURATION, duration)
                chunk_path = Path(tmpdir) / f"chunk_{chunk_index}.mp4"
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(cursor), "-to", str(chunk_end),
                     "-i", str(video_path), "-c", "copy", str(chunk_path)],
                    capture_output=True, check=True,
                )
                segments = self._transcribe_single(chunk_path)
                for seg in segments:
                    all_segments.append(Segment(
                        start=seg.start + cursor,
                        end=seg.end + cursor,
                        text=seg.text,
                    ))
                cursor = chunk_end
                chunk_index += 1
        return all_segments

    def _get_duration(self, video_path: Path) -> float | None:
        import subprocess, re
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1", str(video_path)],
                capture_output=True, text=True,
            )
            match = re.search(r"duration=([\d.]+)", result.stdout)
            return float(match.group(1)) if match else None
        except Exception:
            return None
