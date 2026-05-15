from __future__ import annotations
import whisperx
from pathlib import Path
from typing import Optional
from models.edit_plan import Segment
from config.settings import settings

CHUNK_DURATION = 1800   # 30 minutes per chunk
CHUNK_THRESHOLD = 7200  # only chunk videos longer than 2 hours


class Transcriber:
    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None):
        size = model_size or settings.whisper_model
        dev = device or settings.whisper_device
        if dev == "auto":
            try:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                dev = "cpu"
        self._device = dev
        self._model = whisperx.load_model(size, self._device, compute_type="int8")
        self._hf_token = settings.hf_token
        self._enable_diarization = settings.enable_diarization

    def transcribe(self, video_path: Path) -> list[Segment]:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        duration = self._get_duration(video_path)
        if duration and duration > CHUNK_THRESHOLD:
            return self._transcribe_chunked(video_path, duration)
        return self._transcribe_single(video_path)

    def _transcribe_single(self, video_path: Path) -> list[Segment]:
        audio = whisperx.load_audio(str(video_path))
        result = self._model.transcribe(audio, batch_size=16)

        # Word-level alignment — best-effort, skip on failure
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=self._device
            )
            result = whisperx.align(
                result["segments"], model_a, metadata, audio, self._device,
                return_char_alignments=False,
            )
        except Exception:
            pass

        # Speaker diarization — best-effort, requires HF token
        if self._enable_diarization and self._hf_token:
            try:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=self._hf_token, device=self._device
                )
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception:
                pass

        return [
            Segment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                speaker=seg.get("speaker"),
            )
            for seg in result["segments"]
            if seg.get("text", "").strip()
        ]

    def _transcribe_chunked(self, video_path: Path, duration: float) -> list[Segment]:
        import tempfile
        import subprocess
        all_segments: list[Segment] = []
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
                for seg in self._transcribe_single(chunk_path):
                    all_segments.append(Segment(
                        start=seg.start + cursor,
                        end=seg.end + cursor,
                        text=seg.text,
                        speaker=seg.speaker,
                    ))
                cursor = chunk_end
                chunk_index += 1
        return all_segments

    def _get_duration(self, video_path: Path) -> Optional[float]:
        import subprocess
        import re
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
