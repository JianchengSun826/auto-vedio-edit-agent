from __future__ import annotations
import platform
from pathlib import Path
from typing import Optional
from models.edit_plan import Segment
from config.settings import settings

CHUNK_DURATION = 600    # 10 minutes per chunk — keeps peak RAM manageable
CHUNK_THRESHOLD = 1800  # chunk anything longer than 30 minutes

_MLX_MODEL_MAP = {
    "tiny":     "mlx-community/whisper-tiny-mlx",
    "base":     "mlx-community/whisper-base-mlx",
    "small":    "mlx-community/whisper-small-mlx",
    "medium":   "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class Transcriber:
    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None):
        import logging
        _log = logging.getLogger(__name__)

        size = model_size or settings.whisper_model
        dev = device or settings.whisper_device

        if dev == "auto":
            if _is_apple_silicon():
                try:
                    import mlx_whisper  # noqa: F401
                    self._backend = "mlx"
                    dev = "mps"
                except ImportError:
                    _log.warning("mlx_whisper not installed, falling back to whisperx on CPU")
                    self._backend = "whisperx"
                    dev = "cpu"
            else:
                self._backend = "whisperx"
                try:
                    import torch
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    dev = "cpu"
        else:
            self._backend = "whisperx"

        self._device = dev
        self._size = size
        self._hf_token = settings.hf_token
        self._enable_diarization = settings.enable_diarization

        if self._backend == "mlx":
            _log.info("Apple Silicon — mlx-whisper, model: %s (Neural Engine)", size)
        else:
            import whisperx
            _log.info("whisperx backend, device: %s, model: %s", dev, size)
            self._model = whisperx.load_model(size, self._device, compute_type="int8")

    def transcribe(self, video_path: Path, diarize: bool = False) -> list[Segment]:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        duration = self._get_duration(video_path)
        if duration and duration > CHUNK_THRESHOLD:
            return self._transcribe_chunked(video_path, duration, diarize=diarize)
        return self._transcribe_single(video_path, diarize=diarize)

    def _transcribe_single(self, video_path: Path, diarize: bool = False) -> list[Segment]:
        if self._backend == "mlx":
            return self._transcribe_single_mlx(video_path, diarize=diarize)
        return self._transcribe_single_whisperx(video_path, diarize=diarize)

    def _transcribe_single_mlx(self, video_path: Path, diarize: bool = False) -> list[Segment]:
        import mlx_whisper
        mlx_model = _MLX_MODEL_MAP.get(self._size, _MLX_MODEL_MAP["medium"])
        result = mlx_whisper.transcribe(
            str(video_path),
            path_or_hf_repo=mlx_model,
            word_timestamps=False,
            verbose=False,
        )

        if diarize and self._enable_diarization and self._hf_token:
            try:
                from pyannote.audio import Pipeline
                import torch
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    use_auth_token=self._hf_token,
                )
                pipeline = pipeline.to(torch.device(self._device))
                diarization = pipeline(str(video_path))
                for seg in result["segments"]:
                    mid = (seg["start"] + seg["end"]) / 2
                    seg["speaker"] = None
                    for turn, _, label in diarization.itertracks(yield_label=True):
                        if turn.start <= mid <= turn.end:
                            seg["speaker"] = label
                            break
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Diarization failed: %s", e)

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

    def _transcribe_single_whisperx(self, video_path: Path, diarize: bool = False) -> list[Segment]:
        import whisperx
        audio = whisperx.load_audio(str(video_path))
        result = self._model.transcribe(audio, batch_size=16)

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

        if diarize and self._enable_diarization and self._hf_token:
            try:
                from whisperx.diarize import DiarizationPipeline
                diarize_model = DiarizationPipeline(
                    token=self._hf_token, device=self._device
                )
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Speaker diarization failed: %s", e)

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

    def _transcribe_chunked(self, video_path: Path, duration: float, diarize: bool = False) -> list[Segment]:
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
                for seg in self._transcribe_single(chunk_path, diarize=diarize):
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
