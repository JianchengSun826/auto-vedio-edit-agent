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

    def transcribe(self, video_path: Path, diarize: bool = False,
                   language: Optional[str] = None, task: str = "transcribe") -> list[Segment]:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        duration = self._get_duration(video_path)
        if duration and duration > CHUNK_THRESHOLD:
            return self._transcribe_chunked(video_path, duration, diarize=diarize,
                                            language=language, task=task)
        return self._transcribe_single(video_path, diarize=diarize, language=language, task=task)

    def _transcribe_single(self, video_path: Path, diarize: bool = False,
                           language: Optional[str] = None, task: str = "transcribe") -> list[Segment]:
        if self._backend == "mlx":
            return self._transcribe_single_mlx(video_path, diarize=diarize, language=language, task=task)
        return self._transcribe_single_whisperx(video_path, diarize=diarize, language=language, task=task)

    def _transcribe_single_mlx(self, video_path: Path, diarize: bool = False,
                               language: Optional[str] = None, task: str = "transcribe") -> list[Segment]:
        import mlx_whisper
        import logging
        import tempfile
        import subprocess
        _log = logging.getLogger(__name__)
        mlx_model = _MLX_MODEL_MAP.get(self._size, _MLX_MODEL_MAP["medium"])

        # Extract to a temp 16 kHz mono WAV so mlx_whisper gets clean audio
        # regardless of source container, codec, or non-ASCII path characters.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _tmp:
            audio_path = _tmp.name
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_path),
                 "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", audio_path],
                capture_output=True,
            )
            if proc.returncode != 0:
                _log.warning("音频提取失败 (exit %d): %s",
                             proc.returncode, proc.stderr.decode(errors="replace")[-300:])
            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=mlx_model,
                word_timestamps=False,
                verbose=False,
                no_speech_threshold=1.0,
                logprob_threshold=None,
                condition_on_previous_text=False,
                language=language,
                task=task,
            )
        finally:
            Path(audio_path).unlink(missing_ok=True)

        lang = result.get("language", "未知")
        n_raw = len(result.get("segments", []))
        _log.info("检测到语言: %s，原始片段数: %d", lang, n_raw)

        if diarize and self._enable_diarization and self._hf_token:
            try:
                from pyannote.audio import Pipeline
                import torch
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    token=self._hf_token,
                )
                # pyannote does not support MPS — always run on CPU
                pipeline = pipeline.to(torch.device("cpu"))
                diarization = pipeline(str(video_path))
                # Some pyannote versions return a namedtuple/dataclass instead of
                # a bare Annotation. Walk the object's attributes to find the
                # Annotation (which has itertracks).
                if not hasattr(diarization, "itertracks"):
                    found = None
                    for attr in vars(diarization) if hasattr(diarization, "__dict__") else []:
                        val = getattr(diarization, attr, None)
                        if hasattr(val, "itertracks"):
                            found = val
                            break
                    if found is None and hasattr(diarization, "_fields"):
                        for attr in diarization._fields:
                            val = getattr(diarization, attr, None)
                            if hasattr(val, "itertracks"):
                                found = val
                                break
                    if found is None:
                        raise RuntimeError(
                            f"Cannot find Annotation in {type(diarization).__name__}; "
                            f"fields={getattr(diarization, '_fields', dir(diarization))}"
                        )
                    diarization = found
                n_speakers = len(set(label for _, _, label in diarization.itertracks(yield_label=True)))
                _log.info("说话人分离完成，检测到 %d 位说话人", n_speakers)
                for seg in result["segments"]:
                    mid = (seg["start"] + seg["end"]) / 2
                    seg["speaker"] = None
                    for turn, _, label in diarization.itertracks(yield_label=True):
                        if turn.start <= mid <= turn.end:
                            seg["speaker"] = label
                            break
            except Exception as e:
                _log.warning("Diarization 失败: %s", e)
                print(f"[Diarization ERROR] {type(e).__name__}: {e}", flush=True)

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

    def _transcribe_single_whisperx(self, video_path: Path, diarize: bool = False,
                                     language: Optional[str] = None, task: str = "transcribe") -> list[Segment]:
        import whisperx
        audio = whisperx.load_audio(str(video_path))
        result = self._model.transcribe(audio, batch_size=16, language=language, task=task)

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
                    token=self._hf_token, device="cpu"  # pyannote doesn't support MPS
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

    def _transcribe_chunked(self, video_path: Path, duration: float, diarize: bool = False,
                            language: Optional[str] = None, task: str = "transcribe") -> list[Segment]:
        import tempfile
        import subprocess
        import math
        import logging
        _log = logging.getLogger(__name__)
        all_segments: list[Segment] = []
        cursor = 0.0
        chunk_index = 0
        total_chunks = math.ceil(duration / CHUNK_DURATION)
        with tempfile.TemporaryDirectory() as tmpdir:
            while cursor < duration:
                chunk_end = min(cursor + CHUNK_DURATION, duration)
                # Extract audio-only WAV — avoids keyframe alignment issues with -c copy
                chunk_path = Path(tmpdir) / f"chunk_{chunk_index}.wav"
                _log.info(
                    "处理第 %d/%d 块 (%d:%02d – %d:%02d)",
                    chunk_index + 1, total_chunks,
                    int(cursor) // 60, int(cursor) % 60,
                    int(chunk_end) // 60, int(chunk_end) % 60,
                )
                chunk_dur = chunk_end - cursor
                proc = subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(cursor), "-t", str(chunk_dur),
                     "-i", str(video_path),
                     "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                     str(chunk_path)],
                    capture_output=True,
                )
                if proc.returncode != 0:
                    _log.warning(
                        "ffmpeg chunk %d 失败 (exit %d): %s",
                        chunk_index + 1, proc.returncode,
                        proc.stderr.decode(errors="replace")[-400:],
                    )
                    cursor = chunk_end
                    chunk_index += 1
                    continue
                for seg in self._transcribe_single(chunk_path, diarize=diarize, language=language, task=task):
                    all_segments.append(Segment(
                        start=seg.start + cursor,
                        end=seg.end + cursor,
                        text=seg.text,
                        speaker=seg.speaker,
                    ))
                _log.info("第 %d/%d 块完成", chunk_index + 1, total_chunks)
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
