import re
import subprocess
from pathlib import Path


def _run(cmd: list[str], capture_stderr: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{result.stderr}")
    return result


def cut_segment(src: Path, dest: Path, start: float, end: float) -> None:
    """Cut a segment from src using stream copy (lossless).
    -ss before -i for fast seek; -to after -i is relative to the seek point.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    _run([
        "ffmpeg", "-y",
        "-ss", str(start),   # fast input seek
        "-i", str(src),
        "-t", str(duration), # duration (not absolute end) after -i
        "-c", "copy",
        str(dest),
    ])


def concat_segments(segment_files: list[Path], dest: Path) -> None:
    """Concatenate pre-cut segment files using FFmpeg concat demuxer."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    list_file = dest.parent / "concat_list.txt"
    list_file.write_text(
        "\n".join(f"file '{f.resolve()}'" for f in segment_files)
    )
    _run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(dest),
    ])
    list_file.unlink()


def detect_silence(src: Path, noise_db: float = -40, min_duration: float = 1.0) -> list[tuple[float, float]]:
    """Return list of (start, end) silence intervals in seconds."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(src),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg silencedetect error:\n{result.stderr}")
    starts = re.findall(r"silence_start: ([\d.]+)", result.stderr)
    ends = re.findall(r"silence_end: ([\d.]+)", result.stderr)
    return [(float(s), float(e)) for s, e in zip(starts, ends)]


def transcode_for_platform(src: Path, dest: Path, width: int, height: int) -> None:
    """Re-encode video with center crop to target aspect ratio and resolution."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale='if(gt(a,{width}/{height}),{height}*a,-2)':'if(gt(a,{width}/{height}),-2,{width}/a)',"
        f"crop={width}:{height},"
        f"scale={width}:{height}"
    )
    _run([
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        str(dest),
    ])


def get_video_duration(src: Path) -> float:
    """Return video duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1",
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    match = re.search(r"duration=([\d.]+)", result.stdout)
    if not match:
        raise ValueError(f"Could not parse duration from: {result.stdout}")
    return float(match.group(1))
