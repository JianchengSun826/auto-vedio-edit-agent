from pathlib import Path
from tasks.celery_app import app
from processing.transcriber import Transcriber
from processing.exporter import Exporter
from models.edit_plan import CandidateSegment, OutputFormat


@app.task(bind=True, name="tasks.transcribe")
def transcribe_video(self, video_path: str) -> list[dict]:
    """Transcribe a video file. Returns list of segment dicts."""
    self.update_state(state="PROGRESS", meta={"status": "Transcribing..."})
    transcriber = Transcriber()
    segments = transcriber.transcribe(Path(video_path))
    return [s.model_dump() for s in segments]


@app.task(bind=True, name="tasks.export")
def export_video(
    self,
    video_path: str,
    candidates: list[dict],
    formats: list[dict],
    output_dir: str,
) -> list[str]:
    """Export approved candidates for all platforms. Returns list of output paths."""
    self.update_state(state="PROGRESS", meta={"status": "Exporting..."})
    exporter = Exporter(output_dir=output_dir)
    candidate_objs = [CandidateSegment(**c) for c in candidates]
    format_objs = [OutputFormat(**f) for f in formats]
    results = exporter.export(Path(video_path), candidate_objs, format_objs)
    return [str(p) for p in results]
