from dataclasses import dataclass
from pathlib import Path
from models.edit_plan import EditPlan, Segment, CandidateSegment
from processing.transcriber import Transcriber
from agent.intent_parser import IntentParser
from agent.rule_engine import RuleEngine
from processing.ffmpeg_utils import get_video_duration


@dataclass
class OrchestrationResult:
    transcript: list[Segment]
    plan: EditPlan
    candidates: list[CandidateSegment]


class Orchestrator:
    def __init__(self):
        self._transcriber = Transcriber()
        self._parser = IntentParser()
        self._engine = RuleEngine()

    def run(self, video_path: Path, user_instruction: str) -> OrchestrationResult:
        """Full pipeline: transcribe → parse intent → execute rules → return candidates."""
        # Step 1: Transcribe
        transcript = self._transcriber.transcribe(video_path)

        # Step 2: Get video duration for padding bounds
        try:
            duration = get_video_duration(video_path)
        except Exception:
            duration = None

        # Step 3: Parse intent
        transcript_dicts = [s.model_dump() for s in transcript]
        plan = self._parser.parse(
            user_instruction=user_instruction,
            transcript=transcript_dicts,
        )

        # Step 4: Execute rules
        candidates = self._engine.execute(plan, transcript, video_path, duration)

        return OrchestrationResult(
            transcript=transcript,
            plan=plan,
            candidates=candidates,
        )
