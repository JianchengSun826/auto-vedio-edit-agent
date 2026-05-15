from __future__ import annotations
import json
from pathlib import Path
import anthropic
from models.edit_plan import EditPlan
from config.settings import settings

SYSTEM_PROMPT = """You are a video editing assistant. Given a user's editing instruction and a video transcript,
output a JSON EditPlan that describes how to edit the video.

The transcript may include speaker labels (e.g. "SPEAKER_00:"). Use speaker_filter rules when the user
wants to extract segments from a specific speaker. Only use speaker_filter when the transcript contains
SPEAKER_xx labels; otherwise use keyword_match.

Output ONLY valid JSON matching this exact schema:
{
  "mode": "highlight_extraction" | "material_assembly" | "social_media",
  "rules": [
    {
      "type": "keyword_match" | "time_range" | "silence_cut" | "min_duration" | "speaker_filter",
      "keywords": [...],            // for keyword_match only
      "speakers": [...],            // for speaker_filter only, e.g. ["SPEAKER_00"]
      "padding_before_sec": 3,
      "padding_after_sec": 5,
      "min_duration_sec": 5,
      "start_sec": null,            // for time_range only
      "end_sec": null               // for time_range only
    }
  ],
  "output_formats": [
    {
      "platform": "douyin" | "bilibili" | "youtube" | "wechat",
      "ratio": "9:16" | "16:9" | "1:1",
      "max_duration_sec": null,
      "resolution": "1080p"
    }
  ],
  "segment_count_hint": 3
}

Output JSON only. No explanation."""


class IntentParser:
    def __init__(self):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._preferences = self._load_preferences()

    def _load_preferences(self) -> str:
        path = Path("USER_PREFERENCES.md")
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def _build_system_prompt(self) -> str:
        if self._preferences:
            return f"[用户剪辑偏好]\n{self._preferences}\n\n---\n\n{SYSTEM_PROMPT}"
        return SYSTEM_PROMPT

    def parse(self, user_instruction: str, transcript: list[dict]) -> EditPlan:
        transcript_text = "\n".join(
            f"[{s['start']:.1f}s - {s['end']:.1f}s]"
            f"{' ' + s['speaker'] + ':' if s.get('speaker') else ''} {s['text']}"
            for s in transcript
        )
        user_message = (
            f"User instruction: {user_instruction}\n\n"
            f"Transcript:\n{transcript_text}"
        )

        for attempt in range(2):
            response = self._client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                system=self._build_system_prompt(),
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()
            try:
                data = json.loads(raw)
                return EditPlan.model_validate(data)
            except (json.JSONDecodeError, Exception):
                if attempt == 1:
                    raise ValueError(
                        f"Failed to parse LLM response after 2 attempts. "
                        f"Last response: {raw[:200]}"
                    )
                continue

        raise ValueError("Failed to parse LLM response")
