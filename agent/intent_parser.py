import json
import anthropic
from models.edit_plan import EditPlan
from config.settings import settings

SYSTEM_PROMPT = """You are a video editing assistant. Given a user's editing instruction and a video transcript,
output a JSON EditPlan that describes how to edit the video.

Output ONLY valid JSON matching this exact schema:
{
  "mode": "highlight_extraction" | "material_assembly" | "social_media",
  "rules": [
    {
      "type": "keyword_match" | "time_range" | "silence_cut" | "min_duration",
      "keywords": [...],          // for keyword_match only
      "padding_before_sec": 3,
      "padding_after_sec": 5,
      "min_duration_sec": 5,
      "start_sec": null,          // for time_range only
      "end_sec": null             // for time_range only
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

    def parse(self, user_instruction: str, transcript: list[dict]) -> EditPlan:
        transcript_text = "\n".join(
            f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}"
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
                system=SYSTEM_PROMPT,
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

        raise ValueError("Failed to parse LLM response")  # unreachable but satisfies type checker
