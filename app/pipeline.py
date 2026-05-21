from __future__ import annotations
from models.edit_plan import (
    EditPlan, EditMode, Rule, RuleType, CandidateSegment, Segment,
)


def build_plan_from_buttons(
    selected: list[str],
    keywords: list[str],
    keyword_before: float,
    keyword_after: float,
    time_start: float | None,
    time_end: float | None,
    speaker_ids: list[str],
) -> EditPlan:
    """Construct EditPlan directly from button selections, no LLM needed."""
    rules: list[Rule] = []

    if "keyword" in selected and keywords:
        rules.append(Rule(
            type=RuleType.KEYWORD_MATCH,
            keywords=keywords,
            padding_before_sec=keyword_before,
            padding_after_sec=keyword_after,
        ))

    if "speaker" in selected and speaker_ids:
        rules.append(Rule(
            type=RuleType.SPEAKER_FILTER,
            speakers=speaker_ids,
            padding_before_sec=0.0,
            padding_after_sec=0.0,
        ))

    if "time" in selected and time_start is not None and time_end is not None:
        rules.append(Rule(
            type=RuleType.TIME_RANGE,
            start_sec=time_start,
            end_sec=time_end,
        ))

    if "silence" in selected:
        rules.append(Rule(type=RuleType.SILENCE_CUT))

    return EditPlan(
        mode=EditMode.HIGHLIGHT_EXTRACTION,  # button path always extracts highlights
        rules=rules,
        output_formats=[],
        segment_count_hint=3,
    )


def candidates_to_rows(candidates: list[CandidateSegment]) -> list[list]:
    """Convert CandidateSegment list to Gradio Dataframe rows."""
    def _preview(text: str | None) -> str:
        if not text:
            return "—"
        return (text[:77] + "…") if len(text) > 80 else text

    return [
        [
            i + 1,
            seg.speaker or "—",
            f"{seg.start:.1f}s – {seg.end:.1f}s",
            _preview(seg.text_preview_zh),
            _preview(seg.text_preview_en),
            f"{seg.confidence_score:.2f}",
            True,
        ]
        for i, seg in enumerate(candidates)
    ]


def extract_speakers(transcript: list[Segment]) -> list[str]:
    """Return sorted unique speaker IDs from transcript, excluding None."""
    seen: set[str] = set()
    result: list[str] = []
    for seg in transcript:
        if seg.speaker and seg.speaker not in seen:
            seen.add(seg.speaker)
            result.append(seg.speaker)
    return sorted(result)


def step_html(done: set[int], active: int, skip_llm: bool = False) -> str:
    """Render 3-step progress indicator as HTML. Steps are 1-indexed."""
    steps = [(1, "① 音频转录"), (2, "② 解析意图"), (3, "③ 执行规则")]
    parts: list[str] = []
    has_active = any(num == active for num, _ in steps)
    style = (
        "<style>"
        "@keyframes _sp_pulse{0%,100%{opacity:1}50%{opacity:.5}}"
        "@keyframes _sp_spin{to{transform:rotate(360deg)}}"
        ".sp-active{animation:_sp_pulse 1.4s ease-in-out infinite}"
        ".sp-spinner{display:inline-block;animation:_sp_spin 1s linear infinite}"
        "</style>"
    ) if has_active else ""

    for idx, (num, label) in enumerate(steps):
        if num in done:
            bg, color, content = "#2a6a2a", "#6fa", f"{label} ✓"
            cls = ""
        elif num == active:
            bg, color = "#3a5a9a", "#fff"
            content = f'{label} <span class="sp-spinner">◌</span>'
            cls = ' class="sp-active"'
        elif skip_llm and num == 2:
            bg, color, content, cls = "#252545", "#444", f"{label} —", ""
        else:
            bg, color, content, cls = "#252545", "#555", label, ""
        r = "4px 0 0 4px" if idx == 0 else ("0 4px 4px 0" if idx == 2 else "0")
        parts.append(
            f'<div{cls} style="flex:1;text-align:center;background:{bg};border-radius:{r};'
            f'padding:6px 4px;font-size:12px;color:{color}">{content}</div>'
        )
        if idx < 2:
            parts.append(
                f'<div style="width:0;height:0;border-top:12px solid transparent;'
                f'border-bottom:12px solid transparent;border-left:8px solid {bg}"></div>'
            )
    inner = "".join(parts)
    return f'{style}<div style="display:flex;align-items:center;margin:8px 0">{inner}</div>'
