from pathlib import Path
from models.edit_plan import EditPlan, RuleType, Rule, Segment, CandidateSegment
import uuid


class RuleEngine:
    def execute(
        self,
        plan: EditPlan,
        transcript: list[Segment],
        video_path: Path | None,
        video_duration: float | None = None,
    ) -> list[CandidateSegment]:
        """Execute EditPlan rules against transcript. Returns candidate segments."""
        candidates: list[CandidateSegment] = []

        for rule in plan.rules:
            if rule.type == RuleType.KEYWORD_MATCH:
                candidates.extend(self._keyword_match(rule, transcript, video_duration))
            elif rule.type == RuleType.TIME_RANGE:
                candidates.extend(self._time_range(rule))
            elif rule.type == RuleType.MIN_DURATION:
                candidates = self._filter_min_duration(rule, candidates)
            elif rule.type == RuleType.SILENCE_CUT:
                if video_path is not None:
                    candidates = self._apply_silence_cut(rule, candidates, video_path)

        # Deduplicate overlapping segments
        candidates = self._merge_overlapping(candidates)
        return candidates

    def _keyword_match(
        self, rule: Rule, transcript: list[Segment], video_duration: float | None
    ) -> list[CandidateSegment]:
        results = []
        for seg in transcript:
            if any(kw in seg.text for kw in rule.keywords):
                start = max(0.0, seg.start - rule.padding_before_sec)
                end = seg.end + rule.padding_after_sec
                if video_duration:
                    end = min(end, video_duration)
                results.append(CandidateSegment(
                    id=str(uuid.uuid4()),
                    start=start,
                    end=end,
                    text_preview=seg.text,
                    confidence_score=1.0,
                ))
        return results

    def _time_range(self, rule: Rule) -> list[CandidateSegment]:
        if rule.start_sec is None or rule.end_sec is None:
            return []
        return [CandidateSegment(
            id=str(uuid.uuid4()),
            start=rule.start_sec,
            end=rule.end_sec,
            text_preview=f"[{rule.start_sec:.1f}s - {rule.end_sec:.1f}s]",
            confidence_score=1.0,
        )]

    def _apply_silence_cut(
        self, rule: Rule, candidates: list[CandidateSegment], video_path: Path
    ) -> list[CandidateSegment]:
        """Remove silence intervals from candidate segments."""
        from processing.ffmpeg_utils import detect_silence
        try:
            silences = detect_silence(video_path)
        except RuntimeError:
            return candidates  # if detection fails, return unchanged

        result = []
        for seg in candidates:
            cursor = seg.start
            for s_start, s_end in silences:
                if s_end <= seg.start or s_start >= seg.end:
                    continue  # silence outside segment
                if cursor < s_start:
                    result.append(CandidateSegment(
                        id=str(uuid.uuid4()),
                        start=cursor,
                        end=s_start,
                        text_preview=seg.text_preview,
                        confidence_score=seg.confidence_score,
                    ))
                cursor = max(cursor, s_end)
            if cursor < seg.end:
                result.append(CandidateSegment(
                    id=str(uuid.uuid4()),
                    start=cursor,
                    end=seg.end,
                    text_preview=seg.text_preview,
                    confidence_score=seg.confidence_score,
                ))
        return result if result else candidates

    def _filter_min_duration(self, rule: Rule, candidates: list[CandidateSegment]) -> list[CandidateSegment]:
        return [c for c in candidates if (c.end - c.start) >= rule.min_duration_sec]

    def _merge_overlapping(self, candidates: list[CandidateSegment]) -> list[CandidateSegment]:
        if not candidates:
            return []
        sorted_segs = sorted(candidates, key=lambda c: c.start)
        merged = [sorted_segs[0]]
        for current in sorted_segs[1:]:
            last = merged[-1]
            if current.start <= last.end:
                merged[-1] = CandidateSegment(
                    id=last.id,
                    start=last.start,
                    end=max(last.end, current.end),
                    text_preview=f"{last.text_preview} | {current.text_preview}",
                    confidence_score=max(last.confidence_score, current.confidence_score),
                )
            else:
                merged.append(current)
        return merged
