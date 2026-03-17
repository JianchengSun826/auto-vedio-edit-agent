from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class EditMode(str, Enum):
    HIGHLIGHT_EXTRACTION = "highlight_extraction"
    MATERIAL_ASSEMBLY = "material_assembly"
    SOCIAL_MEDIA = "social_media"


class RuleType(str, Enum):
    KEYWORD_MATCH = "keyword_match"
    TIME_RANGE = "time_range"
    SILENCE_CUT = "silence_cut"
    MIN_DURATION = "min_duration"


class Platform(str, Enum):
    DOUYIN = "douyin"
    BILIBILI = "bilibili"
    YOUTUBE = "youtube"
    WECHAT = "wechat"


# Platform defaults
_PLATFORM_DEFAULTS = {
    Platform.DOUYIN:   {"ratio": "9:16",  "max_duration_sec": 60,   "resolution": "1080p"},
    Platform.WECHAT:   {"ratio": "9:16",  "max_duration_sec": 600,  "resolution": "1080p"},
    Platform.BILIBILI: {"ratio": "16:9",  "max_duration_sec": None, "resolution": "1080p"},
    Platform.YOUTUBE:  {"ratio": "16:9",  "max_duration_sec": None, "resolution": "1080p"},
}


class Rule(BaseModel):
    type: RuleType
    keywords: list[str] = Field(default_factory=list)
    padding_before_sec: float = 3.0
    padding_after_sec: float = 5.0
    min_duration_sec: float = 5.0
    start_sec: Optional[float] = None   # for time_range
    end_sec: Optional[float] = None     # for time_range


class OutputFormat(BaseModel):
    platform: Platform
    ratio: str = "16:9"
    max_duration_sec: Optional[int] = None
    resolution: str = "1080p"

    def __init__(self, **data):
        platform = data.get("platform")
        if platform is not None:
            try:
                p = Platform(platform)
                defaults = _PLATFORM_DEFAULTS.get(p, {})
                for k, v in defaults.items():
                    data.setdefault(k, v)  # only set if not explicitly provided
            except ValueError:
                pass
        super().__init__(**data)


class EditPlan(BaseModel):
    mode: EditMode
    rules: list[Rule]
    output_formats: list[OutputFormat]
    segment_count_hint: int = 3


class Segment(BaseModel):
    """Raw transcript segment from faster-whisper."""
    start: float
    end: float
    text: str


class CandidateSegment(BaseModel):
    """A segment proposed for inclusion in the final edit."""
    id: str
    start: float
    end: float
    text_preview: str
    confidence_score: float = 1.0
    included: bool = True
    source_file: Optional[str] = None   # for material assembly mode
