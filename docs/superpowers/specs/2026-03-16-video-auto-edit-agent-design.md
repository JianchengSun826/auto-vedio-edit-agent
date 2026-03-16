# Video Auto-Edit Agent — Design Spec

**Date**: 2026-03-16
**Status**: Approved

---

## Overview

An AI-powered video auto-editing agent that accepts natural language editing instructions from users, processes local video files, generates candidate edit plans, and presents them for human review before exporting to multiple social media platforms.

---

## Goals

- Accept natural language editing requirements from users
- Support three editing modes: highlight extraction, material assembly, social media content production
- Process local video files (architecture extensible to cloud storage)
- Use STT + LLM to understand intent and generate structured edit plans
- Require human review and approval before any export
- Export to multiple platforms: 抖音/TikTok, B站, YouTube, 微信视频号
- Minimize cost: target < $5/month for moderate usage (100 videos/month)

---

## Non-Goals

- Real-time / live video processing
- Visual AI (face detection, scene understanding) — out of scope for v1
- Mobile app or native desktop UI
- Multi-user / SaaS platform

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Gradio UI (Web)                      │
│  [Upload] [Input Requirements] [Review Timeline]        │
│  [Approve / Adjust] [Export]                            │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                Agent Orchestrator (Python)               │
│   Receives instructions → Calls LLM → Dispatches tasks  │
└──────┬──────────────┬──────────────┬────────────────────┘
       │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌───▼────────────────────┐
│  STT Module │ │ LLM Parser │ │  Video Processing       │
│  faster-    │ │ Claude     │ │  FFmpeg (subprocess)    │
│  whisper    │ │ Haiku      │ │  cut / transcode /      │
│  (local)    │ │            │ │  concat / silence detect│
└──────┬──────┘ └─────┬──────┘ └───┬────────────────────┘
       │              │             │
┌──────▼──────────────▼─────────────▼───────────────────┐
│               Storage Abstraction Layer                  │
│   StorageBackend (abstract)                             │
│   └── LocalStorage (v1 implemented)                     │
│   └── S3Storage (stub, future)                          │
└────────────────────────────────────────────────────────┘
                     │
         Celery + Valkey (async tasks)
```

---

## Module Design

### Storage Abstraction Layer

All modules interact only with the `StorageBackend` abstract interface. This allows future cloud storage providers (S3, GCS) to be added by implementing the interface without modifying any other module.

```python
# storage/base.py
class StorageBackend(ABC):
    def read(self, path: str) -> Path: ...    # returns local temp path
    def write(self, local_path: Path, dest: str): ...
    def list(self, prefix: str) -> list[str]: ...
```

- **v1**: `LocalStorage` — reads/writes local filesystem
- **future**: `S3Storage` — downloads to temp dir, uploads on write
- Switch via `settings.py`: `STORAGE_BACKEND = "local"` → `"s3"`

### Input Manager

- Accepts local file paths
- Validates video format (mp4, mov, avi, mkv)
- Supports batch input (multiple files queued)
- Architecture: delegates to `StorageBackend.read()`

### STT Transcription Module (`processing/transcriber.py`)

- **Tool**: faster-whisper
- **Default model**: `medium` (speed/quality balance)
- **Configurable**: `tiny` / `base` / `small` / `medium` / `large-v3`
- Built-in VAD (Voice Activity Detection) to skip silent segments
- Output format: `List[Segment]` with `{start, end, text}`
- Long videos (>2h): split into 30-minute chunks, merge results
- Falls back gracefully if no speech detected (returns empty transcript)

### LLM Intent Parser (`agent/intent_parser.py`)

- **Tool**: Claude Haiku (`claude-haiku-4-5`)
- Input: user natural language instruction + transcript JSON
- Output: structured `EditPlan` (Pydantic model, JSON mode)
- Retry once on invalid JSON output; fall back to raw rule parsing
- Prompt design: system prompt defines JSON schema strictly

**EditPlan schema:**
```json
{
  "mode": "highlight_extraction | material_assembly | social_media",
  "rules": [
    {
      "type": "keyword_match | time_range | silence_cut | min_duration",
      "keywords": ["..."],
      "padding_before_sec": 3,
      "padding_after_sec": 5,
      "min_duration_sec": 10
    }
  ],
  "output_formats": [
    {
      "platform": "douyin | bilibili | youtube | wechat",
      "ratio": "9:16 | 16:9 | 1:1",
      "max_duration_sec": 60,
      "resolution": "1080p"
    }
  ],
  "segment_count_hint": 3
}
```

### Rule Execution Engine (`agent/rule_engine.py`)

- Consumes `EditPlan`, runs rules against transcript
- `keyword_match`: fuzzy match keywords in transcript text → extract time windows with padding
- `time_range`: direct time-code extraction
- `silence_cut`: call FFmpeg `silencedetect` filter, remove silent segments
- `min_duration`: filter out segments shorter than threshold
- Output: `List[CandidateSegment]` with `{id, start, end, text_preview, confidence_score}`

### FFmpeg Utilities (`processing/ffmpeg_utils.py`)

- All FFmpeg calls via `subprocess` (no ffmpeg-python wrapper)
- **Cut**: stream-copy for lossless fast cuts (`-c copy`)
- **Re-encode**: only when format conversion required
- **Concat**: FFmpeg concat demuxer (no re-encode for same-format files)
- **Silence detection**: `silencedetect` audio filter, parse stderr output
- **Platform export**: aspect ratio crop (center crop), resolution scale, codec presets per platform

### Human Review UI (`app/main.py` — Gradio 5.x)

- Candidate segment list: index, time range, transcript preview, confidence score
- Inline video player: click segment → seek and play that clip
- Controls per segment: checkbox (include/exclude), time code fine-tune
- Drag-to-reorder for material assembly mode
- "Approve & Export" button triggers export pipeline
- "Re-generate" button re-runs LLM with modified instructions

### Export Module (`processing/exporter.py`)

Platform specifications:

| Platform | Ratio | Max Duration | Resolution | Notes |
|----------|-------|-------------|------------|-------|
| 抖音/TikTok | 9:16 | 60s (auto-split if longer) | 1080×1920 | H.264 |
| B站 | 16:9 | None | 1080p / 4K | H.264 / HEVC |
| YouTube | 16:9 | None | 1080p / 4K | H.264 |
| 微信视频号 | 9:16 | 10min | 1080×1920 | H.264 |

- Batch export: all selected platforms in parallel (Celery group)
- Auto-split: if clip exceeds platform max duration, split into numbered parts
- Output naming: `{original_name}_{platform}_{part}.mp4`

### Async Task Queue (`tasks/celery_tasks.py`)

- **Broker + Backend**: Valkey (Redis-compatible, BSD license)
- Transcription task: long-running, `acks_late=True`, `max_tasks_per_child=10`
- Export task: per-platform, run as Celery group (parallel)
- Task state surfaced to Gradio UI via polling

---

## Three Editing Modes

### Mode 1: Highlight Extraction
- User: `"提取所有提到竞品价格的片段，前后各保留5秒"`
- Flow: Transcribe → LLM identifies keywords → Rule engine matches timestamps → Candidate list

### Mode 2: Material Assembly
- User: `"按脚本顺序拼接：开场30秒用第一个视频，产品演示用第二个视频2-4分钟"`
- Flow: LLM parses script structure → Matches segments across multiple input files → FFmpeg concat

### Mode 3: Social Media Production
- User: `"把这个采访剪成3条抖音短视频，每条不超过60秒，内容完整"`
- Flow: Transcribe → LLM semantic segmentation → Auto-crop 9:16 → 3 candidate packages for review

---

## Directory Structure

```
auto-video-edit-agent/
├── app/
│   └── main.py                  # Gradio UI entry point
├── agent/
│   ├── orchestrator.py          # Main coordinator
│   ├── intent_parser.py         # Claude Haiku → EditPlan
│   └── rule_engine.py           # EditPlan → CandidateSegments
├── processing/
│   ├── transcriber.py           # faster-whisper wrapper
│   ├── ffmpeg_utils.py          # FFmpeg subprocess helpers
│   └── exporter.py              # Multi-platform export
├── storage/
│   ├── base.py                  # StorageBackend ABC
│   ├── local.py                 # LocalStorage (v1)
│   └── s3.py                    # S3Storage (stub, not implemented)
├── models/
│   └── edit_plan.py             # Pydantic models: EditPlan, CandidateSegment
├── tasks/
│   └── celery_tasks.py          # Async task definitions
├── config/
│   └── settings.py              # API keys, model config, storage backend
├── tests/
│   ├── test_transcriber.py
│   ├── test_intent_parser.py
│   ├── test_rule_engine.py
│   └── test_exporter.py
├── docker-compose.yml           # Valkey + Celery worker
├── requirements.txt
└── README.md
```

---

## Error Handling

| Scenario | Handling |
|----------|---------|
| Video has no speech | Skip STT, offer time-range and silence-cut rules only |
| LLM returns invalid JSON | Retry once; if fails, return parse error with raw output |
| FFmpeg encode failure | Capture stderr, surface specific error line to user |
| Celery worker crash | `acks_late=True` — task re-queued automatically |
| Video > 2 hours | Split into 30-min chunks for transcription, merge results |
| Unsupported video format | Validate at input stage, reject with clear message |

---

## Cost Estimate

| Component | Cost |
|-----------|------|
| faster-whisper (local) | $0 |
| FFmpeg | $0 |
| Claude Haiku per video (10min) | ~$0.016 |
| 100 videos/month LLM cost | ~$1.60 |
| Local storage | $0 |
| **Total (moderate use)** | **< $5/month** |

---

## Tech Stack Summary

| Role | Tool | Version |
|------|------|---------|
| Video processing | FFmpeg (subprocess) | 7.x |
| Speech-to-text | faster-whisper | ~1.0.x |
| LLM parsing | Claude Haiku (`claude-haiku-4-5`) | — |
| Review UI | Gradio | 5.x |
| Async queue | Celery + Valkey | 5.x / 7.x |
| Data models | Pydantic | v2 |
| Cloud storage (future) | boto3 + S3 | — |

---

## Future Extensions

- **S3 / Cloud Storage**: implement `S3Storage(StorageBackend)`, change `STORAGE_BACKEND` setting
- **Visual AI**: add scene detection module between transcription and rule engine
- **Speaker diarization**: integrate pyannote.audio for multi-speaker scenarios
- **Web API**: wrap orchestrator with FastAPI for programmatic access
- **Batch job scheduling**: Celery beat for scheduled processing pipelines
