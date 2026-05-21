# 双语字幕并行转录 — 设计文档

**日期：** 2026-05-21  
**状态：** 待实施

---

## 背景

当前 Whisper 转录在混合语言视频中存在随机性：语言检测仅基于前 30 秒，一旦锁定，同视频中另一语言的片段会被强行翻译或乱码输出。large-v3 模型足够强大，会将另一语言"顺手翻译"而非输出乱码，但这是不可预测的涌现行为。

**目标：** 对同一音频运行两次 Whisper，分别强制输出中文和英文字幕，同时展示、同时导出，消除随机性。

---

## 方案

运行两次 Whisper（`large-v3`），并行执行：

| Pass | 参数 | 输出 |
|------|------|------|
| 中文 | `language="zh"`, `task="transcribe"` | 中文说话人 → 原始中文；英文说话人 → large-v3 的中文渲染 |
| 英文 | `language=None`, `task="translate"` | 所有语言 → 英文（Whisper 内置翻译，可靠） |

不使用外部翻译模型（无 Helsinki-NLP）。

---

## 改动范围

### 1. `models/edit_plan.py` — Segment 扩展

新增两个可选字段：

```python
class Segment(BaseModel):
    start: float
    end: float
    text: str           # 向后兼容，值等于 text_zh
    text_zh: Optional[str] = None
    text_en: Optional[str] = None
    speaker: Optional[str] = None
```

### 2. `processing/transcriber.py` — 参数透传 + 并行方法

- `language` 和 `task` 参数沿完整调用链透传：
  `transcribe()` → `_transcribe_single()` / `_transcribe_chunked()` → `_transcribe_single_mlx()` / `_transcribe_single_whisperx()` → 底层模型
- 新增 `transcribe_bilingual(video_path, diarize=False) -> list[Segment]`：
  - 用 `ThreadPoolExecutor(max_workers=2)` 并行运行两次 `transcribe()`
  - zh pass: `language="zh"`, `task="transcribe"`
  - en pass: `language=None`, `task="translate"`
  - 按索引合并结果，填入 `text_zh` / `text_en`；索引不一致时按时间戳最近匹配兜底
  - `text` 字段设为 `text_zh` 的值

### 3. `processing/subtitle.py` — 双语导出

`segments_to_srt` 增加 `lang: Literal["zh", "en"] = "zh"` 参数，根据 lang 选择 `text_zh` 或 `text_en` 字段写入 SRT。

### 4. `app/pipeline.py` — 表格双列

`candidates_to_rows` 将"内容预览"列拆为"中文字幕"和"英文字幕"两列，预览截断规则相同（77 字符 + `…`）。

### 5. `app/main.py` — 主流程

- `run_pipeline` 调用 `transcribe_bilingual` 替换原 `transcribe`
- 表格 headers 改为 `["序号", "说话人", "时间范围", "中文字幕", "英文字幕", "置信度", "包含"]`
- 字幕导出：生成 `{stem}_zh.srt` 和 `{stem}_en.srt` 两个文件
- `srt_download` 组件改为 `file_count="multiple"`
- 进度状态显示：`正在转录音频（双语并行）···  ⏱ {elapsed}`

---

## 数据流

```
video_path
    │
    ├── Thread-1: _transcribe(language="zh", task="transcribe")
    │       → [Segment(text="中文..."), ...]
    │
    └── Thread-2: _transcribe(language=None, task="translate")
            → [Segment(text="English..."), ...]
                    │
              merge by index
                    │
         [Segment(text_zh="中文", text_en="English"), ...]
                    │
          ┌─────────┴──────────┐
    candidates_to_rows()   segments_to_srt(lang="zh")
    (两列预览)              segments_to_srt(lang="en")
```

---

## 错误处理

- 任一 pass 失败：记录 warning，对应列显示 `—`，不中断整体流程
- 两次 pass 片段数不一致：按时间戳最近匹配，无法匹配的片段对应列填 `—`
- `text` 字段兜底：若 `text_zh` 为空，`text` 保留原始 Whisper 输出

---

## 不在本次范围内

- 用户选择"只用中文"或"只用英文"运行单次转录
- 其他功能（关键词、说话人筛选）的语言偏好设置
- Helsinki-NLP 外部翻译模型集成
