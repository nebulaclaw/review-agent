# Agent tool packs

LangChain **tool** groups wired into the review agent. Built-ins live in `reviewagent/toolpacks/builtins.py`; all registered packs are attached (today: `review.rules` only).

## `review.rules`

- **`text_detector`** — Scan plain text (wordlists / text detector pipeline).
- **`image_detector`** — Review a local image path (OCR + rules / models as configured).
- **`video_detector`** — Review a local video path (sampled frames + subtitles / ASR as configured).
- **`audio_detector`** — Review a local audio path (transcribe, then same style of text checks).
