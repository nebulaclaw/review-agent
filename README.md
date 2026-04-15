# Content Review Agent

**LangChain**-based content moderation orchestration: a **deterministic pipeline** (wordlists, fingerprints, OCR, etc.) runs alongside an **Agent + tool packs**; multi-vendor LLM adapters are unified, with optional queues, SQLite audit trails, and metrics. Suited for content safety and UGC governance.

**Docs**: [Layering (L0)](docs/architecture.md) · [Agent tool packs](docs/agent-tool-packs.md)

## Requirements

- Python **≥ 3.9** (3.10+ recommended)
- Local models: [Ollama](https://ollama.com/) or a compatible inference server

## Quick start

```bash
python3 -m pip install -e .
cp .env.example .env
# Edit .env and config.yaml as needed (keep secrets in the environment or local .env, not in committed config.yaml)

content-review check "text to review"
content-review check /path/to/image.jpg
content-review check /path/to/video.mp4
content-review file ./a.txt ./b.png
content-review interactive
content-review server --host 0.0.0.0 --port 18080
```

`check` infers text / image / video automatically; you do not need `--type`. The server root serves a simple review page.

If the console script is not on your PATH: `python cli.py …` (same subcommands).

### Optional installs

- `pip install -e ".[image]"` / `".[video]"` — image / video detection extras (see `pyproject.toml`); the terminal UI (`content-review tui`) is included in the default dependencies.

### Video and FFmpeg

Local video detection needs **`ffprobe` and `ffmpeg` on PATH** (e.g. on macOS: `brew install ffmpeg`). If they are missing, the pipeline degrades and records labels such as `ffprobe_not_found`.

Optional Whisper: `REVIEW_AGENT_ASR_MODEL` (default `tiny`), `REVIEW_AGENT_ASR_LANGUAGE` (use `zh` for Chinese).

### Wordlists and knowledge base

- Wordlists: `config/wordlists/`
- Vector store: put documents under `config/knowledge/`, then run `content-review knowledge ingest-config` (or `ingest <dir>`)

### Offline bundle

```bash
./scripts/package_bundle.sh   # or: python3 scripts/package_bundle.py
```

Artifacts land in `dist/`; follow `INSTALL.txt` inside the bundle to install.
