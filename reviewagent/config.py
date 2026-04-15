from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

# Provider → env vars tried when llm.api_key is empty (after ${VAR} expand). Same order as llm_factory vendors.
_PROVIDER_API_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "glm": ("ZHIPUAI_API_KEY",),
    "zhipu": ("ZHIPUAI_API_KEY",),
    "zhipuai": ("ZHIPUAI_API_KEY",),
    "kimi": ("MOONSHOT_API_KEY",),
    "moonshot": ("MOONSHOT_API_KEY",),
    "qwen": ("DASHSCOPE_API_KEY",),
    "dashscope": ("DASHSCOPE_API_KEY",),
    "tongyi": ("DASHSCOPE_API_KEY",),
    "minimax": ("MINIMAX_API_KEY",),
    "mini_max": ("MINIMAX_API_KEY",),
    "ollama": (),
    "local": (),
    "llamacpp": (),
}

_ZHIPU_PROV_ALIASES = frozenset(
    {"智谱", "质谱", "智谱glm", "智谱GLM", "智谱ai", "智谱AI", "智谱大模型"}
)


def _provider_token_for_api_key_env(prov: Any) -> str:
    """Match llm_factory-style provider names so switching provider picks the right env key."""
    if not isinstance(prov, str):
        return "openai"
    t = prov.strip()
    if t in _ZHIPU_PROV_ALIASES:
        return "glm"
    return t.lower() if t.isascii() else t


def _fill_llm_api_key_from_env(llm: dict) -> None:
    """
    After YAML and ${VAR} expansion: if api_key is still empty, fill from environment.

    1) Vendor vars for **current llm.provider** (e.g. glm → ZHIPUAI_API_KEY) so multiple keys can coexist.
    2) **LLM_API_KEY** if still empty (single key for all providers, or unknown provider).
    """
    ak = llm.get("api_key")
    if isinstance(ak, str) and ak.strip():
        return
    prov = _provider_token_for_api_key_env(llm.get("provider"))
    for env_name in _PROVIDER_API_KEY_ENVS.get(prov, ()):
        v = os.environ.get(env_name, "").strip()
        if v:
            llm["api_key"] = v
            return
    u = os.environ.get("LLM_API_KEY", "").strip()
    if u:
        llm["api_key"] = u


def _env_var(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.3
    max_tokens: int = 8192
    timeout: int = 60
    # Some MiniMax deployments require Group-Id
    minimax_group_id: str = ""


class MemoryConfig(BaseModel):
    short_term_max_messages: int = 10
    long_term_enabled: bool = True
    vector_store_type: str = "chroma"
    persist_directory: str = "data/memory"


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "review"
    max_iterations: int = 10
    max_loops: int = 5
    verbose: bool = True
    # Whether to send local image as base64 to the vision model (mitigate OCR misses; needs capable model/channel)
    attach_local_image_to_vision_llm: bool = True


class PipelineWordlistConfig(BaseModel):
    """Local wordlists, text preprocessing, OCR-then-wordlist on images, and dual LLM image review flags."""

    preprocess_nfkc: bool = True
    preprocess_lowercase: bool = True
    strip_zero_width: bool = True
    wordlist_paths: list[str] = Field(
        default_factory=lambda: ["config/wordlists/default.txt"]
    )
    # Skip LLM on wordlist hit (hard block)
    early_exit_on_match: bool = True
    # When no hit, inject a short "wordlist scan missed" hint before user message (still call LLM)
    inject_recall_hint: bool = True
    # Add toneless pinyin variants when phrases contain CJK (e.g. phrase → liantong, lian tong)
    expand_cjk_pinyin: bool = True
    # OCR image first, then run same wordlist on OCR text (catch text hidden in images)
    scan_image_ocr_for_wordlist: bool = True
    # After no hard block: call LLM on **OCR text** only (no tools; separate from vision branch)
    image_llm_review_ocr_text: bool = True
    # After no hard block: call **vision LLM** on raw pixels (no tools)
    image_llm_review_pixels: bool = True


class PipelineFingerprintConfig(BaseModel):
    """Image fingerprint blocklist, light metadata signals, and how dual-branch image results merge."""

    # Compute perceptual hash for local images and match blocklist (needs pillow + ImageHash; see pip install -e '.[image]')
    image_phash_enabled: bool = False
    # SQLite path for phash blocklist; empty → use storage.review_db_path (same DB as review runs, table image_phash_blocklist)
    image_phash_db_path: str = ""
    # Hamming threshold: 0 = exact only; 1–5 tolerates light recompress/resize (higher false-positive risk)
    image_phash_max_hamming: int = Field(default=0, ge=0, le=32)
    # If true: log this image's pHash even on miss (helps curating blocklist files)
    image_phash_log_on_miss: bool = False
    # Write light signals (dims, size, aspect, etc.) to pipeline_trace; no model inference
    image_collect_light_signals: bool = True
    # Dual LLM (OCR text / pixels) merge: max_severity = heaviest verdict; vision_primary / ocr_primary = that branch leads
    image_dual_merge_policy: Literal["max_severity", "vision_primary", "ocr_primary"] = (
        "max_severity"
    )


class PipelineImageDualConfig(BaseModel):
    """Dual image LLM (OCR vs vision) consistency check and how to escalate on disagreement."""

    # Compare OCR-text LLM vs vision LLM verdict for consistency
    image_dual_consistency_enabled: bool = True
    # On disagreement: none = trace only; elevate_warn = bump final verdict to at least WARN (when it was PASS)
    image_dual_disagreement_action: Literal["none", "elevate_warn"] = "elevate_warn"
    # Locale for dual-merge summary / consistency notes: zh = Chinese labels; en = English (avoid mixed labels)
    report_locale: Literal["zh", "en"] = "zh"


class PipelineConfig(BaseModel):
    """
    Review pipeline: text/image through local wordlist + preprocess (may early-exit), then LLM as needed;
    video etc. goes through Agent. Keys: wordlist, fingerprint, image_dual_check.
    """

    wordlist: PipelineWordlistConfig = Field(default_factory=PipelineWordlistConfig)
    fingerprint: PipelineFingerprintConfig = Field(default_factory=PipelineFingerprintConfig)
    image_dual_check: PipelineImageDualConfig = Field(default_factory=PipelineImageDualConfig)

    @model_validator(mode="before")
    @classmethod
    def _normalize_pipeline_dict(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        d.pop("mode", None)
        return d


class QueueConfig(BaseModel):
    enabled: bool = True
    max_concurrent: int = 4
    persist_path: str = "data/queue.db"
    poll_interval_ms: int = 200


class StorageConfig(BaseModel):
    review_db_path: str = "data/review.db"


class ObservabilityConfig(BaseModel):
    log_json: bool = False
    metrics_enabled: bool = True
    # Relative to repo root or absolute; empty = console only (no log file)
    log_file_path: str = ""


class LimitsConfig(BaseModel):
    """
    Limits are in KiB (1 KiB = 1024 bytes); 0 means unlimited.
    Code uses max_*_bytes read-only properties for byte values.
    """

    max_text_kb: int = Field(default=256, ge=0, description="Max UTF-8 text to review (KiB)")
    max_file_kb: int = Field(default=51200, ge=0, description="Max single file size (KiB); default ~50 MiB")
    max_user_message_kb: int = Field(
        default=512,
        ge=0,
        description="Max full user message including prompt template (KiB); should be ≥ max_text_kb",
    )

    @property
    def max_text_bytes(self) -> int:
        return 0 if self.max_text_kb <= 0 else self.max_text_kb * 1024

    @property
    def max_file_bytes(self) -> int:
        return 0 if self.max_file_kb <= 0 else self.max_file_kb * 1024

    @property
    def max_user_message_bytes(self) -> int:
        return 0 if self.max_user_message_kb <= 0 else self.max_user_message_kb * 1024


class RagConfig(BaseModel):
    """RAG knowledge store (Chroma); needs embedding model and network (offline: ollama embedding only)."""

    enabled: bool = False
    persist_directory: str = "data/knowledge_chroma"
    collection_name: str = "review_knowledge"
    retrieve_k: int = Field(default=5, ge=1, le=50)
    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)
    # Empty → follow llm.provider; for anthropic chat, set explicitly to openai / ollama / glm / etc.
    embedding_provider: str = ""
    embedding_model: str = ""
    # Relative to repo root; server may scan on startup (see auto_ingest_on_startup)
    knowledge_dirs: list[str] = Field(default_factory=lambda: ["config/knowledge"])
    auto_ingest_on_startup: bool = False
    max_ingest_file_kb: int = Field(default=2048, ge=0, description="Max file size to index (KiB); 0 = unlimited")


class Settings(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    # Offline: only local models (e.g. Ollama), skip optional components that need outbound network
    offline_mode: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        data = dict(raw)

        # Any string field under llm may use ${ENV_NAME}, expanded from environment at startup
        llm = data.get("llm")
        if isinstance(llm, dict):
            for lk, val in list(llm.items()):
                if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                    env_var = val[2:-1]
                    llm[lk] = os.environ.get(env_var, "")
            _fill_llm_api_key_from_env(llm)

        lim = data.get("limits")
        if isinstance(lim, dict):
            lim = dict(lim)

            def _bytes_to_kb(key_kb: str, key_bytes: str) -> None:
                if key_kb not in lim and key_bytes in lim:
                    b = int(lim.pop(key_bytes) or 0)
                    lim[key_kb] = 0 if b <= 0 else max(1, (b + 1023) // 1024)
                else:
                    lim.pop(key_bytes, None)

            _bytes_to_kb("max_text_kb", "max_text_bytes")
            _bytes_to_kb("max_file_kb", "max_file_bytes")
            _bytes_to_kb("max_user_message_kb", "max_user_message_bytes")
            data["limits"] = lim

        obs = data.get("observability")
        if isinstance(obs, dict):
            obs = dict(obs)
            for ok, val in list(obs.items()):
                if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                    obs[ok] = os.environ.get(val[2:-1], "")
            data["observability"] = obs

        sto = data.get("storage")
        if isinstance(sto, dict):
            data["storage"] = dict(sto)

        return cls(**data)


_settings: Optional[Settings] = None


def _default_config_path() -> str:
    """Prefer repo-root config.yaml next to the package; else cwd config.yaml."""
    root = Path(__file__).resolve().parent.parent
    bundled = root / "config.yaml"
    if bundled.is_file():
        return str(bundled)
    return "config.yaml"


def get_config_yaml_path() -> str:
    """Resolved config.yaml path for this process (same as first get_settings load)."""
    return _default_config_path()


def apply_pipeline_report_locale_to_yaml_file(path: str, report_locale: str) -> None:
    """
    Update pipeline.image_dual_check.report_locale in YAML and write back.
    Call reload_settings() afterward.
    """
    if report_locale not in ("zh", "en"):
        raise ValueError("report_locale must be zh or en")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    pipe = dict(data.get("pipeline") or {})
    idc = dict(pipe.get("image_dual_check") or {})
    idc["report_locale"] = report_locale
    validated = PipelineImageDualConfig.model_validate(idc)
    pipe["image_dual_check"] = validated.model_dump()
    data["pipeline"] = pipe
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def apply_llm_patch_to_yaml_file(path: str, patch: dict[str, Any]) -> None:
    """
    Merge patch into the llm section of YAML and write back.
    Keys with value None are skipped; call reload_settings() afterward.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    current = dict(data.get("llm") or {})
    for k, v in patch.items():
        if v is None:
            continue
        current[k] = v
    LLMConfig(**current)
    data["llm"] = current
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def get_settings(config_path: Optional[str] = None) -> Settings:
    global _settings
    if _settings is None:
        path = config_path or _default_config_path()
        _settings = Settings.from_yaml(path)
    return _settings


def reload_settings(config_path: Optional[str] = None) -> Settings:
    """Drop cache and reload from YAML (after editing config file)."""
    global _settings
    _settings = None
    return get_settings(config_path)