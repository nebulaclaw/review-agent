"""Perceptual hash (pHash) blocklist in SQLite; early BLOCK on hit (independent of OCR)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from reviewagent.config import Settings
from reviewagent.pipeline.biz_context import BizContext
from reviewagent.storage.phash_blocklist import PhashBlocklistStore

logger = logging.getLogger(__name__)
_MC = "[review.core]"

_fingerprint_warned_keys: set[str] = set()


def _warn_once(key: str, fmt: str, *args: Any) -> None:
    if key in _fingerprint_warned_keys:
        return
    _fingerprint_warned_keys.add(key)
    logger.warning("%s " + fmt, _MC, *args)


def _fingerprint_sqlite_path(settings: Settings) -> str:
    fp_cfg = settings.pipeline.fingerprint
    explicit = (fp_cfg.image_phash_db_path or "").strip()
    if explicit:
        return explicit
    return (settings.storage.review_db_path or "").strip()


def log_fingerprint_config_warnings(settings: Settings) -> None:
    """Log warnings for image fingerprint config at HTTP startup."""
    fp_cfg = settings.pipeline.fingerprint
    if not fp_cfg.image_phash_enabled:
        return
    db_path = _fingerprint_sqlite_path(settings)
    if not db_path:
        logger.warning(
            "%s [startup] image fingerprint: image_phash_enabled=true but no SQLite path "
            "(set pipeline.fingerprint.image_phash_db_path or storage.review_db_path).",
            _MC,
        )
        return
    try:
        store = PhashBlocklistStore(db_path)
        n = store.active_count()
        if n == 0:
            logger.warning(
                "%s [startup] image fingerprint: table image_phash_blocklist has no active rows "
                "(disabled=0); matches pass until fingerprints are inserted. db=%s",
                _MC,
                db_path,
            )
    except OSError as e:
        logger.warning(
            "%s [startup] image fingerprint: cannot open SQLite %s: %s",
            _MC,
            db_path,
            e,
        )


def _phash_block_response(matched_hex: str, query_hex: str, hamming: int, *, locale: str = "zh") -> str:
    if locale == "en":
        summary = "Image matches a platform-banned fingerprint (perceptual hash); the request was auto-blocked."
    else:
        summary = "图片命中平台封禁指纹（感知哈希），已被系统自动拦截。"
    body = {
        "verdict": "BLOCK",
        "confidence": 1.0,
        "violations": [
            {
                "type": "image_phash",
                "content": matched_hex,
                "severity": "high",
                "position": f"hamming={hamming};image_phash={query_hex}",
            }
        ],
        "summary": summary,
    }
    return json.dumps(body, ensure_ascii=False)


def try_fingerprint_early_block(
    image_path: str,
    biz: BizContext,
    settings: Settings,
) -> Optional[dict[str, Any]]:
    fp_cfg = settings.pipeline.fingerprint
    if not fp_cfg.image_phash_enabled:
        return None

    db_path = _fingerprint_sqlite_path(settings)
    if not db_path:
        _warn_once(
            "fingerprint_no_db_path",
            "image fingerprint enabled but pipeline.fingerprint.image_phash_db_path and "
            "storage.review_db_path are empty; cannot load blocklist.",
        )
        return None

    try:
        import imagehash
        from PIL import Image
    except ImportError:
        logger.warning(
            "%s fingerprint_phash image_phash_enabled but ImageHash/Pillow missing; install .[image]",
            _MC,
        )
        return None

    p = Path(image_path)
    if not p.is_file():
        _warn_once(
            "fingerprint_image_not_a_file",
            "image fingerprint skipped: content must be a local readable image path "
            "(not a readable file) brief=%r",
            (image_path[:300] + ("…" if len(image_path) > 300 else "")),
        )
        return None

    t_all0 = time.perf_counter()
    trace: dict[str, Any] = {
        "mode": "fingerprint_match",
        "biz_line": biz.biz_line,
        "tenant_id": biz.tenant_id,
        "trust_tier": biz.trust_tier,
        "audience": biz.audience,
        "policy_pack_id": biz.policy_pack_id,
        "stages": [],
    }

    t0 = time.perf_counter()
    try:
        store = PhashBlocklistStore(db_path)
        hashes = store.list_active_parsed_hashes(imagehash)
    except OSError as e:
        _warn_once(
            f"fingerprint_db_open:{db_path}",
            "image fingerprint: cannot open or init SQLite blocklist %s: %s",
            db_path,
            e,
        )
        return None

    load_ms = (time.perf_counter() - t0) * 1000.0
    trace["stages"].append(
        {
            "name": "blocklist_load",
            "ms": round(load_ms, 3),
            "source": "sqlite",
            "db_path": db_path,
            "entry_count": len(hashes),
        }
    )

    if not hashes:
        _warn_once(
            f"fingerprint_blocklist_empty:{db_path}",
            "image fingerprint: table image_phash_blocklist has no usable rows "
            "(disabled=0 and parseable hex); skipped compare. "
            "Use PhashBlocklistStore.add() or SQL; for debug set image_phash_log_on_miss=true. db=%s",
            db_path,
        )
        return None

    t0 = time.perf_counter()
    try:
        with Image.open(p) as img:
            h = imagehash.phash(img)
    except Exception as e:
        ph_ms = (time.perf_counter() - t0) * 1000.0
        trace["stages"].append(
            {"name": "phash_compute", "ms": round(ph_ms, 3), "ok": False, "error": str(e)}
        )
        trace["duration_ms"] = round((time.perf_counter() - t_all0) * 1000.0, 3)
        logger.info("%s fingerprint_phash compute_failed path=%s err=%s", _MC, p, e)
        return None

    ph_ms = (time.perf_counter() - t0) * 1000.0
    query_hex = str(h)
    trace["stages"].append(
        {"name": "phash_compute", "ms": round(ph_ms, 3), "ok": True, "phash": query_hex}
    )

    max_d = fp_cfg.image_phash_max_hamming
    best: Optional[tuple[str, int]] = None
    for ref_hex, ref_h in hashes:
        try:
            d = h - ref_h
        except Exception:
            continue
        if d <= max_d:
            best = (ref_hex, int(d))
            break

    match_ms = 0.0
    trace["stages"].append(
        {
            "name": "blocklist_match",
            "ms": round(match_ms, 3),
            "max_hamming": max_d,
            "hit": best is not None,
            "matched_hex": best[0] if best else None,
            "hamming": best[1] if best else None,
        }
    )

    duration_ms = (time.perf_counter() - t_all0) * 1000.0
    trace["duration_ms"] = round(duration_ms, 3)

    if best is None:
        if fp_cfg.image_phash_log_on_miss:
            logger.info(
                "%s fingerprint_phash miss decision=CONTINUE_LLM phash=%s image=%s",
                _MC,
                query_hex,
                p.name,
            )
        return None

    matched_hex, hamming = best
    locale = settings.pipeline.image_dual_check.report_locale or "zh"
    logger.info(
        "%s fingerprint_phash decision=EARLY_BLOCK matched=%s hamming=%s",
        _MC,
        matched_hex,
        hamming,
    )
    return {
        "success": True,
        "response": _phash_block_response(matched_hex, query_hex, hamming, locale=locale),
        "iterations": 0,
        "duration_ms": round(duration_ms, 2),
        "pipeline_trace": trace,
    }


__all__ = ["log_fingerprint_config_warnings", "try_fingerprint_early_block"]
