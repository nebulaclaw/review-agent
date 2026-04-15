from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_OCR_PREVIEW_MAX = 400

# Lazy EasyOCR reader + serialized inference to avoid concurrent init races
_ocr_lock = threading.Lock()
_easyocr_reader: Any = None


def _ocr_preview(text: str, limit: int = _OCR_PREVIEW_MAX) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = t.strip()
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"


def _prepare_image_for_ocr(img: Any) -> Any:
    """Preprocess for OCR: flatten transparency, upscale small images, etc."""
    from PIL import Image, ImageOps, ImageStat

    if img.mode in ("RGBA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Dark UI / light-on-dark: invert when mean brightness is low (EasyOCR prefers dark-on-light)
    stat = ImageStat.Stat(img)
    lum = float(sum(stat.mean)) / float(len(stat.mean))
    if lum < 88.0:
        img = ImageOps.invert(img)

    w, h = img.size
    if min(w, h) > 0 and max(w, h) < 500:
        scale = max(2, int(500 / max(w, h)))
        nw, nh = w * scale, h * scale
        if nw * nh <= 25_000_000:
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS  # type: ignore[attr-defined]
            img = img.resize((nw, nh), resample)
    return img


def _ocr_easyocr_rgb(img: Any) -> tuple[str, list[str]]:
    """
    EasyOCR with simplified Chinese + English. Returns (merged text, raw line list).
    """
    import numpy as np

    global _easyocr_reader

    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    with _ocr_lock:
        if _easyocr_reader is None:
            import easyocr

            try:
                import torch

                use_gpu = bool(torch.cuda.is_available())
            except Exception:
                use_gpu = False
            _easyocr_reader = easyocr.Reader(
                ["ch_sim", "en"],
                gpu=use_gpu,
                verbose=False,
            )
            logger.info(
                "easyocr_reader_ready gpu=%s (first run may download models to local cache)",
                use_gpu,
            )
        raw = _easyocr_reader.readtext(arr)

    line_texts: list[str] = []
    for item in raw:
        if len(item) >= 2:
            t = str(item[1]).strip()
            if t:
                line_texts.append(t)

    seen: set[str] = set()
    merged_lines: list[str] = []
    for t in line_texts:
        if t not in seen:
            seen.add(t)
            merged_lines.append(t)
    return "\n".join(merged_lines), line_texts


class ImageDetector:
    def __init__(self) -> None:
        self.ocr_enabled = True
        self.nsfw_detection = False

    def _ocr_detect_sync(self, image_path: str) -> dict:
        pil_err: str | None = None
        easy_err: str | None = None
        try:
            from PIL import Image
        except ImportError as e:
            pil_err = str(e).strip() or repr(e)

        try:
            import easyocr  # noqa: F401
        except ImportError as e:
            easy_err = str(e).strip() or repr(e)

        if pil_err is not None or easy_err is not None:
            logger.warning(
                "image_ocr skipped path=%s python=%s pil_error=%r easyocr_error=%r "
                "(install deps in the same interpreter that runs serve: %s -m pip install -e '.[image]')",
                Path(image_path).name,
                sys.executable,
                pil_err,
                easy_err,
                Path(sys.executable).name,
            )
            return {
                "text": "",
                "violations": [],
                "note": "pillow/easyocr (and torch, etc.) not installed; OCR skipped. Try: pip install -e '.[image]'",
                "pil_import_error": pil_err,
                "easyocr_import_error": easy_err,
                "python_executable": sys.executable,
            }

        try:
            img = Image.open(image_path)
            img = _prepare_image_for_ocr(img)
            text, raw_lines = _ocr_easyocr_rgb(img)

            from reviewagent.toolpacks.text_detector import TextDetector

            detector = TextDetector()
            result = detector.detect(text)

            meta: dict = {
                "ocr_engine": "easyocr",
                "ocr_raw_pass_count": len(raw_lines),
            }
            if not text and not raw_lines:
                meta["ocr_hint"] = "easyocr_empty"

            return {
                "text": text,
                "violations": result.get("violations", []),
                **meta,
            }
        except Exception as e:
            logger.exception("image_ocr failed path=%s", Path(image_path).name)
            return {"text": "", "violations": [], "ocr_error": str(e)[:200]}

    def detect_sync(self, image_path: str) -> dict:
        """Synchronous detection; safe from async agent / LangChain tool threads."""
        path = Path(image_path)

        if not path.exists():
            logger.warning("image_detect file_missing path=%s", path.name)
            return {
                "success": False,
                "error": f"Image not found: {image_path}",
            }

        violations: list = []
        detected_text = ""
        ocr_meta: dict = {}

        if self.ocr_enabled:
            text_result = self._ocr_detect_sync(str(path))
            detected_text = text_result.get("text", "")
            ocr_meta = {k: v for k, v in text_result.items() if k not in ("text", "violations")}
            if text_result.get("violations"):
                violations.extend(text_result["violations"])

            n = len(detected_text or "")
            logger.info(
                "image_ocr_done file=%s text_chars=%d has_text=%s ocr_engine=%s ocr_lines=%s note=%s ocr_hint=%s ocr_error=%s",
                path.name,
                n,
                bool(detected_text and detected_text.strip()),
                ocr_meta.get("ocr_engine"),
                ocr_meta.get("ocr_raw_pass_count"),
                text_result.get("note"),
                ocr_meta.get("ocr_hint"),
                text_result.get("ocr_error"),
            )
            if n > 0:
                logger.debug("image_ocr_preview file=%s text=%r", path.name, _ocr_preview(detected_text))
        elif not self.ocr_enabled:
            logger.info("image_ocr_disabled file=%s", path.name)

        return {
            "success": True,
            "verdict": "BLOCK" if violations else "PASS",
            "confidence": 0.8,
            "violations": violations,
            "details": {
                "detected_text": detected_text,
                "has_text": bool(detected_text),
                **ocr_meta,
            },
        }

    async def detect(self, image_path: str) -> dict:
        """Async API preserved; delegates to sync to avoid nested asyncio.run."""
        return self.detect_sync(image_path)

    def __call__(self, image_path: str) -> dict:
        return self.detect_sync(image_path)
