"""Image pHash blocklist — SQLite (requires ImageHash + Pillow)."""

from __future__ import annotations

import json

import pytest

try:
    import imagehash
    from PIL import Image
except ImportError:  # not installed or wheel/arch mismatch on this machine
    imagehash = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    imagehash is None or Image is None,
    reason="ImageHash/Pillow unavailable (pip install -e '.[dev]' or use a Pillow wheel for your arch)",
)

from reviewagent.config import (
    PipelineConfig,
    PipelineFingerprintConfig,
    PipelineWordlistConfig,
    Settings,
    StorageConfig,
)
from reviewagent.pipeline.biz_context import BizContext
from reviewagent.pipeline import image_fingerprint
from reviewagent.pipeline.image_fingerprint import try_fingerprint_early_block
from reviewagent.storage.phash_blocklist import PhashBlocklistStore


@pytest.fixture(autouse=True)
def _reset_fingerprint_warn_once() -> None:
    image_fingerprint._fingerprint_warned_keys.clear()
    yield
    image_fingerprint._fingerprint_warned_keys.clear()


def _settings_fingerprint_db(db_file: str, **fp_kw) -> Settings:
    fp = PipelineFingerprintConfig(image_phash_enabled=True, **fp_kw)
    return Settings(
        storage=StorageConfig(review_db_path=db_file),
        pipeline=PipelineConfig(wordlist=PipelineWordlistConfig(), fingerprint=fp),
    )


def test_phash_disabled_returns_none(tmp_path) -> None:
    assert imagehash is not None and Image is not None
    db = tmp_path / "x.db"
    img_path = tmp_path / "x.png"
    Image.new("RGB", (32, 32), color="green").save(img_path)
    PhashBlocklistStore(str(db)).add("ffffffffffffffff", note="x")
    s = Settings(
        storage=StorageConfig(review_db_path=str(db)),
        pipeline=PipelineConfig(
            wordlist=PipelineWordlistConfig(),
            fingerprint=PipelineFingerprintConfig(
                image_phash_enabled=False,
            ),
        ),
    )
    assert try_fingerprint_early_block(str(img_path), BizContext(), s) is None


def test_phash_hit_early_block(tmp_path) -> None:
    assert imagehash is not None and Image is not None
    db = tmp_path / "ban.db"
    img_path = tmp_path / "a.png"
    im = Image.new("RGB", (64, 64), color=(200, 30, 40))
    im.save(img_path)
    h = str(imagehash.phash(Image.open(img_path)))
    PhashBlocklistStore(str(db)).add(h, note="banned")

    s = _settings_fingerprint_db(str(db))
    r = try_fingerprint_early_block(str(img_path), BizContext(), s)
    assert r is not None
    body = json.loads(r["response"])
    assert body["verdict"] == "BLOCK"
    assert body["violations"][0]["type"] == "image_phash"
    assert r["pipeline_trace"]["mode"] == "fingerprint_match"
    assert r["pipeline_trace"]["stages"][0].get("source") == "sqlite"


def test_phash_miss_when_image_differs(tmp_path) -> None:
    assert imagehash is not None and Image is not None
    db = tmp_path / "m.db"
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.new("RGB", (64, 64), color="red").save(a)
    Image.new("RGB", (64, 64), color="blue").save(b)
    PhashBlocklistStore(str(db)).add(str(imagehash.phash(Image.open(a))))

    s = _settings_fingerprint_db(str(db))
    assert try_fingerprint_early_block(str(b), BizContext(), s) is None


def test_phash_uses_explicit_db_path(tmp_path) -> None:
    assert imagehash is not None and Image is not None
    main_db = tmp_path / "main.db"
    phash_db = tmp_path / "phash_only.db"
    img_path = tmp_path / "a.png"
    Image.new("RGB", (8, 8), color="yellow").save(img_path)
    h = str(imagehash.phash(Image.open(img_path)))
    PhashBlocklistStore(str(phash_db)).add(h)

    fp = PipelineFingerprintConfig(
        image_phash_enabled=True,
        image_phash_db_path=str(phash_db),
    )
    s = Settings(
        storage=StorageConfig(review_db_path=str(main_db)),
        pipeline=PipelineConfig(wordlist=PipelineWordlistConfig(), fingerprint=fp),
    )
    r = try_fingerprint_early_block(str(img_path), BizContext(), s)
    assert r is not None
    assert json.loads(r["response"])["verdict"] == "BLOCK"
