#!/usr/bin/env python3
"""
Bundle full source plus one-shot install scripts into dist/*.tar.gz and *.zip.
Run from repo root: python3 scripts/package_bundle.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import stat
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "scripts" / "bundle"

EXCLUDE_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".cursor",
    ".pytest_cache",
    ".ruff_cache",
    ".env",
    ".DS_Store",
}


def read_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError("could not parse version from pyproject.toml")
    return m.group(1)


def ignore_copy(path: str, names: list[str]) -> list[str]:
    skipped: list[str] = []
    for name in names:
        if name in EXCLUDE_NAMES:
            skipped.append(name)
        elif name.endswith(".egg-info"):
            skipped.append(name)
        elif name.endswith(".pyc"):
            skipped.append(name)
        elif name.endswith(".db"):
            skipped.append(name)
    return skipped


def chmod_x(path: Path) -> None:
    if not sys.platform.startswith("win"):
        mode = path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        path.chmod(mode)


def stage_bundle(version: str) -> Path:
    name = f"content-review-{version}"
    dist = ROOT / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    stage = dist / name
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(ROOT, stage, ignore=ignore_copy, dirs_exist_ok=False)

    for fname in ("install.sh", "INSTALL.txt"):
        src = BUNDLE_DIR / fname
        if not src.is_file():
            raise FileNotFoundError(f"missing bundle asset: {src}")
        shutil.copy2(src, stage / fname)

    chmod_x(stage / "install.sh")
    return stage


def make_tar_gz(stage: Path, version: str) -> Path:
    dist = ROOT / "dist"
    out = dist / f"content-review-{version}-bundle.tar.gz"
    if out.exists():
        out.unlink()

    with tarfile.open(out, "w:gz", format=tarfile.GNU_FORMAT) as tf:
        try:
            tf.add(stage, arcname=stage.name, filter=lambda ti: ti)
        except TypeError:
            # Python < 3.12
            tf.add(stage, arcname=stage.name)

    return out


def make_zip(stage: Path, version: str) -> Path:
    dist = ROOT / "dist"
    out = dist / f"content-review-{version}-bundle.zip"
    if out.exists():
        out.unlink()

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                arc = f"{stage.name}/{f.relative_to(stage).as_posix()}"
                zf.write(f, arc)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Package source + install scripts into dist/")
    ap.add_argument(
        "--no-zip",
        action="store_true",
        help="skip .zip (tar.gz only)",
    )
    args = ap.parse_args()

    if not (ROOT / "pyproject.toml").is_file():
        print("Run from repo root: python3 scripts/package_bundle.py", file=sys.stderr)
        return 1

    version = read_version()
    print(f"version: {version}")
    print("staging source under dist/ …")
    stage = stage_bundle(version)
    print(f"stage dir: {stage}")

    tg = make_tar_gz(stage, version)
    print(f"wrote: {tg}")

    if not args.no_zip:
        zp = make_zip(stage, version)
        print(f"wrote: {zp}")

    shutil.rmtree(stage)
    print("removed stage dir (archives only).")

    print("\nExtract an archive and run install.sh (macOS/Linux, or Git Bash/WSL on Windows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
