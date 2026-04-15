#!/usr/bin/env bash
# Run from repo root: writes dist/content-review-<version>-bundle.{tar.gz,zip}
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python3 scripts/package_bundle.py "$@"
