#!/usr/bin/env bash
# Content review agent: create a venv and install this package after unpack.
# Usage: chmod +x install.sh && ./install.sh
#        ./install.sh --extras image,video
# Note: video extra includes moviepy + openai-whisper.
#
# Windows: use Git Bash or MSYS2 (same script). Native venv uses .venv/Scripts/python.exe.
# WSL: treat as Linux. cmd/PowerShell: create venv manually, then pip install -e .

set -euo pipefail

# Bundled install.sh lives next to pyproject.toml; from repo source it may be under scripts/bundle/ — walk up to repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
while [[ "$ROOT" != "/" ]]; do
  if [[ -f "$ROOT/pyproject.toml" ]]; then
    break
  fi
  PARENT="$(dirname "$ROOT")"
  if [[ "$PARENT" == "$ROOT" ]]; then
    ROOT=""
    break
  fi
  ROOT="$PARENT"
done
if [[ ! -f "$ROOT/pyproject.toml" ]]; then
  echo "Error: pyproject.toml not found when walking up from \"$SCRIPT_DIR\"." >&2
  echo "  · From source: run from repo root, e.g. bash scripts/bundle/install.sh, or cd to the directory that contains pyproject.toml." >&2
  echo "  · From a bundle archive: run ./install.sh in the top-level folder next to pyproject.toml." >&2
  exit 1
fi
cd "$ROOT"

EXTRAS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --extras)
      EXTRAS="${2:-}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--extras EXTRAS]   e.g. --extras image,video"
      echo "       video extra includes moviepy + openai-whisper"
      echo "       Requires Python ≥ 3.9. Override: PYTHON=python3.12 $0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

py_ge_39() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null
}

# Host interpreter: PYTHON env, then python3 / python, versioned names, then Windows "py -3".
resolve_host_python() {
  local c exe
  for c in "${PYTHON:-}" python3 python python3.12 python3.11 python3.10 python3.9; do
    [[ -z "$c" ]] && continue
    if [[ -x "$c" ]] || command -v "$c" >/dev/null 2>&1; then
      if py_ge_39 "$c"; then
        if [[ -x "$c" ]]; then
          echo "$c"
        else
          command -v "$c"
        fi
        return 0
      fi
    fi
  done
  if command -v py >/dev/null 2>&1; then
    exe="$(py -3 -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
    if [[ -n "$exe" ]] && [[ -f "$exe" || -x "$exe" ]] && py_ge_39 "$exe"; then
      echo "$exe"
      return 0
    fi
  fi
  return 1
}

# Interpreter inside venv: Windows uses Scripts/python.exe; Unix uses bin/python3|python.
venv_bin_python() {
  local v="$1" p
  for p in "$v/Scripts/python.exe" "$v/Scripts/python" "$v/bin/python3" "$v/bin/python"; do
    [[ -f "$p" || -x "$p" ]] || continue
    if py_ge_39 "$p"; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

PY="$(resolve_host_python || true)"
if [[ -z "${PY:-}" ]]; then
  echo "Error: need Python 3.9+ (tried PYTHON, python3, python, python3.12 … python3.9)." >&2
  echo "  Install Python 3.9+, or on Windows install from python.org and ensure \"py\" or \"python\" is on PATH." >&2
  echo "  Override:  PYTHON=/path/to/python3 bash \"$0\"" >&2
  exit 1
fi

echo "Using interpreter: $($PY -c 'import sys; print(sys.executable)')"

VENV="$ROOT/.venv"
VPY=""
if [[ -d "$VENV" ]]; then
  VPY="$(venv_bin_python "$VENV" || true)"
fi
if [[ -z "${VPY:-}" ]]; then
  if [[ -d "$VENV" ]]; then
    echo "Existing .venv is missing or unusable; recreating..."
    rm -rf "$VENV"
  fi
  echo "Creating virtual environment: $VENV"
  if "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
    "$PY" -m venv --upgrade-deps "$VENV"
  else
    "$PY" -m venv "$VENV"
  fi
  VPY="$(venv_bin_python "$VENV" || true)"
fi
if [[ -z "${VPY:-}" ]]; then
  echo "Error: virtualenv at \"$VENV\" has no usable Python (≥ 3.9)." >&2
  echo "  Expected Unix: .venv/bin/python3 or bin/python; Windows: .venv/Scripts/python.exe" >&2
  echo "  Try: rm -rf \"$VENV\" && bash \"$0\"" >&2
  exit 1
fi

# Prefer "python -m pip": some systems ship venv without a bin/pip launcher.
if ! "$VPY" -m pip --version >/dev/null 2>&1; then
  echo "Bootstrapping pip in venv (ensurepip)..."
  "$VPY" -m ensurepip --upgrade 2>/dev/null || true
fi
if ! "$VPY" -m pip --version >/dev/null 2>&1; then
  echo "Error: pip is not available inside the venv. Try: $PY -m venv --upgrade-deps \"$VENV\"" >&2
  echo "  Or install OS packages: python3-venv / ensure your Python build includes ensurepip." >&2
  exit 1
fi

"$VPY" -m pip install -q -U pip setuptools wheel

if [[ -n "$EXTRAS" ]]; then
  echo "Installing project with optional extras: [$EXTRAS]"
  "$VPY" -m pip install -q ".[${EXTRAS}]"
else
  echo "Installing project (core dependencies)..."
  "$VPY" -m pip install -q "."
fi

echo ""
echo "Done."
if [[ -f "$VENV/Scripts/activate" ]]; then
  echo "  Activate (Git Bash):  source \"$VENV/Scripts/activate\""
  echo "  Activate (cmd):       .venv\\Scripts\\activate.bat"
else
  echo "  Activate venv:  source \"$VENV/bin/activate\""
fi
echo "  Start server:   content-review server"
echo "  Or:             python \"$ROOT/cli.py\" server"
echo ""
if [[ ! -f "$ROOT/.env" ]]; then
  echo "Tip: copy .env.example to .env and add your API keys if you have not already."
fi
