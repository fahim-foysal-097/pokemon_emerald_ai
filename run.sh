#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "[launcher] Missing project virtual environment: $PYTHON" >&2
  echo "Activate/create the project's .venv first, then run this launcher." >&2
  exit 2
fi

# Prefer an explicitly supplied native build directory, then the user's known mGBA
# build, then a project-local build if one exists.
if [[ -n "${MGBA_BUILD_DIR:-}" ]]; then
  BUILD_DIR="$MGBA_BUILD_DIR"
elif [[ -d "/home/fahim/mgba/build-py311" ]]; then
  BUILD_DIR="/home/fahim/mgba/build-py311"
elif [[ -d "$ROOT/.runtime/mgba-build-py311" ]]; then
  BUILD_DIR="$ROOT/.runtime/mgba-build-py311"
else
  echo "[launcher] Could not find the mGBA native build directory." >&2
  echo "Set MGBA_BUILD_DIR to the directory containing libmgba.so, e.g.:" >&2
  echo "  MGBA_BUILD_DIR=/path/to/mgba/build-py311 ./run.sh --profile fast_train" >&2
  exit 2
fi

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "[launcher] MGBA_BUILD_DIR is not a directory: $BUILD_DIR" >&2
  exit 2
fi

export LD_LIBRARY_PATH="$BUILD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

printf '[launcher] Project: %s\n' "$ROOT"
printf '[launcher] Python:  %s\n' "$PYTHON"
printf '[launcher] mGBA:    %s\n' "$BUILD_DIR"

if [[ "${1:-}" == "--auto" || "${1:-}" == "auto" ]]; then
  shift
  exec "$PYTHON" -u "$ROOT/run_autonomous.py" "$@"
fi

exec "$PYTHON" -u "$ROOT/main.py" "$@"
