#!/bin/bash
# One-time setup: Python venv, ANEMLL Forge (Swift bridge to Core AI) at the tested commit, bridge build.
#   scripts/setup.sh            (env PYTHON=/path/to/python3.13 to pick the interpreter; otherwise python3.13 on PATH or uv)
set -euo pipefail
cd "$(dirname "$0")/.."
FORGE_URL=https://github.com/Anemll/anemll-forge.git
FORGE_COMMIT=703131ebaa8fb43180926e2fbe4480ceaa445bdf   # tested 2026-11-25

# Python 3.13: $PYTHON if set, else python3.13 on PATH, else let uv provide one
if [ ! -d .venv ]; then
  PY=${PYTHON:-$(command -v python3.13 || true)}
  if [ -n "$PY" ]; then "$PY" -m venv .venv
  elif command -v uv >/dev/null; then uv venv -q --seed --python 3.13 .venv
  else
    echo "Python 3.13 not found. Install it, then rerun scripts/setup.sh. Either:" >&2
    echo "  brew install python@3.13" >&2
    echo "  or install uv (https://docs.astral.sh/uv/getting-started/installation/); setup.sh then fetches Python 3.13 itself" >&2
    echo "  or point PYTHON=/path/to/python3.13 at an existing interpreter" >&2
    exit 1
  fi
fi
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -r requirements.txt
# mlx-vlm only provides the MTP drafter class; its full dependency set (audio, vision, a newer transformers) is not needed
.venv/bin/pip install -q --no-deps mlx-vlm==0.7.12

mkdir -p third_party
[ -d third_party/anemll-forge/.git ] || git clone -q "$FORGE_URL" third_party/anemll-forge
git -C third_party/anemll-forge fetch -q origin
git -C third_party/anemll-forge checkout -q "$FORGE_COMMIT"
bash third_party/anemll-forge/coreai/swift_bridge/build.sh > /dev/null
echo "setup done: .venv, third_party/anemll-forge @ ${FORGE_COMMIT:0:8}, libcoreai_bridge.dylib built"
