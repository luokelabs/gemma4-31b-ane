#!/bin/bash
# One-time setup: Python venv, ANEMLL Forge (Swift bridge to Core AI) at the tested commit, bridge build.
#   scripts/setup.sh            (env PYTHON=python3.13 to pick the interpreter)
set -euo pipefail
cd "$(dirname "$0")/.."
FORGE_URL=https://github.com/Anemll/anemll-forge.git
FORGE_COMMIT=703131ebaa8fb43180926e2fbe4480ceaa445bdf   # tested 2026-11-25

PY=${PYTHON:-python3.13}
[ -d .venv ] || "$PY" -m venv .venv
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
