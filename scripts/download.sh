#!/bin/bash
# Download the converted ANE model (~16 GB) and the MTP drafter for the GPU (~0.3 GB).
#   scripts/download.sh [dest_dir]      (default: ./models)
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:-models}
ANE_REPO=${ANE_REPO:-__HF_REPO__}
[ "$ANE_REPO" = __HF_REPO__ ] && { echo "the Hugging Face repo is not published yet; set ANE_REPO=<org>/<repo>" >&2; exit 1; }
HF=.venv/bin/hf; [ -x "$HF" ] || HF=hf
"$HF" download "$ANE_REPO" --local-dir "$DEST/gemma-4-31B-it-ane-16k"
"$HF" download mlx-community/gemma-4-31B-it-qat-assistant-4bit --local-dir "$DEST/gemma-4-31B-it-qat-assistant-4bit-mlx"
cat <<EOF

done. Run:
  export MTP_MLX=$PWD/$DEST/gemma-4-31B-it-qat-assistant-4bit-mlx
  .venv/bin/python chat.py $DEST/gemma-4-31B-it-ane-16k
EOF
