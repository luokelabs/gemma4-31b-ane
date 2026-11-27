#!/bin/bash
# Build the released 16K model from quantized weights (docs/quantization.md):
#   Q_FRONT=<dir with L0..L35.npz + head.npz>   4-bit LUT layers 0-35 (e2e1)
#   Q_BACK=<dir with L36..L59.npz>              3-bit vector-LUT layers 36-59 (g3e)
#   scripts/build_16k.sh <out_dir>
# Needs the original checkpoint (GEMMA_HF) for norms / embeddings. ~45 min on an M6 mini, ~15 GB of output plus
# 20-30 GB of Core AI compile cache (keep 40+ GB free: macOS evicts fresh cache entries when the disk runs low).
set -euo pipefail
cd "$(dirname "$0")/.."
D=${1:?usage: build_16k.sh <out_dir>}; PY=${PY:-.venv/bin/python}
: "${Q_FRONT:?set Q_FRONT}" "${Q_BACK:?set Q_BACK}"
# KT_ATTN: global K history pre-transposed, attention in 2048-key blocks; CTX_TIERS: extra entries that attend over
# only the first 4096 (and 8192) rows; TAPS: hidden outputs the drafters can use. A chunk keeps at most 7 entries
# on the ANE (the 8th exported falls off), hence the explicit export orders.
COMMON=(KT_ATTN=1 KT_TILE_MIN=0 TAPS=1,12,23,35,46,57 CTX_TIERS=4096)
env "${COMMON[@]}" TS=8,1,32 TS_ORDER=8c4096,1c4096,32c4096,8,1,32,32c8192 LUT_GROUP=128 \
  $PY gemma_ane.py build "$D" --ctx 16384 --chunk 6 --quant lut --qdir "$Q_FRONT" --layers 0-35
env "${COMMON[@]}" TS=8,32,64 TS_ORDER=64c4096,64,8c4096,1c4096,32c4096,8,64c8192 LUT_GROUP=0 \
  $PY gemma_ane.py build "$D" --ctx 16384 --chunk 6 --quant lut --qdir "$Q_BACK" --layers 36-59
TS=8,1,32 $PY gemma_ane.py head "$D" --quant lut --qdir "$Q_FRONT"
$PY - "$D" <<'EOF'
import json, sys
p = sys.argv[1] + "/manifest.json"; m = json.load(open(p))
m.update(Ts=[8, 1, 32, 64], tiers=[4096, 8192], kT=True, feat_layers=[1, 12, 23, 35, 46, 57])
for c in m["chunks"]: c["Ts"] = [8, 1, 32] if c["layers"][0] < 36 else [8, 1, 32, 64]
json.dump(m, open(p, "w")); print("manifest:", [c["Ts"] for c in m["chunks"]])
EOF
cp "$(${PY} -c 'from paths import HF_MODEL; print(HF_MODEL)')"/{tokenizer.json,tokenizer_config.json,chat_template.jinja,config.json,generation_config.json} "$D"/
$PY fix_cache.py "$D"
echo "built $D"
