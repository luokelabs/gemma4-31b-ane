# Quantization

The released weights come from two chains of GPTQ plus distillation, all run in PyTorch on the Mac's GPU (MPS),
one layer at a time (the fp32 model does not fit in 32 GB). This page lists the steps and commands as they were run.
The chain was not re-run end to end from this repo, so treat it as a recipe, not a one-click script; expect 1–2 days
on an M6 mini.

## Formats

| Layers | Format | Bits/weight | Why |
|---|---|---|---|
| 0–35 | 4-bit LUT: one 16-entry fp16 table per tensor, uint8 indices, fp16 scale per output channel | ~4 | errors made early get amplified by later layers |
| 36–59 | 3-bit vector LUT: pairs of weights indexed into a 64-entry 2-D codebook (`vector 2x64 + pcs`), on block-Hadamard-rotated weights | ~3 | later layers mostly amplify incoming error; their own quantization matters less |
| head | 4-bit LUT (from the front chain) | ~4 | |

About 3.8 bits/weight overall (estimated from file sizes). Each matrix is one `.npz` (`lut`/`vlut`, `idx`, `scale`).
The rotations are applied online in the exported model (`quant/qfmt.py` and `gemma_ane.Hadamard` share seeds).

ANE note: with one table per tensor, the ANE compiler stores 4-bit weights at about 6.5 bits in wired memory. Writing
the same table once per 128 output channels (`LUT_GROUP=128`) makes it store real 4 bits: 0.40 → 0.30 GB per layer,
4.6 GB less wired memory overall, identical outputs. The released build does this for layers 0–35.

## Inputs

```bash
export GEMMA_HF=~/models/gemma-4-31B-it-qat-q4_0-unquantized    # hf download google/gemma-4-31B-it-qat-q4_0-unquantized
mkdir -p work/calib && cp data/chat_ids.json work/calib/
hf download Salesforce/wikitext --repo-type dataset --include "wikitext-2-raw-v1/*" --local-dir work/calib/wikitext
```

Calibration = 200 WikiText-2 rows (`NCAL_WIKI=200`, 384 tokens each) + 24 chat rows; the remaining chat rows and a
WikiText split are held out for evaluation. `data/chat_ids.json` holds greedy answers of an earlier build to the prompts
in `quant/gen_calib.py` (regenerate with it if you like).

## Front chain (layers 0–35 and head): `e2e1`

```bash
cd quant
# 1. GPTQ, 4-bit LUT, all 60 layers (damping picked per matrix on held-out rows)
NCAL_WIKI=200 DAMPS=0.05,0.3,1 METHOD=gptq OUT=../work/q/gptq4b python gptq_gemma.py
# 2. per-layer distillation: indices frozen, train the 16 LUT values + channel scales so each quantized layer on
#    the quantized stream matches the fp layer on the fp stream; also save the stream entering layer 36
QDIR=../work/q/gptq4b DOUT=../work/q/dist1 STEPS=150 LR=1e-2 SAVE_STREAM_AT=36 \
  STREAM_OUT=../work/stream_L36_dist1.pt python distill_gemma.py
# 3. 6-bit GPTQ for layers 36-59, then distill them on top of dist1's stream
#    (the original g6 was run by hand; START/CACHE let gptq_gemma.py start from a saved stream the same way)
METHOD=gptq NBITS=6 START=36 END=60 NCAL_WIKI=200 OUT=../work/q/g6 python gptq_gemma.py
QDIR=../work/q/g6 DOUT=../work/q/d6 START=36 END=60 STEPS=150 LR=1e-2 \
  CACHE=../work/stream_L36_dist1.pt python distill_gemma.py
cp ../work/q/dist1/L{0..35}.npz ../work/q/dist1/head.npz ../work/q/d6/
# 4. whole-model distillation of every LUT value and scale against the fp32 logits (KL); streams the 60 layers
#    forward and back each step. The fp teacher is computed once and cached in TEACHER.
NCAL_WIKI=200 QDIR=../work/q/d6 DOUT=../work/q/e2e1 STEPS=100 EVAL_EVERY=25 LR=4e-4 WARM=10 \
  TEACHER=../work/teacher_final.pt python e2e_distill.py
```

Only layers 0–35 and the head of `e2e1` are used in the release; the 6-bit back half only served as a better context
for training the front.

## Back chain (layers 36–59): `g3e`

```bash
# 1. GPTQ with the 3-bit vector LUT on Hadamard-rotated MLP weights (attention stays 4-bit LUT)
METHOD=gptq NCAL_WIKI=200 DAMPS=0.05 MLP_FMT="vector 2x64 + pcs" ROT=1 OUT=../work/q/g3r python gptq_gemma.py
cp ../work/q/e2e1/head.npz ../work/q/g3r/
# 2. per-layer distillation, 3. whole-model distillation (same as the front chain)
NCAL_WIKI=200 QDIR=../work/q/g3r DOUT=../work/q/g3d START=0 END=60 STEPS=100 LR=1e-2 python distill_gemma.py
NCAL_WIKI=200 QDIR=../work/q/g3d DOUT=../work/q/g3e STEPS=100 EVAL_EVERY=25 LR=4e-4 WARM=10 \
  TEACHER=../work/teacher_final.pt python e2e_distill.py
```

The original `g3r` run also mixed in 40 rows of self-generated drafter training data (`DF_ROWS=40`, `DF_DIR`),
which is not part of this repo; `DF_ROWS=0` (the default) uses WikiText and chat rows only.

## Build

```bash
Q_FRONT=work/q/e2e1 Q_BACK=work/q/g3e scripts/build_16k.sh models/my-gemma-ane-16k
REF64_DIR=work/ref64 .venv/bin/python eval/validate.py models/my-gemma-ane-16k   # expect KL ≈ 0.22, 59/64
```

`work/ref64` is the fp32 reference (`ids.npy`, `logits.npy`); the copy shipped with the model under `eval/` works too.

## How each step helped

KL vs fp32 on `ref64`, measured on the ANE:

| Step | KL | top-1 | Note |
|---|---:|---:|---|
| plain 4-bit LUT (round to nearest) | 0.529 | 76.6% | not usable |
| + GPTQ (82K calibration tokens) | 0.331 | 76.6% | error builds up in later layers |
| + per-layer distillation | 0.214 | 82.8% | helps layers 0–20 most, little after 36 |
| + whole-model distillation (4-bit front, 6-bit back) `e2e1` | **0.164** | **92.2%** | best quality, too big/slow at 6 bits |
| released mix (4-bit front from `e2e1`, 3-bit vector back from `g3e`) | 0.220 | 92.2% | ~3.8 bits, fastest; same 160-item scores as `e2e1` |

Things that did not help: GPTAQ (overfits), an 8-bit head (almost no change), keeping the back half at full precision
(signal-to-noise still falls, because the back layers amplify error from the front).
