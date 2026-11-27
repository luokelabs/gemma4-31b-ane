# Gemma 4 31B on the Apple Neural Engine

Run all of **Gemma 4 31B (dense)** on the **Apple Neural Engine (ANE)** of a 32 GB Mac through Core AI, with Google's MTP
draft model on the GPU for speculative decoding. On a Mac mini (M6, 32 GB) it generates faster than the GPU
(MLX, llama.cpp), prefills about as fast, and uses roughly half the energy per token.

[中文说明](README_zh.md) · Model weights: Hugging Face (coming soon) · Engineering notes: [docs/](docs/)

| Mac mini M6 32 GB, greedy | ANE (this repo) | GPU, MLX 4-bit | GPU, llama.cpp Q4_0 |
|---|---|---|---|
| Decode with MTP drafter, 16 short prompts | **26.1 tok/s** | 19.9 | 19.7 |
| Decode, 25 min sustained (1024-token answers) | **24.4 tok/s** | 17.6 | — |
| Energy per generated token (25 min, whole machine) | **0.92 J** (21.8 W) | 2.24 J (39.2 W) | — |
| Prefill, 500–1000 token prompts | 220–250 tok/s | 236–244 | ~169 |
| Prefill, 15K-token prompt | 69.6–75.1 s | 71.4 s | — |
| Prefill energy per prompt token (1K–15K prompts) | **108–125 mJ** | 155–194 mJ | — |
| Decode with MTP at 1K / 4K / 8K / 15K context | **32.1 / 21.4 / 20.5 / 20.1** | 17.8 / 14.3 / 13.8 / 11.2 | — |
| Peak chip temperature (25 min) | **70 °C** | 81.5 °C | — |

Details, methods and caveats: [docs/results.md](docs/results.md).

## What is in here

- `gemma_ane.py`: Gemma 4 31B rewritten for the ANE (1×1 convs, fp16-safe norms, KV history as inputs), chunk export
  to Core AI `.aimodel` packages, LUT / vector-LUT weight formats, context tiers.
- `gemma_rt.py`: runtime over the [ANEMLL Forge](https://github.com/Anemll/anemll-forge) Swift bridge: IOSurface
  buffers bound once, KV cache management, 64-token prefill, tier switching, speculative verify/commit.
- `mtp_mlx.py`: Google's MTP drafter (`gemma-4-31B-it-qat-assistant`, 4-bit MLX) on the GPU, fed by the ANE target.
- `chat.py`: interactive chat. `ane_run.py`: benchmark (prefill, decode, end-to-end, memory).
- `quant/`: the quantization pipeline (GPTQ → per-layer distillation → whole-model distillation).
- `eval/`: reference check (KL vs fp32), prompt sets, GPU baseline client, power integration.

## Requirements

- Apple silicon Mac with **32 GB** of memory (the ANE keeps ~20–23 GB wired; 24 GB machines do not fit).
- macOS 27 with Core AI, Xcode command line tools (`swiftc`), Python 3.13.
- ~16 GB of disk for the model plus 20–30 GB for the Core AI compile cache. Keep 30+ GB free: when the disk runs low,
  macOS evicts freshly compiled programs and loading fails with `nilError` (run `fix_cache.py <model_dir>`).

## Quick start

```bash
git clone https://github.com/luokelabs/gemma4-31b-ane && cd gemma4-31b-ane
scripts/setup.sh                 # venv, ANEMLL Forge at the tested commit, Swift bridge
scripts/download.sh              # ANE model (~16 GB) + MTP drafter (~0.3 GB) into ./models
export MTP_MLX=$PWD/models/gemma-4-31B-it-qat-assistant-4bit-mlx
.venv/bin/python chat.py models/gemma-4-31B-it-ane-16k
```

The first load compiles the model for the ANE (several minutes, once per macOS build). Later loads take seconds.

Benchmark like the tables above:

```bash
.venv/bin/python ane_run.py models/gemma-4-31B-it-ane-16k eval/prompts/e2e_prompts.json --max 512 --out logs/e2e.json
.venv/bin/python ane_run.py models/gemma-4-31B-it-ane-16k eval/prompts/long_prompts.json --max 128 --out logs/long.json
REF64_DIR=models/gemma-4-31B-it-ane-16k/eval .venv/bin/python eval/validate.py models/gemma-4-31B-it-ane-16k
```

For MTP runs set `PREWARM_MS=95 MLX_CACHE=268435456 WARM=30 ND_ZH=3` (what the numbers above use; `chat.py` sets the first three itself).

## How it works

- **Whole model on the ANE.** 60 layers in 10 chunks of 6 layers plus the output head, every chunk fully placed on the
  ANE (no silent GPU/CPU fallback). Activations stay fp16 without overflow (min cosine 0.9997 per layer vs fp32).
- **Mixed low-bit weights, ~3.8 bits/weight.** Layers 0–35: 4-bit per-tensor LUT with per-channel scales; layers
  36–59: 3-bit vector LUT (pairs of weights, 64-entry codebook) on Hadamard-rotated weights. GPTQ, then per-layer and
  whole-model distillation against the fp32 model. On the ANE, fewer bits save bandwidth, not wired memory.
- **Speculative decoding across ANE and GPU.** Verifying 8 tokens costs about as much as 1 on the ANE (weight reads
  dominate), so the GPU drafter proposes 7 tokens (~23 ms) and the ANE verifies them in one call. Matching the
  drafter to the QAT target matters: the generic drafter accepted 1.33 tokens per round, the QAT one 3.49.
- **Fast prefill.** 64-token prefill entries (32 for the grouped-LUT front chunks, run twice), global attention over the
  context in 2048-key blocks with the K history stored pre-transposed, and context tiers (entries that attend over
  only the first 4K or 8K rows) so short prompts do not pay for the 16K window.
- **A chunk keeps at most 7 entry points on the ANE**; an 8th exported function silently runs off the ANE
  (100+ ms per call, ~5 GB more wired memory). The export order in `scripts/build_16k.sh` follows from this.

More in [docs/engineering_notes.md](docs/engineering_notes.md).

## Quality

| | KL vs fp32 (ref64) | top-1 agreement | IFEval (strict, prompt) | HumanEval+ | CMMLU |
|---|---|---|---|---|---|
| This model (16K) | 0.220 | 59/64 | 82.8%* | 90.6%* | 67.2%* |
| Gemma 4 31B QAT, MLX 4-bit (GPU) | 0.172 | 61/64 | 85.9% | 93.8% | 68.8% |

\* 160-item pilot (64 IFEval, 32 HumanEval+, 64 CMMLU) on the 4K build with identical weights; the 16K build matches it
within fp16 noise (KL 0.2202 vs 0.2196). Close to, but slightly below, the 4.5-bit GPU model; small sample.

## Limitations

- 32 GB Macs only, and little headroom: with the drafter running, 3–5 GB of memory stays free.
- When the GPU is saturated by another job, ANE decode drops by about half (shared memory bandwidth).
- Occasional 300–900 ms ANE call stalls (system-level; adds 0–2 s to a long prompt).
- Greedy decoding only (the speculative verify is exact for greedy).
- Measured on one machine (Mac mini M6 32 GB, macOS 27.0.1). Other chips have different ANE/GPU balances.

## Reproducing the weights

`docs/quantization.md` lists the full chain (GPTQ, per-layer and whole-model distillation) with the commands used.
It takes 1–2 days on the M6 mini. `scripts/build_16k.sh` turns quantized weights into the released model.

## License and credits

Code: Apache-2.0. Model weights: derived from [google/gemma-4-31B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized)
(Apache-2.0, [Gemma 4 terms](https://ai.google.dev/gemma/docs/gemma_4_license)). The Core AI Swift bridge and LUT helpers
come from [ANEMLL Forge](https://github.com/Anemll/anemll-forge) (MIT), fetched by `scripts/setup.sh`. The MTP drafter is
[mlx-community/gemma-4-31B-it-qat-assistant-4bit](https://huggingface.co/mlx-community/gemma-4-31B-it-qat-assistant-4bit),
run with [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (MIT). See [NOTICE](NOTICE).
