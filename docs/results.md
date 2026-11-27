# Results

All numbers measured on one Mac mini (Apple M6, 32 GB), macOS 27.0.1, greedy decoding, thinking off.
GPU baselines: `mlx-vlm` 0.7.12 / `mlx` 0.32.4 with `mlx-community/gemma-4-31B-it-qat-4bit` (+ the same QAT MTP
drafter), and `llama.cpp` b12651 with Q4_0. Same prompts on both sides.

Model versions. The released model is **`gemma-4-31B-it-ane-16k`** (internal name `m3k-16k`). Some numbers were
measured on earlier builds with the **same weights** but different entry points; each table says which:

| Build | Context | Notes |
|---|---|---|
| `m3` | 4K | first mixed 4/3-bit build, 8-token prefill only |
| `m3u2-4k` | 4K | + 32/64-token prefill entries, grouped LUT packing (fastest for short prompts, not released) |
| `m3k-16k` (**released**) | 16K | + 4K/8K/16K context tiers, blocked global attention, pre-transposed K |

## Decode

| | ANE + MTP | MLX + MTP | llama.cpp + MTP | Build |
|---|---|---|---|---|
| 16 short prompts × 128 tokens | **26.1 tok/s** | 19.9 | 19.7 | `m3` (`m3u2-4k`: 25.3) |
| 25 min sustained, 1024-token answers | **24.4 tok/s** | 17.6 | — | `m3` |
| 500–1000 token prompts, ≤512 tokens out | 20.4–21.3 tok/s | 19.9 | — | `m3k-16k` / `m3u2-4k` |
| ANE without drafter | 8.6 tok/s | — | — | `m3` |

Speed depends on how often the drafter is right, so it varies by task much more on the ANE (std 10.0 tok/s, range
12.5–43.4) than on the GPU (std 3.7):

| Task (16-prompt set) | ANE + MTP | GPU (MLX) + MTP |
|---|---|---|
| Code | **37.3** | 22.9 |
| Math | **33.5** | 22.5 |
| English prose | **22.1** | 18.9 |
| Chinese prose | 15.1 | 15.3 |

Draft acceptance: 3.49 tokens per round on average (first-token hit 83%, code 4.98).

## Prefill and end-to-end

Short and medium prompts (6 prompts, 500 and 1000 tokens, English/Chinese/code, up to 512 tokens out):

| | Prefill | Decode | Average total time |
|---|---|---|---|
| ANE `m3u2-4k` | 242–254 tok/s | 21.3 tok/s | **15.6–15.8 s** |
| ANE `m3k-16k` (released) | 220–235 tok/s | 20.4–20.7 tok/s | **16.4–16.6 s** |
| GPU MLX + MTP | 236–244 tok/s | 19.9 tok/s | 20.5 s |
| ANE `m3` (8-token prefill) | 66 tok/s | — | 23.7 s |

Long prompts (`m3k-16k`, a WikiText needle-in-a-haystack prompt with a passcode and a summary request,
`eval/prompts/long_prompts.json`). All passcodes recalled on both sides.

| Prompt | ANE prefill, first run | ANE prefill, back-to-back | GPU prefill | ANE decode + MTP | GPU decode + MTP | ANE decode, no drafter |
|---|---|---|---|---|---|---|
| 1K | **4.3 s** | 4.5 s | 4.4 s | **32.1 tok/s** | 17.8 | 8.2 |
| 4K | **16.8 s** | 17.1–17.8 s | 17.4 s | **21.4** | 14.3 | 7.1 |
| 8K | **34.2 s** | 35.3–38.1 s | 35.8 s | **20.5** | 13.8 | 7.0 |
| 15K | **69.6 s** | 73.5–75.1 s | 71.4 s | **20.1** | 11.2 | 6.8 |

Prefill is on par with the GPU (±5%): slightly faster on a machine that was idle, slightly slower right after other
runs. Decode stays well ahead at every length.

## Power and energy

Whole-machine power sampled every 0.5 s with `mactop` 0.2.7 (on M6 the per-unit CPU/ANE readings of `powermetrics`
and `mactop` are zero), integrated over the decode or prefill interval. Idle baseline 2.1–2.9 W.

Sustained decode (25 minutes, 1024-token answers, `m3` + MTP vs MLX + MTP):

| | Rounds | Speed | Machine power (GPU part) | Energy per token | Peak chip temp | Throttling |
|---|---|---|---|---|---|---|
| ANE + MTP | 35 | **24.4 tok/s** | **21.8 W** (0.8 W) | **0.92 J** | **70.3 °C** | none |
| GPU MLX + MTP | 26 | 17.6 tok/s | 39.2 W (17.3 W) | 2.24 J | 81.5 °C | none |

Short prompts: ANE + MTP 18.8 W / 0.77 J/tok, llama.cpp + MTP 34.7 W / 1.92 J/tok, MLX + MTP 39.4 W / 2.00 J/tok.

Prefill (no drafter; ANE `m3k-16k` for long prompts, `m3u2-4k` for short):

| Prompt | ANE time | ANE power | ANE per token | GPU time | GPU power (GPU part) | GPU per token |
|---|---|---|---|---|---|---|
| 1K | 4.5 s | 26.1 W | **108 mJ** | 4.5 s | 37.5 W (21.2) | 155 mJ |
| 4K | 17.1 s | 27.2 W | **114 mJ** | 17.4 s | 41.9 W (22.9) | 179 mJ |
| 8K | 38.1 s | 25.7 W | **121 mJ** | 35.5 s | 43.3 W (21.3) | 190 mJ |
| 15K | 75.1 s | 25.0 W | **125 mJ** | 71.4 s | 41.1 W (21.1) | 194 mJ |
| 6 × 500/1000 tokens | 19.2 s | 22.8 W | **98 mJ** | 19.5 s | 34.8 W | 151 mJ |

Minus idle power: 110 vs 179 mJ/token for long prompts, 88 vs 142 for short ones (38% less on the ANE). Short
prompts give only 4–8 samples each, so they are noisier.

## Quality

KL divergence of the next-token distribution against the fp32 PyTorch model over 64 chat positions (`ref64`,
`eval/validate.py`; the reference logits ship with the model under `eval/`):

| | KL (ref64) | top-1 agreement |
|---|---|---|
| `m3k-16k` (released) | 0.2202 | 59/64 |
| `m3u2-4k` | 0.2196 | 59/64 |
| GPU MLX QAT 4-bit | 0.172 | 61/64 |

The 16K build differs from the 4K one only by fp16 rounding in the blocked global attention. Prefill (64-token) and
step-by-step decode agree on hidden states (min cosine 0.9996 at 4K, 0.9953 at 16K).

160-item task pilot (64 IFEval, 32 HumanEval+, 64 CMMLU), measured on `m3` (same weights):

| | Bits/weight | IFEval strict (prompt) | IFEval strict (instruction) | HumanEval+ | CMMLU |
|---|---|---|---|---|---|
| Gemma 4 31B, MLX QAT (GPU) | ~4.5 | **85.9%** | **89.5%** | **93.8%** | **68.8%** |
| Gemma 4 31B, this repo (ANE) | ~3.8 | 82.8% | 88.6% | 90.6% | 67.2% |
| Qwen3.8-27B, ANEMLL Forge (ANE), reference | 3.17 | 81.3% | 88.6% | 81.3% | 65.6% |

The GPU model gets 4 more items right out of 160: close to it, slightly below, on a small sample.

## Memory

| | Wired after load | Min free while running with MTP |
|---|---|---|
| `m3k-16k` (released) | ~19.8 GB | 3.8–4.8 GB |
| `m3u2-4k` | ~22.9 GB | ~2.9 GB |

24 GB Macs do not fit (estimate from wired memory, not tested).

## Running next to a GPU job

GPU saturated by another process (fp16 4096×4096 matmul loop, 18.8 TFLOPS alone):

| | GPU idle | GPU saturated |
|---|---|---|
| ANE + MTP | 26.1 tok/s | 11.2 tok/s (−57%) |
| ANE alone (no GPU use) | 8.6 tok/s | 4.5 tok/s (−48%) |
| GPU matmul | 18.8 TFLOPS | 12.1–13.7 TFLOPS (−27 to −36%) |

Even the ANE-only run halves, most likely because both sides hit the shared memory bandwidth (not verified).

## Stalls

Some ANE calls take 300–900 ms instead of ~100 ms. They mostly hit the first call after the ANE has been idle
(≥0.3 s, worst after decode + 1 s idle), so a long prompt loses 0–2 s and a chat turn occasionally 0.3–0.6 s.
Ruled out: input data, memory pinning, switching entry points, the drafter, and 19 system services (Siri, Apple
Intelligence, media analysis, Spotlight knowledge; disabling them did not help). A background keep-alive call every
100 ms halves stalls in a synthetic test but slows MTP decode 2–4% in real use, so it is off
(`KEEPALIVE_MS=100` turns it on).
