# Engineering notes

## Layout of the model

```
gemma-4-31B-it-ane-16k/
  chunk_L00-05.aimodel … chunk_L54-59.aimodel   10 chunks of 6 layers, every entry point fully on the ANE
  head.aimodel                          final norm + tied embedding head (262K vocab), 4-bit LUT
  embed_f16.npy                         token embeddings (fp16, looked up on the CPU)
  final_norm.npy                        final RMSNorm weight (for the MTP drafter input)
  manifest.json                         chunks, entry points per chunk, context tiers, tap layers
  tokenizer.json, chat_template.jinja, config.json, …
  eval/ids.npy, eval/logits.npy         fp32 reference for eval/validate.py
```

Each chunk is a multifunction Core AI model. Entry points share weights and differ in the number of tokens per call
(`T`) and the attention width (context tier):

| Chunks | 4K tier | 8K tier | 16K tier |
|---|---|---|---|
| layers 0–35 (grouped 4-bit LUT) | T = 8, 1, 32 | T = 32 | T = 8, 1, 32 |
| layers 36–59 (3-bit vector LUT) | T = 64, 8, 1, 32 | T = 64 | T = 64, 8 |

Prefill runs in 64-token steps (the front chunks run their 32-token entry twice). Decode uses T = 1 without the
drafter and T = 8 for the speculative verify. When a conversation passes 4096 tokens the runtime copies the global
KV history into the 16K buffers and switches tier; the 8K tier is used only for prefill steps that fit in it.

## The model rewrite (`gemma_ane.py`)

- Linear layers as 1×1 convolutions over `(1, C, 1, T)`, the layout the ANE prefers.
- RMSNorm in fp16 without overflow: scale the input down before squaring. Activations peak around 250 in the
  residual stream; 60 unquantized layers match fp32 with a minimum cosine of 0.9997.
- Sliding-window layers (50 of 60, window 1024) keep a ring buffer of 1024; the 10 global layers keep the whole
  context. KV history is an input of the model and the new K/V rows are outputs; the host writes them into the
  buffers (IOSurfaces bound once, so no copies between calls).
- Global attention over 16K in blocks of 2048 keys with the K history stored transposed `(nkv, hd, ctx)`: 15.6 ms →
  5.8 ms per call at T = 64. Gemma 4 has no 1/√d on global layers; scores are scaled before the max and rescaled after.
- Taps: chunks also output the hidden state at layers 1, 12, 23, 35, 46, 57 (used by DFlash-style drafters; the MTP
  drafter only needs the final hidden state).

## Runtime (`gemma_rt.py`)

- Core AI's Python binding allocated new output buffers on every call and never freed them (a crash after ~125
  steps), so the runtime uses the Swift bridge from ANEMLL Forge: buffers bound once and reused in place.
- Speculative decoding: the MLX drafter proposes 7 tokens (22–24 ms on the GPU), the ANE verifies 8 positions in one
  call (~130 ms vs ~114 ms for one token), the longest matching prefix is committed (KV rows for rejected positions
  are simply overwritten later).
- The GPU idles during the ANE verify and clocks down, which made the next draft 50 ms instead of 28 ms. The runtime
  starts a tiny GPU job about 95 ms into the verify (`PREWARM_MS=95`), and runs 30 warm-up draft tokens at load
  (`WARM=30`) so MLX compiles its shapes before the first prompt.
- Multi-turn chat reuses the KV cache of the common prefix (`reuse(ids)`). Gemma 4 emits four empty thinking tokens at
  the end of an answer that the chat template drops from the history, so prefix matching tolerates up to 8 tokens of
  difference. Prefix reuse turns a 1000-token follow-up into 0.25 s instead of 4.7 s.

## Pitfalls we hit

| Symptom | Cause | Fix |
|---|---|---|
| crash after ~125 decode steps | Core AI Python binding leaks output buffers | Swift bridge, buffers bound once |
| MTP drafter accepts only 1.33 tokens/round | drafter for the non-QAT model paired with the QAT target | use the QAT drafter: 3.49 tokens/round |
| 3/4/6-bit layers use the same wired memory (0.40 GB/layer) | ANE compiler expands a per-tensor 4-bit LUT to ~6.5 bits | repeat the table per 128 output channels (`LUT_GROUP=128`): 0.30 GB/layer |
| an extra entry point makes a chunk 100+ ms per call and +5 GB | a chunk keeps at most 7 entry points on the ANE; the 8th falls off | ≤ 7 entries per chunk, most important first in `TS_ORDER` |
| 48-token entry is 260 ms | T = 48 does not compile for the ANE (runs on the GPU) | use 32 or 64 |
| `nilError` at load after a build | low disk: macOS evicted fresh Core AI compile cache entries | keep 30+ GB free, `fix_cache.py <model>` evicts broken entries, `cache_orphans.py` lists leftovers |
| 300–900 ms ANE calls now and then | ANE waking up after ≥ 0.3 s idle (not data, not system services) | none worth it; keep-alive exists (`KEEPALIVE_MS=100`) but costs 2–4% decode |
| machine near out of memory during long runs | `VTDecoderXPCService` (dynamic wallpaper) leaking up to 20 GB; tokenizer loaded after the model | static wallpaper, tokenize before loading, run heavy jobs under `scripts/memguard.sh` |
| decode halves when another app uses the GPU heavily | shared memory bandwidth (likely) | none; avoid heavy GPU jobs alongside |
| Core AI cache grows by 20–30 GB per build, recompiles after macOS updates | per-build compile cache | clean orphan entries after rebuilding |

## Measuring

- `ane_run.py` prints per-prompt prefill time, decode speed, draft acceptance, and min free memory; `--out` saves JSON.
- `eval/gpu_bench.py` + `eval/srv_run.py`: the same prompts against `mlx_vlm.server` for the GPU baseline.
- `eval/pw_an.py`, `eval/pw_pf.py`: integrate `mactop --format json` power samples over decode / prefill intervals.
- `eval/validate.py`: KL and top-1 vs the fp32 reference; `eval/val_pf.py`: prefill path vs step-by-step path.
- `eval/make_long.py`, `eval/make_e2e.py`: build the long-context and end-to-end prompt sets.
