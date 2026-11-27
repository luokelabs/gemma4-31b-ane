---
license: apache-2.0
base_model: google/gemma-4-31B-it-qat-q4_0-unquantized
pipeline_tag: text-generation
language:
- en
- zh
tags:
- gemma4
- apple-neural-engine
- ane
- core-ai
- macos
- quantized
- speculative-decoding
- research
---

# Gemma 4 31B IT for the Apple Neural Engine (16K context), research release

> Research release from Luoke Labs: an experiment in running a 31B model on the ANE, not a model used in our products.
> No support or update guarantees.

All of Gemma 4 31B (dense, instruction-tuned) compiled for the **Apple Neural Engine** through **Core AI** (macOS 27),
quantized to ~3.8 bits per weight. Runs on a 32 GB Apple silicon Mac with the code at
**[github.com/luokelabs/gemma4-31b-ane](https://github.com/luokelabs/gemma4-31b-ane)**, which pairs it with Google's MTP drafter on the GPU for
speculative decoding.

On a Mac mini (M6, 32 GB), greedy decoding, compared with the same model on the GPU (MLX 4-bit QAT + MTP):

| | ANE (this model) | GPU (MLX) |
|---|---|---|
| Decode with MTP, short prompts | **26.1 tok/s** | 19.9 |
| Decode with MTP at 1K / 8K / 15K context | **32.1 / 20.5 / 20.1** | 17.8 / 13.8 / 11.2 |
| Prefill, 500–1000 / 15K tokens | 220–235 tok/s / 69.6–75.1 s | 236–244 tok/s / 71.4 s |
| Whole-machine energy per generated token | **0.92 J** | 2.24 J |
| KL vs fp32 (ref64) / top-1 | 0.220 / 59 of 64 | 0.172 / 61 of 64 |

Details and caveats: [results](https://github.com/luokelabs/gemma4-31b-ane/blob/main/docs/results.md).

## Use

```bash
git clone https://github.com/luokelabs/gemma4-31b-ane && cd gemma4-31b-ane
scripts/setup.sh && scripts/download.sh
export MTP_MLX=$PWD/models/gemma-4-31B-it-qat-assistant-4bit-mlx
.venv/bin/python chat.py models/gemma-4-31B-it-ane-16k
```

Requires a 32 GB Apple silicon Mac, macOS 27 with Core AI, Xcode command line tools, Python 3.13
(`brew install python@3.13`, or have [uv](https://docs.astral.sh/uv/) installed), and 30+ GB of free disk for the compile cache.
The first load compiles the model for the ANE (several minutes, once per macOS build).
These are Core AI `.aimodel` packages; they do not load in transformers, MLX, or llama.cpp.

## Files

| File | |
|---|---|
| `chunk_L00-05.aimodel` … `chunk_L54-59.aimodel` | 10 multifunction chunks of 6 layers (entry points for 1/8/32/64 tokens and 4K/8K/16K context tiers) |
| `head.aimodel` | final norm + output head |
| `embed_f16.npy`, `final_norm.npy` | token embeddings (looked up on the CPU), final norm weight |
| `manifest.json` | chunk list, entry points, tiers |
| `tokenizer.json`, `chat_template.jinja`, `config.json`, … | from the base model |
| `eval/ids.npy`, `eval/logits.npy` | fp32 reference for the KL check |

## Quantization

- Layers 0–35: 4-bit LUT (16-entry table per tensor, repeated per 128 output channels, fp16 scale per channel).
- Layers 36–59: 3-bit vector LUT (pairs of weights, 64-entry codebook) on block-Hadamard-rotated weights.
- GPTQ, then per-layer distillation, then whole-model distillation against the fp32 logits, using WikiText-2 and
  self-generated chat text. Recipe: [quantization.md](https://github.com/luokelabs/gemma4-31b-ane/blob/main/docs/quantization.md).

160-item pilot (IFEval / HumanEval+ / CMMLU): 82.8% / 90.6% / 67.2%, vs 85.9% / 93.8% / 68.8% for the 4.5-bit GPU model.

## Limitations

32 GB Macs only (3–5 GB of memory left while running); text only (the vision and audio towers are not included);
greedy decoding only; decode roughly halves while another app saturates the GPU; occasional 300–900 ms ANE stalls
after idle. Measured on a single machine.

## License

Apache-2.0, derived from [google/gemma-4-31B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized).
Use is also subject to the [Gemma 4 terms](https://ai.google.dev/gemma/docs/gemma_4_license).
The runtime builds on [ANEMLL Forge](https://github.com/Anemll/anemll-forge) (MIT).
