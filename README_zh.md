# Gemma 4 31B 跑在 Apple 神经网络引擎上

把 **Gemma 4 31B（稠密）** 完整放到 32 GB Mac 的 **ANE** 上，通过 Core AI 运行，并用 Google 官方 MTP 草稿模型在 GPU 上做推测解码。
在 Mac mini（M6，32 GB）上：生成比 GPU（MLX、llama.cpp）快，预填和 GPU 持平，每个 token 耗电约为 GPU 的一半以下。

[English](README.md) · 模型权重：Hugging Face（即将上传） · 工程笔记：[docs/](docs/)

| Mac mini M6 32 GB，贪心解码 | ANE（本项目） | GPU，MLX 4 位 | GPU，llama.cpp Q4_0 |
|---|---|---|---|
| 带 MTP 草稿解码，16 条短提示 | **26.1 tok/s** | 19.9 | 19.7 |
| 持续 25 分钟解码（每条 1024 token） | **24.4 tok/s** | 17.6 | — |
| 每生成一个 token 整机耗电（25 分钟） | **0.92 J**（21.8 W） | 2.24 J（39.2 W） | — |
| 预填，500–1000 token 提示 | 220–250 tok/s | 236–244 | ~169 |
| 预填，15K token 提示 | 69.6–75.1 s | 71.4 s | — |
| 预填每个提示 token 耗电（1K–15K） | **108–125 mJ** | 155–194 mJ | — |
| 1K / 4K / 8K / 15K 上下文带 MTP 解码 | **32.1 / 21.4 / 20.5 / 20.1** | 17.8 / 14.3 / 13.8 / 11.2 | — |
| 芯片最高温度（25 分钟） | **70 °C** | 81.5 °C | — |

测法、版本和注意事项见 [docs/results.md](docs/results.md)（英文）。

## 仓库内容

- `gemma_ane.py`：为 ANE 改写的 Gemma 4 31B（1×1 卷积、fp16 安全的归一化、KV 历史作为输入），导出 Core AI `.aimodel` 分块，LUT / 向量 LUT 权重格式，上下文分档。
- `gemma_rt.py`：基于 [ANEMLL Forge](https://github.com/Anemll/anemll-forge) Swift 桥接的运行时：IOSurface 缓冲一次绑定、KV 管理、64 token 预填、分档切换、推测解码校验与提交。
- `mtp_mlx.py`：Google MTP 草稿（`gemma-4-31B-it-qat-assistant`，MLX 4 位）跑在 GPU 上。
- `chat.py`：交互对话。`ane_run.py`：测速（预填、解码、端到端、内存）。
- `quant/`：量化流程（GPTQ → 逐层蒸馏 → 整模型蒸馏）。
- `eval/`：与 fp32 对比的 KL 检查、提示集、GPU 对照、功耗积分。

## 环境要求

- **32 GB** 内存的 Apple 芯片 Mac（ANE 常驻约 20–23 GB；24 GB 机型装不下）。
- macOS 27（带 Core AI）、Xcode 命令行工具（`swiftc`）、Python 3.13。
- 模型约 16 GB，加上 Core AI 编译缓存 20–30 GB。请保持 30 GB 以上空闲：磁盘不足时系统会清掉刚编译的缓存，加载时报 `nilError`（运行 `fix_cache.py <模型目录>`）。

## 快速开始

```bash
git clone https://github.com/luokelabs/gemma4-31b-ane && cd gemma4-31b-ane
scripts/setup.sh                 # 建 venv、拉取指定版本的 ANEMLL Forge、编译 Swift 桥接
scripts/download.sh              # 下载 ANE 模型（约 16 GB）和 MTP 草稿（约 0.3 GB）到 ./models
export MTP_MLX=$PWD/models/gemma-4-31B-it-qat-assistant-4bit-mlx
.venv/bin/python chat.py models/gemma-4-31B-it-ane-16k
```

第一次加载要为 ANE 编译（几分钟，每个 macOS 版本一次），之后几秒加载完。

测速（带 MTP 时设 `PREWARM_MS=95 MLX_CACHE=268435456 WARM=30 ND_ZH=3`，与上表一致）：

```bash
.venv/bin/python ane_run.py models/gemma-4-31B-it-ane-16k eval/prompts/e2e_prompts.json --max 512 --out logs/e2e.json
.venv/bin/python ane_run.py models/gemma-4-31B-it-ane-16k eval/prompts/long_prompts.json --max 128 --out logs/long.json
REF64_DIR=models/gemma-4-31B-it-ane-16k/eval .venv/bin/python eval/validate.py models/gemma-4-31B-it-ane-16k
```

## 原理

- **整个模型在 ANE 上。** 60 层分成 10 块（每块 6 层）加输出层，每块全部放在 ANE 上，没有悄悄回退到 GPU/CPU。fp16 激活不溢出（逐层与 fp32 余弦最低 0.9997）。
- **混合低位宽，约 3.8 位/权重。** 0–35 层：4 位查表（每张量一张表 + 每通道缩放）；36–59 层：在 Hadamard 旋转后的权重上做 3 位向量查表（两个权重一组，64 项码本）。先 GPTQ，再逐层蒸馏和整模型蒸馏。
- **ANE 与 GPU 分工的推测解码。** ANE 一次校验 8 个 token 和校验 1 个差不多贵（瓶颈是读权重），所以 GPU 草稿先猜 7 个（约 23 ms），ANE 一次校验。草稿必须和 QAT 主模型配对：通用草稿每轮只接受 1.33 个，QAT 草稿 3.49 个。
- **快速预填。** 64 token 一步的预填入口，全局注意力按 2048 个 key 分块、K 历史预先转置，并按 4K / 8K / 16K 分档，短提示不必为 16K 宽度付出代价。
- **每块最多 7 个入口留在 ANE 上**，第 8 个会悄悄掉出 ANE（每次 100 多 ms、多占约 5 GB）。

详见 [docs/engineering_notes.md](docs/engineering_notes.md)。

## 质量

| | KL（ref64，对 fp32） | 首选一致 | IFEval（严格，按题） | HumanEval+ | CMMLU |
|---|---|---|---|---|---|
| 本模型（16K） | 0.220 | 59/64 | 82.8%* | 90.6%* | 67.2%* |
| Gemma 4 31B QAT，MLX 4 位（GPU） | 0.172 | 61/64 | 85.9% | 93.8% | 68.8% |

\* 160 题小样本（IFEval 64、HumanEval+ 32、CMMLU 64），在权重相同的 4K 版上测；16K 版与它只差 fp16 舍入（KL 0.2202 对 0.2196）。接近、略低于 4.5 位的 GPU 版。

## 限制

- 只支持 32 GB Mac，余量小：带草稿运行时剩 3–5 GB 内存。
- GPU 被别的任务跑满时，ANE 解码约降一半（共享内存带宽）。
- 偶发 300–900 ms 的 ANE 调用卡顿（系统层面，长提示多 0–2 秒）。
- 只支持贪心解码（推测校验对贪心是精确的）。
- 只在一台机器上测过（Mac mini M6 32 GB，macOS 27.0.1），其他芯片的 ANE/GPU 比例不同。

## 复现权重

[docs/quantization.md](docs/quantization.md) 列出完整流程和命令，在 M6 mini 上约需 1–2 天；`scripts/build_16k.sh` 把量化权重构建成发布的模型。

## 许可与致谢

代码：Apache-2.0。模型权重源自 [google/gemma-4-31B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized)（Apache-2.0，[Gemma 4 条款](https://ai.google.dev/gemma/docs/gemma_4_license)）。Core AI Swift 桥接和 LUT 工具来自 [ANEMLL Forge](https://github.com/Anemll/anemll-forge)（MIT），由 `scripts/setup.sh` 拉取。MTP 草稿为 [mlx-community/gemma-4-31B-it-qat-assistant-4bit](https://huggingface.co/mlx-community/gemma-4-31B-it-qat-assistant-4bit)，用 [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)（MIT）运行。见 [NOTICE](NOTICE)。
