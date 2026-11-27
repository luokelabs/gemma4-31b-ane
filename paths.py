"""Where things live. Every path can be overridden with an environment variable.

GEMMA_HF        original Hugging Face checkpoint (google/gemma-4-31B-it-qat-q4_0-unquantized): weights for
                conversion / quantization, tokenizer when the ANE model directory has none
FORGE_DIR       ANEMLL Forge checkout (Swift bridge to Core AI + LUT helpers), see scripts/setup.sh
MTP_MLX         MTP drafter for the GPU (mlx-community/gemma-4-31B-it-qat-assistant-4bit)
GEMMA_WORK      scratch space for quantized weights, calibration caches, logs
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _p(env, default):
    return Path(os.path.expanduser(os.environ.get(env, str(default))))


HF_MODEL = _p("GEMMA_HF", "~/models/gemma-4-31B-it-qat-q4_0-unquantized")
FORGE = _p("FORGE_DIR", ROOT / "third_party/anemll-forge")
BRIDGE = FORGE / "coreai/swift_bridge"
FORGE_SCRIPTS = FORGE / "scripts"
MTP_MLX = _p("MTP_MLX", "~/models/gemma-4-31B-it-qat-assistant-4bit-mlx")
WORK = _p("GEMMA_WORK", ROOT / "work")
DATA = ROOT / "data"
REF64 = _p("REF64_DIR", WORK / "ref64")          # ids.npy + logits.npy (fp32 reference, quant/hf_ref.py N=64)
CALIB = _p("CALIB_DIR", WORK / "calib")          # wikitext parquet + chat_ids.json (scripts/get_calib.sh)


def coreai_cache_dirs():
    """Core AI's compiled-program cache, one folder per macOS build (e.g. 26A434/python)."""
    base = Path.home() / "Library/Caches/coreai-cache"
    return sorted(base.glob("*/python")) if base.exists() else []
