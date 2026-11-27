"""prefill64 path vs step path on the same 64 ref tokens: hidden-state agreement (last_hidden of all 64 positions)."""
import asyncio, sys, numpy as np
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parent.parent))   # repo root
from paths import REF64
from gemma_rt import GemmaANE
async def main(d):
    m = GemmaANE(d); await m.load(verbose=False); ids = [int(x) for x in np.load(REF64 / "ids.npy").ravel()[:64]]
    m.reset(); hs = []
    for a in range(0, 64, 8): await m.step(ids[a:a + 8], want_logits=False); hs.append(m.last_hidden.copy())
    hs = np.concatenate(hs)
    m.reset(); m.prefill64(ids); hp = m.last_hidden.copy()
    cos = (hs * hp).sum(1) / np.linalg.norm(hs, axis=1) / np.linalg.norm(hp, axis=1)
    rel = np.linalg.norm(hs - hp, axis=1) / np.linalg.norm(hs, axis=1)
    print(f"prefill64 vs v8 steps: cos min {cos.min():.5f} mean {cos.mean():.5f}; rel err mean {rel.mean():.4f} max {rel.max():.4f}")
asyncio.run(main(sys.argv[1]))
