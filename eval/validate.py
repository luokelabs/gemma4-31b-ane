"""Teacher-forced check of the ANE runtime against the transformers fp32 reference (ref64/): blocks of 8 through
the KV cache, then the last 8 tokens one at a time (v1 entry)."""
import asyncio, sys, time
from pathlib import Path
import numpy as np
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parent.parent))   # repo root
from paths import REF64
from gemma_rt import GemmaANE

async def main(d, ref="ref64"):
    R = REF64 if ref == "ref64" else Path(ref)
    ids, ref_l = np.load(R / "ids.npy"), np.load(R / "logits.npy")
    m = GemmaANE(d); t0 = time.time(); await m.load(); print(f"load {time.time() - t0:.0f}s")
    for mode in ("v8", "v8+v1"):
        m.reset(); out, ts = [], []
        n8 = len(ids) if mode == "v8" else len(ids) - 8
        for a in range(0, n8, 8):
            t1 = time.perf_counter(); out.append(await m.step(ids[a:a + 8])); ts.append(time.perf_counter() - t1)
        t8 = np.median(ts) * 1e3; ts = []
        for a in range(n8, len(ids)):
            t1 = time.perf_counter(); out.append(await m.step(ids[a:a + 1])); ts.append(time.perf_counter() - t1)
        lg = np.concatenate(out); lg = np.tanh(lg / 30.0) * 30.0
        def lsm(z): z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))
        lp, lq = lsm(ref_l.astype(np.float64)), lsm(lg.astype(np.float64))
        kl = (np.exp(lp) * (lp - lq)).sum(-1)
        agree = (lg.argmax(-1) == ref_l.argmax(-1))
        print(f"[{mode}] top-1 agreement {agree.mean():.3f} ({agree.sum()}/{len(ids)}); KL mean {kl.mean():.4f} median {np.median(kl):.4f} max {kl.max():.4f}; "
              f"finite {np.isfinite(lg).all()}; step v8 {t8:.0f} ms" + (f", v1 {np.median(ts) * 1e3:.0f} ms" if ts else ""), flush=True)

asyncio.run(main(*sys.argv[1:]))
