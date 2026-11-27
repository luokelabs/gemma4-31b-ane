"""Gemma 31B on the ANE, end-to-end per prompt: prefill (largest entry point per step) + decode with the MTP drafter (GPU, MLX) or no drafter.
    ane_run.py <model_dir> <prompts.json> [--draft mtp|none] [--nd 7] [--max 128] [--loop SECONDS] [--out logs/x.json]
Each row: prompt_tokens, prefill_s, decode tok/s, e2e_s (prefill + decode, tokenizer excluded), wall window, memory snapshot.
--loop: repeat the prompt list until SECONDS have passed (sustained run).
Env (MTP runs; README numbers use PREWARM_MS=95 MLX_CACHE=268435456 WARM=30 ND_ZH=3):
  PREWARM_MS  wake the GPU drafter this many ms after starting the ANE verify (the GPU clocks down while idle)
  MLX_CACHE   MLX buffer cache limit in bytes;  WARM  drafter warm-up steps;  ND_ZH  drafts per round for "zh" prompts
  PFDBG=1     print per-64-token-block prefill times;  KEEPALIVE_MS  see gemma_rt.keepalive (off by default)"""
import argparse, asyncio, json, os, subprocess, threading, time, resource
import numpy as np
from gemma_rt import GemmaANE
from gemma_ane import wired_gb
from pathlib import Path
import sys


def tok_cli(*a):
    """Tokenizer in a child process (keeps transformers, ~1 GB, out of the ANE process)."""
    return subprocess.run([sys.executable, str(Path(__file__).with_name("tok_cli.py")), *a], capture_output=True, text=True, check=True).stdout
PW = float(os.environ.get("PREWARM_MS", 0))


def mem():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    return {"wired_gb": wired_gb(), "maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, "swap": s}


async def run_one(m, dr, ids, eos, nmax, nd, start=None):
    """start=None: fresh KV. start=L: the caller already holds ids[:L] in the KV (prefix reuse / restored snapshot)."""
    if os.environ.get("RANDIDS"): ids = [ids[0]] + list(np.random.RandomState(len(ids)).randint(1000, 50000, len(ids) - 1))
    if start is None: m.reset(); start = 0
    t0 = time.perf_counter(); w0 = time.time()
    B = max(m.full_Ts)   # prefill block: the largest entry point every chunk has (v32 builds), else 8
    a = start; _bt = []
    while len(ids) - a > 64:   # mixed build: 64-token blocks while more than one block remains (per context tier)
        m._ensure(m.pos + 64)
        if m.split:
            _t = time.perf_counter(); _a0 = _ane[0] if os.environ.get("PFDBG") else 0; m.prefill64(ids[a:a + 64]); a += 64
            if os.environ.get("PFDBG"): _bt.append((round((time.perf_counter() - _t) * 1e3), round((_ane[0] - _a0) * 1e3)))
        else: await m.step(ids[a:a + B], want_logits=False); a += B
    t_blk = time.perf_counter() - t0
    for a in range(a, len(ids), B):
        lg = await m.step(ids[a:a + B], want_logits=a + B >= len(ids))
    tp = time.perf_counter() - t0
    if os.environ.get("PFDBG"): print(f"  prefill blocks {t_blk:.2f}s tail {tp - t_blk:.2f}s blocks ms {_bt}", flush=True)
    tok = int(lg[-1].argmax()); out = [tok]; acc = []; t_start = time.perf_counter()
    h = m.final_norm(m.last_hidden[-1]) if dr else None
    while len(out) < nmax and tok not in eos:
        if dr is None:
            lg = await m.step([tok]); tok = int(lg[-1].argmax()); out.append(tok); continue
        dft = dr.draft(tok, h, m.pos, m.shared_kv(), nd); t1 = time.perf_counter()
        if PW:
            box = {}; th = threading.Thread(target=lambda: box.update(lg=asyncio.run(m.step([tok] + dft, commit=False)))); th.start()
            time.sleep(max(0.0, PW / 1e3 - (time.perf_counter() - t1))); dr.warm(); th.join(); lg = box["lg"]
        else:
            lg = await m.step([tok] + dft, commit=False)
        g = lg.argmax(-1); a = 0
        while a < nd and dft[a] == int(g[a]) and dft[a] not in eos: a += 1
        m.commit_rows(a + 1); acc.append(a)
        for t in dft[:a] + [int(g[a])]:
            out.append(t)
            if t in eos or len(out) >= nmax: break
        tok = out[-1]; h = m.final_norm(m.last_hidden[a])
    td = time.perf_counter() - t_start
    return out, tp, td, acc, w0


async def main(a):
    if (Path(a.model_dir) / "tokenizer.json").exists(): os.environ.setdefault("GEMMA_TOKENIZER", a.model_dir)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    prompts = json.load(open(a.prompts))
    enc = {}
    def E(p):
        if p not in enc:
            e = json.loads(tok_cli("enc", p).strip().splitlines()[-1]); enc[p] = (e["ids"], set(e["eos"]))
        return enc[p]
    for _, p, *_r in prompts: E(p)   # tokenize everything before the model is wired (tok_cli spawns a transformers process)
    w0 = wired_gb()
    m = GemmaANE(a.model_dir); await m.load(verbose=False); w1 = wired_gb()
    if os.environ.get("PFDBG"):   # per-block ANE time (plan runs) next to the block wall time
        global _ane; _ane = [0.0]
        for st in m.S.values():
            if st["split"] is None: continue
            for pl in (st["plan_front"], st["plan"][64][False]):
                def w(*x, _o=pl.run):
                    t = time.perf_counter(); r = _o(*x); _ane[0] += time.perf_counter() - t; return r
                pl.run = w
    dr = None
    if a.draft == "mtp":
        from mtp_mlx import MTPMLX; dr = MTPMLX(m.emb)
    print(f"ctx {m.ctx}; ANE wired +{w1 - w0:.1f} GB; draft {a.draft}", flush=True)
    ids, eos = E(prompts[0][1]); await run_one(m, dr, ids, eos, 32, a.nd)  # warm-up
    res = []; T0 = time.time(); rnd = 0
    while True:
        for cat, prompt, *nn in prompts:
            nmax = nn[0] if nn else a.max; nd = int(os.environ.get("ND_ZH", a.nd)) if cat == "zh" else a.nd
            ids, eos = E(prompt)
            out, tp, td, acc, w = await run_one(m, dr, ids, eos, nmax, nd)
            n = len(out) - 1
            r = {"cat": cat, "round": rnd, "prompt": prompt[:80], "prompt_tokens": len(ids), "prefill_s": tp, "prefill_tok_s": len(ids) / tp,
                 "tokens": n, "decode_s": td, "tok_s": n / td if td > 0 else float("nan"), "e2e_s": tp + td,
                 "tok_per_cycle": n / max(1, len(acc)) if acc else 1.0, "wall": [w, w + tp + td], "dec_wall": [w + tp, w + tp + td],
                 "t_rel": w - T0, "mem": mem(), "out": out}
            res.append(r)
            print(f"[{rnd}] {cat:6s} prompt {len(ids):5d} prefill {tp:5.1f}s ({r['prefill_tok_s']:.0f}/s) | {n:4d} tok {r['tok_s']:5.1f} tok/s "
                  f"| e2e {r['e2e_s']:5.1f}s | t+{r['t_rel']:.0f}s wired {r['mem']['wired_gb']:.1f}", flush=True)
            json.dump({"res": res, "wired_load_gb": w1 - w0}, open(a.out, "w"), ensure_ascii=False)
            if a.loop and time.time() - T0 > a.loop: break
        rnd += 1
        if not a.loop or time.time() - T0 > a.loop: break
    dec = [r for r in res]
    tot = sum(r["tokens"] for r in dec) / sum(r["decode_s"] for r in dec)
    print(f"ALL decode {tot:.2f} tok/s over {len(dec)} gens; mean e2e {np.mean([r['e2e_s'] for r in dec]):.1f}s; wired now +{wired_gb() - w0:.1f} GB", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("prompts")
    ap.add_argument("--draft", default="mtp"); ap.add_argument("--nd", type=int, default=7); ap.add_argument("--max", type=int, default=128)
    ap.add_argument("--loop", type=float, default=0); ap.add_argument("--out", default="logs/ane_run.json")
    asyncio.run(main(ap.parse_args()))
