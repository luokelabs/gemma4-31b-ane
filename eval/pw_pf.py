"""Prefill energy per prompt from mactop samples.  pw_pf.py <mactop.json> <bench.json> <idle_W> [ane|gpu]
ane: window = [wall0, dec_wall0] (ane_run.py json);  gpu: window = [wall0 - ttft, wall0] (gpu_bench.py json)."""
import json, sys, re
from datetime import datetime
S = []
for m in re.finditer(r'\{"timestamp":"([^"]+)","soc_metrics":(\{[^}]*\})', open(sys.argv[1]).read()):
    d = json.loads(m.group(2)); S.append((datetime.fromisoformat(m.group(1)).timestamp(), d["total_power"], d["gpu_power"]))
B = json.load(open(sys.argv[2]))["res"]; idle = float(sys.argv[3]); kind = sys.argv[4]
NT = {r["cat"]: r["prompt_tokens"] for r in json.load(open(sys.argv[5]))["res"]} if len(sys.argv) > 5 else {}
E = T = N = 0
for r in B:
    if kind == "ane": a, b = r["wall"][0], r["dec_wall"][0]; n = r["prompt_tokens"]
    else: b = r["wall"][0]; a = b - r["ttft"]; n = NT.get(r["cat"], 0)
    ss = [x for x in S if a + 0.25 <= x[0] <= b]
    if len(ss) < 2: print(f"  {r['cat']}: window {b-a:.1f}s too short"); continue
    p = sum(x[1] for x in ss) / len(ss); g = sum(x[2] for x in ss) / len(ss); dt = b - a
    E += p * dt; T += dt; N += n
    print(f"  {r['cat']:8s} {dt:6.1f}s  {p:5.1f} W (GPU {g:4.1f})  " + (f"{n} tok {p*dt/n*1e3:6.1f} mJ/tok (above idle {(p-idle)*dt/n*1e3:6.1f})" if n else ""))
if N: print(f"TOTAL {T:.1f}s avg {E/T:.1f} W, {E/N*1e3:.1f} mJ/tok (above idle {(E-idle*T)/N*1e3:.1f}), {N/T:.0f} tok/s")
