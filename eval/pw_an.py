"""Integrate mactop headless samples over a bench's decode windows.
    pw_an.py <mactop.json> <bench.json> [idle_W]  -> avg total W, avg GPU W, J/token, tokens"""
import json, sys, re
from datetime import datetime
def samples(p):
    s = open(p).read(); out = []
    for m in re.finditer(r'\{"timestamp":"([^"]+)","soc_metrics":(\{[^}]*\})', s):
        t = datetime.fromisoformat(m.group(1)).timestamp(); d = json.loads(m.group(2)); out.append((t, d["total_power"], d["gpu_power"]))
    return out
S = samples(sys.argv[1]); B = json.load(open(sys.argv[2])); idle = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
E = Eg = T = N = 0.0; cats = {}
for r in B["res"]:
    w0, w1 = r["wall"]; ss = [x for x in S if w0 <= x[0] <= w1 + 0.5]
    if not ss: continue
    p = sum(x[1] for x in ss) / len(ss); g = sum(x[2] for x in ss) / len(ss); dt = w1 - w0
    n = r.get("tokens", len(r.get("out", [])) - 1)
    E += p * dt; Eg += g * dt; T += dt; N += n
    c = cats.setdefault(r["cat"], [0, 0, 0]); c[0] += p * dt; c[1] += dt; c[2] += n
print(f"avg total {E/T:.1f} W (above idle {E/T-idle:.1f}), avg GPU {Eg/T:.1f} W, {N/T:.2f} tok/s, {E/N:.2f} J/tok (above idle {(E-idle*T)/N:.2f}), tokens {int(N)}, {len(S)} samples")
print("  " + "  ".join(f"{k} {v[0]/v[1]:.1f}W {v[0]/v[2]:.2f}J/tok" for k, v in sorted(cats.items())))
