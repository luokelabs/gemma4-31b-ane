"""GPU (mlx_vlm server) decode speed on the same 16 prompts as spec_df.py (128 new tokens, greedy, thinking off).
decode tok/s = (n-1)/(t_last - t_first) per prompt; ALL = total tokens / total decode time (same aggregation as spec_df).
    URL=http://127.0.0.1:8770 MODEL=<path> OUT=logs/gpu_x.json python gpu_bench.py spec_prompts.json"""
import json, os, sys, time, urllib.request
from pathlib import Path

URL, MODEL = os.environ["URL"], os.environ["MODEL"]


def chat(prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); times = []; text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            c = d["choices"][0]["delta"].get("content") if d.get("choices") else None
            if c: times.append(time.time()); text.append(c)
    return t0, times, "".join(text)


from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
prompts = json.load(open(sys.argv[1])); res = []
chat(prompts[0][1], 32)  # warm-up, not timed
for cat, prompt, *nn in prompts:
    t0, ts, text = chat(prompt, nn[0] if nn else 128)
    n = len(tok.encode(text, add_special_tokens=False))
    dt = ts[-1] - ts[0] if len(ts) > 1 else float("nan")
    row = {"cat": cat, "prompt": prompt, "tokens": n, "wall": [ts[0], ts[-1]], "ttft": ts[0] - t0, "decode_s": dt, "tok_s": (n - 1) / dt, "text": text}
    res.append(row)
    print(f"{cat:5s} {n:4d} tok {row['tok_s']:5.1f} tok/s ttft {row['ttft']:.2f}s | {prompt[:36]}", flush=True)
tot = sum(r["tokens"] - 1 for r in res) / sum(r["decode_s"] for r in res)
cats = sorted({r["cat"] for r in res})
summ = {c: sum(r["tokens"] - 1 for r in res if r["cat"] == c) / sum(r["decode_s"] for r in res if r["cat"] == c) for c in cats}
print("ALL %.2f tok/s; " % tot + " ".join(f"{c} {v:.1f}" for c, v in summ.items()), flush=True)
json.dump({"res": res, "summary": summ, "all": tot}, open(os.environ.get("OUT", "gpu_bench.json"), "w"), ensure_ascii=False)
