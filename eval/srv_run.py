"""Same measurements as ane_run.py against an OpenAI-compatible server (mlx_vlm.server or llama-server), streaming.
ttft ~ prefill, decode tok/s = (n-1)/(t_last - t_first), e2e = request sent -> last token.
    URL=http://127.0.0.1:8770 MODEL=<name> TOK=<hf tokenizer dir> srv_run.py <prompts.json> [--max 128] [--loop S] [--out f]"""
import argparse, json, os, time, urllib.request
from transformers import AutoTokenizer
URL, MODEL = os.environ["URL"], os.environ["MODEL"]
tok = AutoTokenizer.from_pretrained(os.environ.get("TOK", MODEL))


def chat(prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); ts = []; text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            c = d["choices"][0]["delta"].get("content") if d.get("choices") else None
            if c: ts.append(time.time()); text.append(c)
    return t0, ts, "".join(text), time.time()


ap = argparse.ArgumentParser(); ap.add_argument("prompts"); ap.add_argument("--max", type=int, default=128)
ap.add_argument("--loop", type=float, default=0); ap.add_argument("--out", default="logs/srv_run.json"); a = ap.parse_args()
prompts = json.load(open(a.prompts)); chat(prompts[0][1], 32)
res = []; T0 = time.time(); rnd = 0
while True:
    for cat, prompt, *nn in prompts:
        t0, ts, text, tend = chat(prompt, nn[0] if nn else a.max)
        n = len(tok.encode(text, add_special_tokens=False)); dt = ts[-1] - ts[0] if len(ts) > 1 else float("nan")
        pt = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
        pt = len(pt["input_ids"] if hasattr(pt, "keys") else pt)
        r = {"cat": cat, "round": rnd, "prompt": prompt[:80], "prompt_tokens": pt, "prefill_s": ts[0] - t0, "prefill_tok_s": pt / (ts[0] - t0),
             "tokens": n, "decode_s": dt, "tok_s": (n - 1) / dt, "e2e_s": tend - t0, "wall": [t0, tend], "dec_wall": [ts[0], ts[-1]],
             "t_rel": t0 - T0, "text": text}
        res.append(r)
        print(f"[{rnd}] {cat:6s} prompt {pt:5d} ttft {r['prefill_s']:5.1f}s ({r['prefill_tok_s']:.0f}/s) | {n:4d} tok {r['tok_s']:5.1f} tok/s "
              f"| e2e {r['e2e_s']:5.1f}s | t+{r['t_rel']:.0f}s", flush=True)
        json.dump({"res": res}, open(a.out, "w"), ensure_ascii=False)
        if a.loop and time.time() - T0 > a.loop: break
    rnd += 1
    if not a.loop or time.time() - T0 > a.loop: break
tot = sum(r["tokens"] - 1 for r in res) / sum(r["decode_s"] for r in res)
print(f"ALL decode {tot:.2f} tok/s over {len(res)} gens", flush=True)
