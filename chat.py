"""Interactive chat with Gemma 4 31B on the ANE (MTP drafter on the GPU by default).
    python chat.py <model_dir> [--draft mtp|none] [--max 1024]
Multi-turn: each turn reuses the KV of the conversation so far, only the new tokens are prefilled.
Commands: /reset clears the conversation, /quit exits. Greedy decoding."""
import argparse, asyncio, json, os, subprocess, sys, time
from pathlib import Path

# measured defaults (README): GPU prewarm before the ANE verify returns, MLX buffer cache cap, drafter warm-up steps
for k, v in {"PREWARM_MS": "95", "MLX_CACHE": "268435456", "WARM": "30"}.items(): os.environ.setdefault(k, v)
from ane_run import run_one
from gemma_rt import GemmaANE


class Tok:
    """Tokenizer in a long-lived child process (tok_cli.py serve)."""
    def __init__(self):
        self.p = subprocess.Popen([sys.executable, str(Path(__file__).with_name("tok_cli.py")), "serve"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def __call__(self, op, arg):
        self.p.stdin.write(json.dumps({"op": op, "arg": arg}, ensure_ascii=False) + "\n"); self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())


async def main(a):
    if (Path(a.model_dir) / "tokenizer.json").exists(): os.environ.setdefault("GEMMA_TOKENIZER", a.model_dir)
    tok = Tok(); tok("encm", [{"role": "user", "content": "hi"}])   # start the tokenizer before the model is wired
    m = GemmaANE(a.model_dir); t0 = time.time(); await m.load(verbose=False)
    dr = None
    if a.draft == "mtp":
        from mtp_mlx import MTPMLX; dr = MTPMLX(m.emb)
    print(f"loaded in {time.time() - t0:.0f}s (context {m.ctx} tokens, drafter {a.draft}). /reset, /quit", flush=True)
    msgs = []; m.reset()
    while True:
        try: q = input("\n> ").strip()
        except EOFError: break
        if not q: continue
        if q == "/quit": break
        if q == "/reset": msgs = []; m.reset(); print("(new conversation)"); continue
        msgs.append({"role": "user", "content": q})
        e = tok("encm", msgs); ids, eos = e["ids"], set(e["eos"])
        if len(ids) + a.max > m.ctx:
            print(f"(conversation is {len(ids)} tokens, context is {m.ctx}: /reset)"); msgs.pop(); continue
        start = m.reuse(ids)
        out, tp, td, acc, _ = await run_one(m, dr, ids, eos, a.max, a.nd, start=start)
        body = [t for t in out if t not in eos]
        text = tok("dec", body)["text"]
        print(text)
        print(f"[prefill {len(ids) - start} tok {tp:.1f}s | {len(out) - 1} tok {(len(out) - 1) / td if td else 0:.1f} tok/s]", flush=True)
        msgs.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("--draft", default="mtp")
    ap.add_argument("--nd", type=int, default=7); ap.add_argument("--max", type=int, default=1024)
    asyncio.run(main(ap.parse_args()))
