"""Whole-model (end-to-end) distillation of the LUT quantization: indices frozen (QDIR), train every matrix's LUT values and
per-output-channel log-scale jointly so that the quantized model's final logits match the fp model (KL, softcap 30).
31B does not fit in memory, so every step streams layers: forward 0..59 without grad (stream kept), loss on the final
hidden state, then backward 59..0 recomputing each layer with grad. Params + Adam state (~4M) live on the GPU; per-layer
indices are re-read from QDIR. The head stays as in QDIR.
  QDIR= DOUT= STEPS=120 LR=1e-3 B=4 EVAL_EVERY=30 NCAL_WIKI=200 [TEACHER=<file>] [MAXSTEPS_SMOKE=]"""
import json, os, sys, time, math, shutil
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.argv = sys.argv[:1]; os.environ.setdefault("OUT", "/tmp/unused_gptq_out")
src = (Path(__file__).parent / "gptq_gemma.py").read_text().rsplit("\nmain()", 1)[0]
exec(compile(src, "gptq_gemma.py", "exec"))      # cfg, get, rows, wmap, dev, L, hid, G, BS

QDIR = Path(os.path.expanduser(os.environ["QDIR"])); DOUT = Path(os.path.expanduser(os.environ["DOUT"])); DOUT.mkdir(parents=True, exist_ok=True)
STEPS = int(os.environ.get("STEPS", 120)); LR = float(os.environ.get("LR", 1e-3)); B = int(os.environ.get("B", 4))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 30)); WARM = int(os.environ.get("WARM", 5))
TEACHER = Path(os.path.expanduser(os.environ.get("TEACHER", "~/gemma-ane-work/teacher_final.pt")))
NL = cfg.num_hidden_layers; CAP = 30.0
NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
torch.manual_seed(0)


def par_of(layer, n):
    return layer.mlp if n in ("gate_proj", "up_proj", "down_proj") else layer.self_attn


class QLin(nn.Module):
    """weights = lut[idx] * (s0 * exp(ls)); tensors are attached per layer (self.t), not registered."""
    def __init__(self):
        super().__init__(); self.t = None
    def forward(self, x):
        lut, ls, s0, idx, vec, seed = self.t
        w = lut[idx.int()]
        if vec: w = w.permute(0, 2, 1).reshape(idx.shape[0] * lut.shape[1], idx.shape[1])
        return F.linear(x if seed is None else qfmt.rot(x, seed, hid), w * (s0 * ls.exp())[:, None])


def main():
    ids, valid, is_cal, names = rows(); R = len(ids)
    rope = G.Gemma4TextRotaryEmbedding(cfg).to(dev); pos = torch.arange(L, device=dev)[None]
    pe = {lt: rope(torch.zeros(1, L, hid, device=dev), pos, lt) for lt in set(cfg.layer_types)}
    mask = torch.full((L, L), float("-inf"), device=dev).triu(1)[None, None]
    cal_rows = torch.nonzero(is_cal)[:, 0]
    val_rows = cal_rows[2::5]; fit_rows = torch.tensor([r for r in cal_rows.tolist() if r not in set(val_rows.tolist())])
    ev_rows = torch.nonzero(~is_cal)[:, 0]
    print(f"rows {R}: fit {len(fit_rows)}, val {len(val_rows)}, eval {len(ev_rows)}; B {B}, steps {STEPS}, lr {LR}", flush=True)
    emb = get("model.language_model.embed_tokens.weight")                      # (V, hid) bf16, CPU
    norm = G.Gemma4RMSNorm(hid, eps=cfg.rms_norm_eps).to(dev); norm.weight.data = get("model.language_model.norm.weight").float().to(dev)
    for p in norm.parameters(): p.requires_grad_(False)

    def fwd(layer, i, x, r):
        return layer(x, position_embeddings=tuple(t.expand(len(r), -1, -1) for t in pe[cfg.layer_types[i]]), attention_mask=mask,
                     shared_kv_states={}, position_ids=pos.expand(len(r), -1))

    # ---- teacher: fp model, normed final hidden state for all rows (fp16, CPU)
    if TEACHER.exists():
        tf = torch.load(TEACHER); print("teacher loaded", flush=True)
    else:
        t0 = time.time(); x = (emb[ids] * hid ** 0.5).float()
        with torch.no_grad():
            for i in range(NL):
                layer = G.Gemma4TextDecoderLayer(cfg, i).eval(); pre = f"model.language_model.layers.{i}."
                assert not layer.load_state_dict({k[len(pre):]: get(k) for k in wmap if k.startswith(pre)}, strict=False).unexpected_keys
                layer.to(dev); out = []
                for a in range(0, R, BS):
                    r = torch.arange(a, min(a + BS, R)); out.append(fwd(layer, i, x[r].to(dev), r).cpu())
                x = torch.cat(out); del layer; torch.mps.empty_cache()
                if i % 10 == 9: print(f"  teacher L{i} ({time.time() - t0:.0f}s)", flush=True)
            tf = torch.cat([norm(x[a:a + 16].to(dev)).half().cpu() for a in range(0, R, 16)])
        torch.save(tf, TEACHER); print(f"teacher saved ({time.time() - t0:.0f}s)", flush=True); del x

    # ---- heads on the GPU (fp16): fp teacher head, quantized student head (fixed)
    E = emb.to(dev).half()
    zh = np.load(QDIR / "head.npz")
    hl, hsc = torch.from_numpy(zh["lut"].astype(np.float32)), torch.from_numpy(zh["scale"].astype(np.float32)); hidx = zh["idx"]
    QH = torch.cat([(hl[torch.from_numpy(hidx[a:a + 16384]).long()] * hsc[a:a + 16384, None]).half() for a in range(0, len(hsc), 16384)]).to(dev); del zh, hidx
    shutil.copy(QDIR / "head.npz", DOUT / "head.npz")

    # ---- student params, small per-layer state, two template layers (sliding / full)
    P, small, META = [], [], {}
    for i in range(NL):
        z = np.load(QDIR / f"L{i}.npz"); d = {}
        for n in NAMES:
            key = n + (".vlut" if n + ".vlut" in z else ".lut")
            if key in z:
                META[(i, n)] = (key, key.endswith("vlut"), qfmt.seed_of(z, n))
                d[n] = (nn.Parameter(torch.from_numpy(z[key].astype(np.float32)).to(dev)),
                        nn.Parameter(torch.zeros(z[n + ".scale"].shape[0], device=dev)),
                        torch.from_numpy(z[n + ".scale"].astype(np.float32)).to(dev))
        P.append(d); pre = f"model.language_model.layers.{i}."
        small.append({k[len(pre):]: get(k) for k in wmap if k.startswith(pre) and not any(f".{n}." in k for n in NAMES)})
    params = [t for d in P for (lut, ls, _) in d.values() for t in (lut, ls)]
    print(f"trainable params {sum(p.numel() for p in params) / 1e6:.2f} M", flush=True)
    tmpl = {}
    for lt in set(cfg.layer_types):
        i0 = cfg.layer_types.index(lt); layer = G.Gemma4TextDecoderLayer(cfg, i0).eval()
        for n in NAMES:
            if getattr(par_of(layer, n), n, None) is not None: setattr(par_of(layer, n), n, QLin())
        miss = layer.load_state_dict(small[i0], strict=False)
        assert not miss.unexpected_keys and all(any(f"{n}." in k for n in NAMES) for k in miss.missing_keys), miss
        layer.to(dev)
        for p in layer.parameters(): p.requires_grad_(False)
        tmpl[lt] = layer
    for i in range(NL):   # every layer of a type must have identical small-state keys / shapes
        ref = small[cfg.layer_types.index(cfg.layer_types[i])]
        assert {k: v.shape for k, v in small[i].items()} == {k: v.shape for k, v in ref.items()}, i

    def setup(i):
        layer = tmpl[cfg.layer_types[i]]; layer.load_state_dict(small[i], strict=False)
        z = np.load(QDIR / f"L{i}.npz")
        for n, (lut, ls, s0) in P[i].items():
            _, vec, seed = META[(i, n)]
            getattr(par_of(layer, n), n).t = (lut, ls, s0, torch.from_numpy(z[n + ".idx"]).to(dev), vec, seed)
        return layer

    def release(layer):
        for n in NAMES:
            m = getattr(par_of(layer, n), n, None)
            if isinstance(m, QLin): m.t = None

    def kl_tokens(hs, ht, v):
        """hs (n, hid) student normed (any dtype, may need grad), ht teacher normed fp16 -> per-token KL (fp32)."""
        lt = torch.tanh((ht.to(dev).half() @ E.T).float() / CAP) * CAP
        ls_ = torch.tanh((hs.half() @ QH.T).float() / CAP) * CAP
        pt = lt.log_softmax(-1); return (pt.exp() * (pt - ls_.log_softmax(-1))).sum(-1)

    def forward_rows(sel, keep=False):
        x = (emb[ids[sel]] * hid ** 0.5).float().to(dev); xs = [x] if keep else None
        with torch.no_grad():
            for i in range(NL):
                layer = setup(i); out = []
                for a in range(0, len(sel), BS):
                    r = sel[a:a + BS]; out.append(fwd(layer, i, x[a:a + BS], r))
                x = torch.cat(out); release(layer)
                if keep: xs.append(x)
        return x, xs

    def evaluate(tag):
        t0 = time.time(); sel = torch.cat([val_rows, ev_rows]); x, _ = forward_rows(sel); res = {}
        with torch.no_grad():
            groups = {"val": [int(r) for r in val_rows]} | {g: [int(r) for r in ev_rows if names[r] == g] for g in ("chat", "wiki", "ref64")}
            pos_of = {int(r): j for j, r in enumerate(sel.tolist())}
            for g, rs in groups.items():
                kl, agree = [], []
                for r in rs:
                    v = valid[r]; hs = norm(x[pos_of[r]][v.to(dev)]); ht = tf[r][v]
                    kl.append(kl_tokens(hs, ht, v).cpu())
                    lt = (ht.to(dev).half() @ E.T); lq = (hs.half() @ QH.T); agree.append((lt.argmax(-1) == lq.argmax(-1)).float().cpu())
                kl, agree = torch.cat(kl), torch.cat(agree)
                res[g] = {"tokens": len(kl), "kl_mean": float(kl.mean()), "kl_median": float(kl.median()), "top1": float(agree.mean())}
        print(f"EVAL {tag}: " + "  ".join(f"{g} KL {r['kl_mean']:.4f} top1 {r['top1']:.3f}" for g, r in res.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
        return res

    def save():
        for i in range(NL):
            z = np.load(QDIR / f"L{i}.npz"); out = dict(z)
            for n, (lut, ls, s0) in P[i].items():
                out[META[(i, n)][0]] = lut.detach().half().cpu().numpy(); out[n + ".scale"] = (s0 * ls.exp()).detach().half().cpu().numpy()
            np.savez(DOUT / f"L{i}.npz", **out)

    def round_params():   # snap to what the export stores (f16)
        with torch.no_grad():
            for d in P:
                for lut, ls, s0 in d.values():
                    lut.data = lut.data.half().float(); ls.data = ((s0 * ls.exp()).half().float() / s0).log()

    opt = torch.optim.Adam(params, lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
    hist = {"eval": [], "loss": []}
    best = evaluate("step 0"); hist["eval"].append({"step": 0, **best}); best_val = best["val"]["kl_mean"]
    g_ = torch.Generator().manual_seed(1)
    for step in range(1, STEPS + 1):
        t0 = time.time(); r = fit_rows[torch.randperm(len(fit_rows), generator=g_)[:B]]
        _, xs = forward_rows(r, keep=True); t1 = time.time()
        v = valid[r].to(dev); xe = xs[-1].detach().requires_grad_(True)
        with torch.enable_grad():
            hs = norm(xe)[v]; ht = tf[r][valid[r]]; n = hs.shape[0]; loss = 0.0
            for a in range(0, n, 256):
                l = kl_tokens(hs[a:a + 256], ht[a:a + 256], None).sum() / n; l.backward(retain_graph=a + 256 < n); loss += float(l)
        g = xe.grad; del xe, hs
        for i in reversed(range(NL)):
            layer = setup(i); xin = xs[i].detach().requires_grad_(True)
            with torch.enable_grad():
                y = torch.cat([fwd(layer, i, xin[a:a + BS], r[a:a + BS]) for a in range(0, len(r), BS)])
                y.backward(g)
            g = xin.grad; release(layer); xs[i + 1] = None; del y, xin
        gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))
        opt.step(); opt.zero_grad(set_to_none=True); sched.step(); del xs, g; torch.mps.empty_cache()
        hist["loss"].append(loss)
        print(f"step {step:3d} loss {loss:.4f} gnorm {gn:.3f} lr {sched.get_last_lr()[0]:.2e} (fwd {t1 - t0:.0f}s, total {time.time() - t0:.0f}s)", flush=True)
        if step % EVAL_EVERY == 0 or step == STEPS:
            round_params(); res = evaluate(f"step {step}"); hist["eval"].append({"step": step, **res})
            if res["val"]["kl_mean"] < best_val:
                best_val = res["val"]["kl_mean"]; save(); print(f"  saved (best val KL {best_val:.4f})", flush=True)
            json.dump(hist, open(DOUT / "hist.json", "w"))
    if not (DOUT / "L0.npz").exists():
        print("no improvement; DOUT holds only head.npz", flush=True)


main()
