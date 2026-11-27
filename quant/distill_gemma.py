"""Layer-wise distillation of the LUT4 quantization (step 5): indices frozen (from QDIR, e.g. gptq4b); per matrix train the
16 LUT values and a per-output-channel scale so that the quantized layer on the quantized stream matches the fp layer on the
fp stream (whole-layer output, residual included). Sequential over layers like gptq_gemma.py.
  QDIR=<gptq npz dir> OUT=<dir> START= END= STEPS=150 LR=2e-3 RB=4 [CACHE=<stream at START>]
Writes OUT/L{i}.npz in the same format (lut f16 (16,), idx uint8, scale f16 (Cout,)), OUT/log.json."""
import json, os, sys, time, copy
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.argv = sys.argv[:1]
os.environ.setdefault("OUT", "/tmp/unused_gptq_out")
src = (Path(__file__).parent / "gptq_gemma.py").read_text().rsplit("\nmain()", 1)[0]
exec(compile(src, "gptq_gemma.py", "exec"))      # cfg, get, rows, wmap, dev, L, hid, G, snr ... (main not run)

QDIR = Path(os.path.expanduser(os.environ["QDIR"])); DOUT = Path(os.path.expanduser(os.environ["DOUT"])); DOUT.mkdir(parents=True, exist_ok=True)
START = int(os.environ.get("START", 0)); END = int(os.environ.get("END", 60))
STEPS = int(os.environ.get("STEPS", 150)); LR = float(os.environ.get("LR", 2e-3)); RB = int(os.environ.get("RB", 4))
CACHE = os.environ.get("CACHE")
NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class QLinear(nn.Module):
    def __init__(self, z, n):
        super().__init__()
        self.vec = n + ".vlut" in z; self.key = "vlut" if self.vec else "lut"; self.seed = qfmt.seed_of(z, n)
        lut, idx, scale = z[n + "." + self.key], z[n + ".idx"], z[n + ".scale"]
        self.lut = nn.Parameter(torch.from_numpy(lut.astype(np.float32)).to(dev))
        self.register_buffer("idx", torch.from_numpy(idx).to(dev))                       # uint8 (Cout, Cin)
        self.register_buffer("s0", torch.from_numpy(scale.astype(np.float32)).to(dev))
        self.ls = nn.Parameter(torch.zeros_like(self.s0))                                  # scale = s0 * exp(ls)
    def weight(self):
        w = self.lut[self.idx.long()]
        if self.vec: w = w.permute(0, 2, 1).reshape(self.idx.shape[0] * self.lut.shape[1], self.idx.shape[1])
        return w * (self.s0 * self.ls.exp())[:, None]
    def forward(self, x):
        return F.linear(x if self.seed is None else qfmt.rot(x, self.seed, hid), self.weight())
    def enc(self):
        return {self.key: self.lut.detach().half().cpu().numpy(), "idx": self.idx.cpu().numpy(),
                "scale": (self.s0 * self.ls.exp()).detach().half().cpu().numpy()}


def mods(layer):
    at, mlp = layer.self_attn, layer.mlp
    out = {}
    for n in NAMES:
        par = mlp if n in ("gate_proj", "up_proj", "down_proj") else at
        if getattr(par, n, None) is not None: out[n] = par
    return out


def main():
    ids, valid, is_cal, names = rows()
    R = len(ids)
    rope = G.Gemma4TextRotaryEmbedding(cfg).to(dev); pos = torch.arange(L, device=dev)[None]
    pe = {lt: rope(torch.zeros(1, L, hid, device=dev), pos, lt) for lt in set(cfg.layer_types)}
    mask = torch.full((L, L), float("-inf"), device=dev).triu(1)[None, None]
    cal_rows = torch.nonzero(is_cal)[:, 0]; ev = ~is_cal
    val_rows = cal_rows[2::5]; fit_rows = torch.tensor([r for r in cal_rows.tolist() if r not in set(val_rows.tolist())])
    emb = get("model.language_model.embed_tokens.weight")
    if CACHE and START > 0:
        z = torch.load(os.path.expanduser(CACHE)); x_fp, x_q = z["x_fp"], z["x_q"]; print(f"stream cache loaded at L{START}", flush=True)
    else:
        assert START == 0; x_fp = (emb[ids] * hid ** 0.5).float(); x_q = x_fp.clone()
    del emb
    log = json.loads((DOUT / "log.json").read_text()) if (DOUT / "log.json").exists() else []

    def fwd(layer, i, x, r):
        return layer(x, position_embeddings=tuple(t.expand(len(r), -1, -1) for t in pe[cfg.layer_types[i]]), attention_mask=mask,
                     shared_kv_states={}, position_ids=pos.expand(len(r), -1))

    @torch.no_grad()
    def run(layer, i, x, sel=None):
        sel = torch.arange(R) if sel is None else sel; outs = []
        for a in range(0, len(sel), BS):
            r = sel[a:a + BS]; outs.append(fwd(layer, i, x[r].to(dev), r).cpu())
        return torch.cat(outs)

    def ssnr(y, yf, sel):
        v = valid[sel]; e = (y[sel] - yf[sel])[v]; return 10 * np.log10(float(yf[sel][v].pow(2).sum() / e.pow(2).sum()))

    for i in range(START, END):
        t0 = time.time()
        layer = G.Gemma4TextDecoderLayer(cfg, i).eval()
        pre = f"model.language_model.layers.{i}."
        sd = {k[len(pre):]: get(k) for k in wmap if k.startswith(pre)}
        assert not layer.load_state_dict(sd, strict=False).unexpected_keys; del sd
        layer.to(dev)
        y_fp = run(layer, i, x_fp)
        z = np.load(QDIR / f"L{i}.npz"); q = {}
        for n, par in mods(layer).items():
            q[n] = QLinear(z, n); setattr(par, n, q[n])
        for p in layer.parameters(): p.requires_grad_(False)
        params = [m.lut for m in q.values()] + [m.ls for m in q.values()]
        for p in params: p.requires_grad_(True)
        y0 = run(layer, i, x_q)
        s_ev0, s_val0 = ssnr(y0, y_fp, ev), ssnr(y0, y_fp, val_rows); del y0
        opt = torch.optim.Adam(params, lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        g = torch.Generator().manual_seed(i); tl = []
        for step in range(STEPS):
            r = fit_rows[torch.randperm(len(fit_rows), generator=g)[:RB]]
            v = valid[r].to(dev); yt = y_fp[r].to(dev)
            with torch.enable_grad():
                y = fwd(layer, i, x_q[r].to(dev), r)
                loss = (y - yt)[v].pow(2).sum() / yt[v].pow(2).sum()
                opt.zero_grad(set_to_none=True); loss.backward()
            opt.step(); sched.step(); tl.append(float(loss))
            continue
            loss = (y - yt)[v].pow(2).sum() / yt[v].pow(2).sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl.append(float(loss))
        with torch.no_grad():
            for m in q.values():   # round to what the export stores (f16)
                m.lut.data = m.lut.data.half().float(); m.ls.data = ((m.s0 * m.ls.exp()).half().float() / m.s0).log()
        y_q = run(layer, i, x_q)
        s_ev, s_val = ssnr(y_q, y_fp, ev), ssnr(y_q, y_fp, val_rows)
        kept = bool(s_val > s_val0)
        if kept:
            np.savez(DOUT / f"L{i}.npz", **{f"{n}.{k}": v for n, m in q.items() for k, v in m.enc().items()},
                     **{k: z[k] for k in qfmt.SEED0 if k in z})
        else:   # distillation did not help on held-out rows: keep the original
            np.savez(DOUT / f"L{i}.npz", **dict(z))
            for n, m in q.items(): m.lut.data = torch.from_numpy(z[n + "." + m.key].astype(np.float32)).to(dev); m.ls.data.zero_()
            y_q = run(layer, i, x_q); s_ev = s_ev0
        log = [e for e in log if e["layer"] != i] + [{"layer": i, "ev0": s_ev0, "ev": s_ev, "val0": s_val0, "val": s_val, "kept": kept,
                                                       "loss_first": float(np.mean(tl[:10])), "loss_last": float(np.mean(tl[-10:]))}]
        json.dump(sorted(log, key=lambda e: e["layer"]), open(DOUT / "log.json", "w"))
        print(f"L{i:02d} {cfg.layer_types[i][:4]} eval stream SNR {s_ev0:5.2f} -> {s_ev:5.2f} dB (held-out {s_val0:5.2f} -> {s_val:5.2f}, "
              f"{'kept' if kept else 'reverted'}; train loss {np.mean(tl[:10]):.4f} -> {np.mean(tl[-10:]):.4f}) ({time.time() - t0:.0f}s)", flush=True)
        x_fp, x_q = y_fp, y_q
        del layer, q, opt; torch.mps.empty_cache()
        if os.environ.get("SAVE_STREAM_AT") and i + 1 == int(os.environ["SAVE_STREAM_AT"]):
            torch.save({"x_fp": x_fp, "x_q": x_q}, os.path.expanduser(os.environ["STREAM_OUT"]))

    if END < cfg.num_hidden_layers:
        return
    # final eval with the QDIR head (unchanged): KL / top-1 vs the fp model on the eval rows
    import shutil; shutil.copy(QDIR / "head.npz", DOUT / "head.npz")
    norm = G.Gemma4RMSNorm(hid, eps=cfg.rms_norm_eps); norm.weight.data = get("model.language_model.norm.weight")
    zh = np.load(QDIR / "head.npz"); qh = (torch.from_numpy(zh["lut"].astype(np.float32))[torch.from_numpy(zh["idx"].astype(np.int64))]
                                          * torch.from_numpy(zh["scale"].astype(np.float32))[:, None]).to(dev)
    emb = get("model.language_model.embed_tokens.weight").to(dev)
    n_fp, n_q = norm(x_fp), norm(x_q); res = {}
    for grp in ("chat", "wiki", "ref64"):
        sel = [j for j in range(R) if names[j] == grp and not is_cal[j]]; kl, agree = [], []
        for j in sel:
            v = valid[j]
            lf = torch.tanh((n_fp[j][v].to(dev) @ emb.T) / 30) * 30; lq = torch.tanh((n_q[j][v].to(dev) @ qh.T) / 30) * 30
            pf, pq_ = lf.log_softmax(-1), lq.log_softmax(-1)
            kl.append((pf.exp() * (pf - pq_)).sum(-1).cpu()); agree.append((lf.argmax(-1) == lq.argmax(-1)).float().cpu())
        kl, agree = torch.cat(kl), torch.cat(agree)
        res[grp] = {"tokens": len(kl), "kl_mean": float(kl.mean()), "kl_median": float(kl.median()), "top1": float(agree.mean())}
        print(f"EVAL {grp}: {res[grp]}", flush=True)
    json.dump({"layers": json.loads((DOUT / "log.json").read_text()), "eval": res}, open(DOUT / "eval.json", "w"))


main()
