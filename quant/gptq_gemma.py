"""Layer-wise quantization of Gemma 4 31B to "LUT4 per-tensor + per-output-channel scale" with GPTQ error compensation
(sequential: every layer is calibrated on the output of the already-quantized layers), all in torch on MPS.
Streams one layer at a time; keeps an fp32 reference stream and the quantized stream for calibration + eval rows.
  METHOD=gptq|rtn  OUT=<dir>  END=<layers to do, default 60>  NCAL_WIKI=40 L=384
Writes OUT/L{i}.npz (per matrix: lut (16,) f16, idx uint8 (Cout, Cin), scale (Cout,) f16), OUT/head.npz, OUT/log.json.
GPTQ / Cholesky code: ANEMLL Forge scripts/qwen3_lut_common.py (MIT)."""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4 import modeling_gemma4 as G
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root (paths, gemma_ane, gemma_rt)
from paths import HF_MODEL, FORGE_SCRIPTS, CALIB, REF64
sys.path.insert(0, str(FORGE_SCRIPTS))
from qwen3_lut_common import gptq, kmeans_grouped_codebook, snr, make_rounder, encode, FORMATS
import qfmt

HERE = Path(__file__).parent
M = HF_MODEL
METHOD = os.environ.get("METHOD", "gptq"); OUT = Path(os.environ["OUT"]); OUT.mkdir(parents=True, exist_ok=True)
END = int(os.environ.get("END", 60)); L = int(os.environ.get("L", 384)); NW = int(os.environ.get("NCAL_WIKI", 40))
NBITS = int(os.environ.get("NBITS", 4)); BS = 8
DAMPS = [float(v) for v in os.environ.get("DAMPS", "0.01,0.1,1").split(",")]; SWEEP = os.environ.get("SWEEP") == "1"
ASYM = os.environ.get("ASYM") == "1"; ADAMP = float(os.environ.get("ADAMP", 0.05))
ADAMPS = [float(v) for v in os.environ.get("ADAMPS", str(ADAMP)).split(",")]
CACHE = os.environ.get("CACHE")   # stream cache file for x_fp, x_q at layer START   # GPTAQ-style asymmetric target
RESUME = os.environ.get("RESUME"); START = int(os.environ.get("START", 0))   # layers < START are loaded from RESUME
dev = torch.device("mps")
MLP_FMT, ATT_FMT = os.environ.get("MLP_FMT"), os.environ.get("ATT_FMT"); ROT = os.environ.get("ROT") == "1"
DF_ROWS = int(os.environ.get("DF_ROWS", 0))
cfg = Gemma4TextConfig(**json.loads((M / "config.json").read_text())["text_config"])
hid = cfg.hidden_size
wmap = json.loads((M / "model.safetensors.index.json").read_text())["weight_map"]
files = {}
def get(k):
    f = wmap[k]
    if f not in files: files[f] = safe_open(str(M / f), "pt")
    return files[f].get_tensor(k).float()


def rows():
    """(ids (R, L), valid (R, L), is_cal (R,), names) : chat rows (self-generated), wikitext rows, the ref64 sentence."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(M))
    out, cal, names = [], [], []
    chat = json.loads((CALIB / "chat_ids.json").read_text())
    for j, r in enumerate(chat):
        out.append(r[:L]); cal.append(j < 24); names.append("chat")
    for split, n, c in (("train", NW, True), ("test", 6, False)):
        txt = "\n".join(pq.read_table(CALIB / f"wikitext/wikitext-2-raw-v1/{split}-00000-of-00001.parquet")["text"].to_pylist())
        ids = tok(txt[:400000], add_special_tokens=False).input_ids
        step = max(L, (len(ids) - L) // n)
        for j in range(n):
            out.append([2] + ids[j * step: j * step + L - 1]); cal.append(c); names.append("wiki")
    if DF_ROWS:   # self-generated answers (zh / en / code / math) of the drafter data, prompt + answer
        import glob
        fs = sorted(glob.glob(os.path.expanduser(os.environ.get("DF_DIR", "~/dfdata/s")) + "/*.npz"), key=lambda f: int(Path(f).stem))
        for f in fs[-DF_ROWS:]:
            out.append(np.load(f)["ids"][:L].tolist()); cal.append(True); names.append("df")
    out.append(np.load(REF64 / "ids.npy").tolist()); cal.append(False); names.append("ref64")
    ids = np.zeros((len(out), L), np.int64); valid = np.zeros((len(out), L), bool)
    for j, r in enumerate(out):
        ids[j, :len(r)] = r; valid[j, :len(r)] = True
    return torch.from_numpy(ids), torch.from_numpy(valid), torch.tensor(cal), names


class Rounder:
    """Per-tensor scalar LUT (2^nb entries, weighted k-means on w / row RMS) + per-output-channel scale."""
    def __init__(self, w, cw=None, nb=NBITS):
        s = w.pow(2).mean(1, keepdim=True).sqrt().half().float(); s[s == 0] = 1
        c = kmeans_grouped_codebook((w / s).cpu(), w.shape[0], nb, None if cw is None else cw.cpu().float())[0]
        self.c = c.sort().values.to(w.device); self.mid = ((self.c[1:] + self.c[:-1]) / 2).contiguous(); self.s = s.to(w.device)
    def index(self, x, s=None):
        return torch.bucketize((x / (self.s if s is None else s)).contiguous(), self.mid)
    def __call__(self, x):
        return self.c[self.index(x)] * self.s
    def rows(self, a, b):   # view on a row range (the head is quantized in row chunks)
        r = Rounder.__new__(Rounder); r.c, r.mid, r.s = self.c, self.mid, self.s[a:b]; return r


def quant_fmt(w, h, hv, name, fmt):
    """Forge format `fmt` (FORMATS key): codebook fit weighted by diag(H), GPTQ damping ladder, held-out output SNR."""
    t0 = time.time()
    rnd = make_rounder(w.cpu(), FORMATS[fmt][1], cw=None if h is None else h.diag().cpu(), device=dev)
    def osnr(q, hh):
        return 10 * np.log10(float(((w @ hh) * w).sum() / (((w - q) @ hh) * (w - q)).sum().clamp_min(1e-30)))
    q = rnd(w); tag = "rtn"
    if METHOD == "gptq" and h is not None:
        base = best = osnr(q, hv); tried = [f"rtn {base:.1f}"]
        for d in DAMPS:
            c = gptq(w, h, rnd, damp=d); v = osnr(c, hv); tried.append(f"{d:g} {v:.1f}")
            if v > best: best, q, tag = v, c, f"gptq{d:g}"
            if v > base + 1 and not SWEEP: break
        tag += " [" + ", ".join(tried) + "]"
    lut, idx, sc = encode(rnd, q)
    if lut.shape[1] == 1: enc = {"lut": lut.reshape(-1).numpy(), "idx": idx.numpy(), "scale": sc.numpy()}
    else: enc = {"vlut": lut.numpy(), "idx": idx.numpy(), "scale": sc.numpy()}
    zz = {f"m.{k}": v for k, v in enc.items()}; wd, sd = qfmt.dense(zz, "m"); q = (wd * sd[:, None]).to(dev)
    wsnr = snr(w, q); o = wsnr if h is None else osnr(q, hv)
    print(f"    {name:10s} {tuple(w.shape)} {fmt}: weight SNR {wsnr:5.2f} dB, held-out output SNR {o:5.2f} dB {tag} ({time.time() - t0:.0f}s)", flush=True)
    return q, enc, o


def quant(w, h, hv, name):
    """w (Cout, Cin) on dev, h / hv = X^T X / N on the fit / held-out calibration rows (or None). Returns (q, enc dict, osnr).
    GPTQ can diverge (ill-conditioned H): candidates are RTN and GPTQ over the damping ladder, chosen by held-out output SNR."""
    t0 = time.time()
    rnd = Rounder(w, None if h is None else h.diag())
    def osnr(q, hh):
        return 10 * np.log10(float(((w @ hh) * w).sum() / (((w - q) @ hh) * (w - q)).sum().clamp_min(1e-30)))
    q = rnd(w); tag = "rtn"
    if METHOD == "gptq" and h is not None:
        base = best = osnr(q, hv); tried = [f"rtn {base:.1f}"]
        for d in DAMPS:
            c = gptq(w, h, rnd, damp=d); v = osnr(c, hv); tried.append(f"{d:g} {v:.1f}")
            if v > best: best, q, tag = v, c, f"gptq{d:g}"
            if v > base + 1 and not SWEEP: break
        tag += " [" + ", ".join(tried) + "]"
    idx = rnd.index(q).to(torch.uint8)
    q = rnd.c[idx.long()] * rnd.s
    enc = {"lut": rnd.c.half().cpu().numpy(), "idx": idx.cpu().numpy(), "scale": rnd.s.reshape(-1).half().cpu().numpy()}
    wsnr = snr(w, q)
    o = wsnr if h is None else osnr(q, hv)
    print(f"    {name:10s} {tuple(w.shape)} weight SNR {wsnr:5.2f} dB, held-out output SNR {o:5.2f} dB {tag} ({time.time() - t0:.0f}s)", flush=True)
    return q, enc, o


def corr_matrix(h, c, damp):
    """M = (H + damp)^-1 C, C = Z_q^T (Z_fp - Z_q) / N.  W' = W + W M^T is the least-squares weight for the quantized-stream input:
    argmin ||Z_q W'^T - Z_fp W^T|| (damped toward W).  float64 on CPU."""
    hd = h.cpu().double().clone(); dead = hd.diag() == 0; hd[dead, dead] = 1
    hd += damp * hd.diag().mean() * torch.eye(len(hd), dtype=hd.dtype)
    m = torch.cholesky_solve(c.cpu().double(), torch.linalg.cholesky(hd))
    return m.float().to(c.device)


def quant_asym(w, h, mcs, zqv, zfv, name):
    """Candidates RTN(W), GPTQ(W' = W + W M^T), GPTQ(W) if needed; chosen by the held-out asymmetric error ||Z_q Q^T - Z_fp W^T||."""
    t0 = time.time()
    def verr(q):
        e = r = 0.0
        for a in range(0, len(zqv), 4096):
            zq, zf = zqv[a:a + 4096].float(), zfv[a:a + 4096].float()
            yf = zf @ w.T; e += float((zq @ q.T - yf).pow(2).sum()); r += float(yf.pow(2).sum())
        return 10 * np.log10(r / max(e, 1e-30))
    d0 = DAMPS[0]
    rw = Rounder(w, h.diag()); q, rnd, tag = rw(w), rw, "rtn"; best = base = verr(q); tried = [f"rtn {base:.1f}"]
    c = gptq(w, h, rw, damp=d0); v = verr(c); tried.append(f"sym {v:.1f}")
    if v > best: best, q, rnd, tag = v, c, rw, "sym"
    for ad, mcorr in mcs:
        wp = w + w @ mcorr.T
        rp = Rounder(wp, h.diag()); c = gptq(wp, h, rp, damp=d0); v = verr(c); tried.append(f"a{ad:g} {v:.1f}/{snr(w, c):.0f}")
        if v > best: best, q, rnd, tag = v, c, rp, f"asym{ad:g}"
    idx = rnd.index(q).to(torch.uint8)
    q = rnd.c[idx.long()] * rnd.s
    enc = {"lut": rnd.c.half().cpu().numpy(), "idx": idx.cpu().numpy(), "scale": rnd.s.reshape(-1).half().cpu().numpy()}
    o = verr(q)
    print(f"    {name:10s} {tuple(w.shape)} weight SNR {snr(w, q):5.2f} dB, held-out asym SNR {o:5.2f} dB {tag} [{', '.join(tried)}] ({time.time() - t0:.0f}s)", flush=True)
    return q, enc, o


class Stop(Exception):
    pass


def main():
    ids, valid, is_cal, names = rows()
    R = len(ids); print(f"rows {R} (calib {int(is_cal.sum())}), L {L}, calib tokens {int(valid[is_cal].sum())}, method {METHOD}, bits {NBITS}", flush=True)
    emb = get("model.language_model.embed_tokens.weight")
    x_fp = (emb[ids] * hid ** 0.5).float(); x_q = x_fp.clone()
    rope = G.Gemma4TextRotaryEmbedding(cfg).to(dev)
    pos = torch.arange(L, device=dev)[None]
    pe = {lt: rope(torch.zeros(1, L, hid, device=dev), pos, lt) for lt in set(cfg.layer_types)}
    mask = torch.full((L, L), float("-inf"), device=dev).triu(1)[None, None]
    cal_rows = torch.nonzero(is_cal)[:, 0]; ev = ~is_cal
    val_rows = cal_rows[2::5]; fit_rows = torch.tensor([r for r in cal_rows.tolist() if r not in set(val_rows.tolist())])
    log = []

    def run(layer, i, x, sel=None, capture=None):
        """Layer over rows `sel` of stream x (CPU) -> outputs (CPU), or the Hessian of `capture`'s input (valid positions)."""
        sel = torch.arange(R) if sel is None else sel
        outs, h, n = [], None, 0
        hook = None
        if capture is not None:
            box = {}
            def pre(mod, args):
                box["x"] = args[0]; raise Stop
            hook = capture.register_forward_pre_hook(pre)
        for a in range(0, len(sel), BS):
            r = sel[a:a + BS]; xb = x[r].to(dev)
            try:
                y = layer(xb, position_embeddings=tuple(t.expand(len(r), -1, -1) for t in pe[cfg.layer_types[i]]), attention_mask=mask,
                          shared_kv_states={}, position_ids=pos.expand(len(r), -1))
                outs.append(y.cpu())
            except Stop:
                z = box["x"][valid[r].to(dev)].float()
                h = z.T @ z if h is None else h + z.T @ z; n += len(z)
        if hook is not None:
            hook.remove(); return h / n
        return torch.cat(outs)

    def run_pair(layer, layer_fp, i, sel, name, fit):
        """Inputs of module `name` in the (partly) quantized layer on x_q and in the fp layer on x_fp, same positions.
        fit: (H, C) = (Z_q^T Z_q, Z_q^T (Z_fp - Z_q)) / N.  else: (Z_q, Z_fp) as fp16 tensors on dev."""
        box = {}
        def mk(key):
            def pre(mod, args):
                box[key] = args[0]; raise Stop
            return pre
        hq = layer.get_submodule(name).register_forward_pre_hook(mk("q")); hf = layer_fp.get_submodule(name).register_forward_pre_hook(mk("f"))
        h = c = None; n = 0; zs = []
        kw = lambda r: dict(position_embeddings=tuple(t.expand(len(r), -1, -1) for t in pe[cfg.layer_types[i]]), attention_mask=mask,
                            shared_kv_states={}, position_ids=pos.expand(len(r), -1))
        for a in range(0, len(sel), BS):
            r = sel[a:a + BS]; vm = valid[r].to(dev)
            for key, lay, x in (("q", layer, x_q), ("f", layer_fp, x_fp)):
                try:
                    lay(x[r].to(dev), **kw(r))
                except Stop:
                    pass
            zq, zf = box["q"][vm].float(), box["f"][vm].float()
            if fit:
                h = zq.T @ zq if h is None else h + zq.T @ zq
                d = zq.T @ (zf - zq); c = d if c is None else c + d; n += len(zq)
            else:
                zs.append((zq.half(), zf.half()))
        hq.remove(); hf.remove()
        if fit:
            return h / n, c / n
        return torch.cat([a for a, _ in zs]), torch.cat([b for _, b in zs])

    with torch.no_grad():
        if CACHE and START > 0 and Path(CACHE).exists():
            z = torch.load(CACHE); x_fp, x_q = z["x_fp"], z["x_q"]; print(f"stream cache loaded at L{START}", flush=True)
        for i in range(END):
            if CACHE and START > 0 and i < START and Path(CACHE).exists():
                continue
            if CACHE and i == START and i > 0 and not Path(CACHE).exists():
                torch.save({"x_fp": x_fp, "x_q": x_q}, CACHE); print(f"stream cache saved at L{START}", flush=True)
            t0 = time.time()
            layer = G.Gemma4TextDecoderLayer(cfg, i).eval()
            pre = f"model.language_model.layers.{i}."
            sd = {k[len(pre):]: get(k) for k in wmap if k.startswith(pre)}
            miss = layer.load_state_dict(sd, strict=False); assert not miss.unexpected_keys
            del sd; layer.to(dev)
            y_fp = run(layer, i, x_fp)
            at, mlp = layer.self_attn, layer.mlp
            stages = [([("q_proj", at.q_proj), ("k_proj", at.k_proj)] + ([("v_proj", at.v_proj)] if getattr(at, "v_proj", None) is not None else [])),
                      [("o_proj", at.o_proj)], [("gate_proj", mlp.gate_proj), ("up_proj", mlp.up_proj)], [("down_proj", mlp.down_proj)]]
            encs, sn = {}, {}
            if RESUME and i < START:
                z = np.load(Path(RESUME) / f"L{i}.npz")
                for st in stages:
                    for name, mod in st:
                        mod.weight.data = (torch.from_numpy(z[name + ".lut"].astype(np.float32))[torch.from_numpy(z[name + ".idx"].astype(np.int64))]
                                           * torch.from_numpy(z[name + ".scale"].astype(np.float32))[:, None]).to(dev)
                if Path(RESUME).resolve() != OUT.resolve(): np.savez(OUT / f"L{i}.npz", **dict(z))
            elif ASYM:
                import copy
                layer_fp = copy.deepcopy(layer)
                full = {m: n for n, m in layer.named_modules()}
                for st in stages:
                    nm = full[st[0][1]]
                    h, c = run_pair(layer, layer_fp, i, fit_rows, nm, True)
                    zqv, zfv = run_pair(layer, layer_fp, i, val_rows, nm, False)
                    mc = [(ad, corr_matrix(h, c, ad)) for ad in ADAMPS]; del c
                    for name, mod in st:
                        q, encs[name], sn[name] = quant_asym(mod.weight.data.float(), h, mc, zqv, zfv, name)
                        mod.weight.data = q
                    del h, mc, zqv, zfv
                del layer_fp
                np.savez(OUT / f"L{i}.npz", **{f"{n}.{k}": v for n, e in encs.items() for k, v in e.items()})
            else:
                seeds = {k: v + i for k, v in qfmt.SEED0.items()} if ROT else {}
                for st in stages:
                    h = hv = None
                    if METHOD == "gptq":
                        h = run(layer, i, x_q, fit_rows, capture=st[0][1]); hv = run(layer, i, x_q, val_rows, capture=st[0][1])
                    for name, mod in st:
                        sd_ = seeds.get(qfmt.ROT_KEY[name]); w = mod.weight.data.float(); hh, hvv = h, hv
                        if sd_ is not None:
                            w = qfmt.rot(w, sd_, hid); hh, hvv = qfmt.rot_h(h, sd_, hid), qfmt.rot_h(hv, sd_, hid)
                        fmt = MLP_FMT if name in ("gate_proj", "up_proj", "down_proj") else ATT_FMT
                        q, encs[name], sn[name] = quant_fmt(w, hh, hvv, name, fmt) if fmt else quant(w, hh, hvv, name)
                        mod.weight.data = qfmt.unrot(q, sd_, hid) if sd_ is not None else q
                        del hh, hvv
                    del h, hv
                np.savez(OUT / f"L{i}.npz", **{f"{n}.{k}": v for n, e in encs.items() for k, v in e.items()}, **seeds)
            y_q = run(layer, i, x_q)
            e = (y_q - y_fp)[ev][valid[ev]]; ref = y_fp[ev][valid[ev]]
            s_stream = 10 * np.log10(float(ref.pow(2).sum() / e.pow(2).sum().clamp_min(1e-30)))
            cos = float(torch.nn.functional.cosine_similarity(y_q[ev][valid[ev]], ref, dim=-1).mean())
            log.append({"layer": i, "stream_snr": s_stream, "cos": cos, **sn}); json.dump(log, open(OUT / "log.json", "w"))
            print(f"L{i:02d} {cfg.layer_types[i][:4]} eval stream SNR {s_stream:5.2f} dB cos {cos:.5f} ({time.time() - t0:.0f}s)", flush=True)
            x_fp, x_q = y_fp, y_q
            del layer
            torch.mps.empty_cache()
        if END < cfg.num_hidden_layers or os.environ.get("NO_HEAD"):
            return
        # head: logits = norm(x) @ E^T, E quantized in row chunks (GPTQ is independent per row given the LUT and H)
        norm = G.Gemma4RMSNorm(hid, eps=cfg.rms_norm_eps); norm.weight.data = get("model.language_model.norm.weight")
        n_fp, n_q = norm(x_fp), norm(x_q)
        z = n_q[is_cal][valid[is_cal]].to(dev); h = z.T @ z / len(z)
        HD = float(os.environ.get("HEAD_DAMP", 0.1))
        if ASYM:
            zf = n_fp[is_cal][valid[is_cal]].to(dev); c = z.T @ (zf - z) / len(z); del zf
            mc = corr_matrix(h, c, ADAMP).cpu(); del c
            emb_t = emb.float().clone()
            for a in range(0, emb.shape[0], 16384):
                emb_t[a:a + 16384] += (emb[a:a + 16384].to(dev).float() @ mc.to(dev).T).cpu()
        else:
            emb_t = emb
        rnd = Rounder(emb_t, h.diag().cpu())      # fitted on CPU (5.6 GB matrix)
        idx = torch.empty(emb.shape, dtype=torch.uint8); CH = 16384
        for a in range(0, emb.shape[0], CH):
            w = emb_t[a:a + CH].to(dev); r = rnd.rows(a, a + CH); r.c, r.mid, r.s = r.c.to(dev), r.mid.to(dev), r.s.to(dev)
            q = gptq(w, h, r, damp=HD) if METHOD == "gptq" else r(w)
            idx[a:a + CH] = r.index(q).to(torch.uint8).cpu()
        np.savez(OUT / "head.npz", lut=rnd.c.half().numpy(), idx=idx.numpy(), scale=rnd.s.reshape(-1).half().numpy())
        qh = (rnd.c[idx.long()] * rnd.s)
        print(f"head weight SNR {snr(emb, qh):.2f} dB", flush=True)
        res = {}
        for grp in ("chat", "wiki", "ref64"):
            sel = [j for j in range(R) if names[j] == grp and not is_cal[j]]
            kl, agree, klh = [], [], []
            for j in sel:
                v = valid[j]
                lf = n_fp[j][v].to(dev) @ emb.to(dev).T; lq = n_q[j][v].to(dev) @ qh.to(dev).T
                lf, lq = torch.tanh(lf / 30) * 30, torch.tanh(lq / 30) * 30
                pf, pq_ = lf.log_softmax(-1), lq.log_softmax(-1)
                kl.append((pf.exp() * (pf - pq_)).sum(-1).cpu()); agree.append((lf.argmax(-1) == lq.argmax(-1)).float().cpu())
                lh = n_q[j][v].to(dev) @ emb.to(dev).T; lh = (torch.tanh(lh / 30) * 30).log_softmax(-1)
                klh.append((pf.exp() * (pf - lh)).sum(-1).cpu())
            kl, agree = torch.cat(kl), torch.cat(agree)
            res[grp] = {"tokens": len(kl), "kl_mean": float(kl.mean()), "kl_median": float(kl.median()), "top1": float(agree.mean()), "kl_mean_fp_head": float(torch.cat(klh).mean())}
            print(f"EVAL {grp}: {res[grp]}", flush=True)
        json.dump({"layers": log, "eval": res}, open(OUT / "log.json", "w"))


main()
