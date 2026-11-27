"""Gemma 4 31B on the Apple Neural Engine via Core AI: chunk builder + host runtime.
  build  <out_dir> [--layers a-b] [--chunk 6] [--ctx 4096] [--quant none|palg<gs>|int8c|int4c]
  head   <out_dir> [--quant ...]
  Runtime: class GemmaANE (see validate.py / generate.py).
Design (after anemll-forge): x as (1, C, 1, T); 1x1 convs; scale-free RMSNorm; KV history as read-only inputs,
k/v rows of the block as outputs, host commits rows in place; entry points v1 / v8 share one weight copy."""
from __future__ import annotations
import asyncio, ctypes, gc, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import HF_MODEL as MODEL
_CFG = MODEL / "config.json" if (MODEL / "config.json").exists() else Path(__file__).resolve().parent / "data/gemma-4-31B-it.config.json"
CFG = json.loads(_CFG.read_text())["text_config"]   # runtime works without the original checkpoint
hid, inter, nh, NL = CFG["hidden_size"], CFG["intermediate_size"], CFG["num_attention_heads"], CFG["num_hidden_layers"]
EPS, WIN, VOCAB = CFG["rms_norm_eps"], CFG["sliding_window"], CFG["vocab_size"]
f16 = np.float16
NEG = -1e4
KT_ATTN = bool(int(os.environ.get("KT_ATTN", 0)))   # global K history as (nkv, hd, ctx); manifest kT=true
KT_BLOCK = int(os.environ.get("KT_BLOCK", 2048))
# K history in TL-key tiles (nkv, ctx/TL, hd, TL) for global layers with ctx >= KT_TILE_MIN, so the host writes whole
# 128-byte runs per row instead of scattering 2-byte columns. Added while chasing 300-900 ms ANE call stalls, which later
# turned out to come from the ANE waking up after idle, not from the layout; kept (the released build uses KT_TILE_MIN=0),
# costs 0.3-0.8 ms per call over the plain transposed layout.
KT_TILE = int(os.environ.get("KT_TILE", 64)); KT_TILE_MIN = int(os.environ.get("KT_TILE_MIN", 8192))
# global-layer scores computed at KT_QSCALE and rescaled after the max is subtracted (Gemma 4 has no 1/sqrt(d) there);
# mathematically the same, kept from the same stall hunt
KT_QSCALE = float(os.environ.get("KT_QSCALE", 0.125))
def kt_tile(ctx): return KT_TILE if (KT_ATTN and KT_TILE and ctx >= KT_TILE_MIN) else 0
_idx = None


def lcfg(i):
    sliding = CFG["layer_types"][i] == "sliding_attention"
    return sliding, (CFG["num_key_value_heads"] if sliding else CFG["num_global_key_value_heads"]), (CFG["head_dim"] if sliding else CFG["global_head_dim"])


def tensor(key):
    from safetensors import safe_open
    global _idx
    if _idx is None:
        _idx = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(MODEL / _idx[key]), "pt") as f:
        return f.get_tensor(key)


LUT_NB = 4   # LUT bits for --quant lut (set per chunk from the npz LUT size)
QDIR = None   # set by --qdir: gptq_gemma.py export (LUT4 per tensor + per-output-channel scale)
TAPS = {int(t) for t in os.environ.get("TAPS", "").split(",") if t}   # extra hidden outputs (drafter features)
QMAP = dict(q="q_proj", k="k_proj", v="v_proj", o="o_proj", gate="gate_proj", up="up_proj", down="down_proj")


KNOWN_LUTS = {}   # wkey(dense fp16 weight) -> (lut (K, d), idx) for exact vector / scalar LUT packing
HB_HID = int(os.environ.get("HB_HID", 256))   # Hadamard block on the 5376-wide inputs (5376 = 21 x 256)


def wkey(a):
    import hashlib
    a = np.ascontiguousarray(a, f16); return f"{a.size}:{hashlib.sha1(a.data).hexdigest()}"


def qmat(z, name):
    """(dense fp16 (Cout, Cin) = scalar LUT[idx] or vector LUT rows, scale (Cout,) fp16); vector LUTs are registered."""
    if f"{name}.vlut" in z:
        lut, idx = z[f"{name}.vlut"].astype(f16), z[f"{name}.idx"]
        w = lut[idx].transpose(0, 2, 1).reshape(idx.shape[0] * lut.shape[1], idx.shape[1])
        KNOWN_LUTS[wkey(w)] = (lut, idx)
        return w, z[f"{name}.scale"].astype(f16)
    return z[f"{name}.lut"].astype(f16)[z[f"{name}.idx"]], z[f"{name}.scale"].astype(f16)


class Hadamard(nn.Module):
    """x (1, n, 1, T) -> x M, M = blockdiag(diag(signs) H_block) / sqrt(block) (online rotation, seeded signs).
    Sylvester H_block = H_b1 (x) H_b2, applied as two small matmuls (constants n + 2*b1^2 instead of n*block)."""
    def __init__(self, n, seed, block):
        super().__init__()
        from scipy.linalg import hadamard
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], n)
        b1 = 1 << ((block.bit_length() - 1) // 2); b2 = block // b1
        self.n, self.block, self.b1, self.b2 = n, block, b1, b2
        self.register_buffer("s", torch.from_numpy((signs / np.sqrt(block)).astype(f16)).view(1, n, 1, 1))
        self.register_buffer("h1", torch.from_numpy(hadamard(b1).T.astype(f16)))
        self.register_buffer("h2", torch.from_numpy(hadamard(b2).T.astype(f16)))

    def forward(self, x):
        T = x.shape[-1]; nb = self.n // self.block
        x = x * self.s
        x = torch.matmul(self.h2, x.reshape(1, nb * self.b1, self.b2, T))
        x = torch.matmul(self.h1, x.reshape(1, nb, self.b1, self.b2 * T))
        return x.reshape(1, self.n, 1, T)


def hblock(n):
    return HB_HID if n == hid else 1024


def layer_weights(i):
    W = _layer_weights(i)
    if QDIR is not None:
        z = np.load(Path(QDIR) / f"L{i}.npz")
        for k, n in QMAP.items():
            if k in W:
                W[k], W[k + "_s"] = qmat(z, n)
        for r in ("rot_in", "rot_mid", "rot_mix_in", "rot_mix_out"):
            if r in z and int(z[r]) >= 0: W[r] = int(z[r])
    return W


def _layer_weights(i):
    pre = f"model.language_model.layers.{i}."
    def g(n):
        t = tensor(pre + n)
        return t.to(torch.float16).numpy() if t.ndim == 2 else t.float().numpy()
    W = dict(q=g("self_attn.q_proj.weight"), k=g("self_attn.k_proj.weight"), o=g("self_attn.o_proj.weight"),
             gate=g("mlp.gate_proj.weight"), up=g("mlp.up_proj.weight"), down=g("mlp.down_proj.weight"),
             qn=g("self_attn.q_norm.weight"), kn=g("self_attn.k_norm.weight"), scalar=g("layer_scalar"),
             ln_in=g("input_layernorm.weight"), ln_post_attn=g("post_attention_layernorm.weight"),
             ln_pre_ff=g("pre_feedforward_layernorm.weight"), ln_post_ff=g("post_feedforward_layernorm.weight"))
    if lcfg(i)[0] or not CFG.get("attention_k_eq_v"):
        W["v"] = g("self_attn.v_proj.weight")
    return W


def rms_hidden(x, w):
    m = x.abs().amax(1, keepdim=True).clamp_min(1e-3)
    xs = x / m
    return xs * torch.rsqrt((xs * xs).mean(1, keepdim=True) + EPS / (m * m)) * w


def rms_last(x, w=None):
    m = x.abs().amax(-1, keepdim=True).clamp_min(1e-3)
    xs = x / m
    y = xs * torch.rsqrt((xs * xs).mean(-1, keepdim=True) + EPS / (m * m))
    return y if w is None else y * w


class SConv(nn.Module):
    """1x1 conv, optionally followed by a per-output-channel scale (weights = LUT[idx], see QDIR)."""
    def __init__(self, w, s=None):
        super().__init__()
        self.c = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=False)
        self.c.weight = nn.Parameter(torch.as_tensor(w).reshape(w.shape[0], w.shape[1], 1, 1), requires_grad=False)
        self.has_s = s is not None
        if self.has_s:
            self.register_buffer("s", torch.as_tensor(s).reshape(1, -1, 1, 1))

    def forward(self, x):
        return self.c(x) * self.s if self.has_s else self.c(x)


def conv(w, s=None):
    return SConv(w, s)


class Layer(nn.Module):
    def __init__(self, W, i, T, ctx):
        super().__init__()
        self.sliding, self.nkv, self.hd = lcfg(i)
        self.T, self.ctx, self.grp = T, ctx, nh // self.nkv
        self.keqv = "v" not in W
        g = lambda n: conv(W[n], W.get(n + "_s"))
        self.q, self.k, self.o = g("q"), g("k"), g("o")
        self.v = None if self.keqv else g("v")
        self.gate, self.up, self.down = g("gate"), g("up"), g("down")
        R = lambda key, n: Hadamard(n, W[key], hblock(n)) if key in W else None
        self.rin, self.rmid = R("rot_in", hid), R("rot_mid", inter)
        self.rmix, self.rout = R("rot_mix_in", hid), R("rot_mix_out", nh * self.hd)
        for n in ("ln_in", "ln_post_attn", "ln_pre_ff", "ln_post_ff"):
            self.register_buffer(n, torch.as_tensor(W[n]).reshape(1, -1, 1, 1))
        self.register_buffer("qn", torch.as_tensor(W["qn"]))
        self.register_buffer("kn", torch.as_tensor(W["kn"]))
        self.register_buffer("scalar", torch.as_tensor(W["scalar"]).reshape(1, 1, 1, 1))
        i_, j_ = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
        self.register_buffer("causal", torch.as_tensor(np.where(j_ <= i_, 0, NEG).astype(np.float32)), persistent=False)

    def forward(self, x, cos, sin, mask, k_st, v_st):
        T, nkv, hd, grp, ctx = self.T, self.nkv, self.hd, self.grp, self.ctx
        h = rms_hidden(x, self.ln_in)
        if self.rmix is not None: h = self.rmix(h)

        def tmajor(t, c):
            return t.reshape(c, T).transpose(0, 1)
        qh = rms_last(tmajor(self.q(h), nh * hd).reshape(T, nh, hd), self.qn)
        kraw = tmajor(self.k(h), nkv * hd).reshape(T, nkv, hd)
        kh = rms_last(kraw, self.kn)
        vh = rms_last(kraw if self.keqv else tmajor(self.v(h), nkv * hd).reshape(T, nkv, hd))
        c3, s3 = cos.reshape(T, 1, hd), sin.reshape(T, 1, hd)

        def rope(t):
            return t * c3 + torch.cat([-t[..., hd // 2:], t[..., :hd // 2]], -1) * s3
        qh = rope(qh)
        kt, vt = rope(kh).permute(1, 0, 2), vh.permute(1, 0, 2)  # (nkv, T, hd)
        q4 = qh.reshape(T, nkv, grp, hd).permute(1, 2, 0, 3).reshape(nkv, grp * T, hd)
        if KT_ATTN and not self.sliding:
            # 2026-11-25: K history pre-transposed (nkv, hd, ctx) + keys in blocks of <= 2048 with one shared max:
            # 16K attention at T=64 15.6 -> 5.8 ms on the ANE (attn_bench.py); same math as the softmax below
            n = min(ctx, KT_BLOCK); qs = KT_QSCALE; m1 = mask[:1].reshape(1, 1, ctx) * qs; TL = kt_tile(ctx); q4 = q4 * qs
            if TL:
                nt = n // TL
                scs = [q4 @ k_st[:, j // TL:j // TL + nt].permute(0, 2, 1, 3).reshape(nkv, hd, n) + m1[..., j:j + n] for j in range(0, ctx, n)]
            else:
                scs = [q4 @ k_st[..., j:j + n] + m1[..., j:j + n] for j in range(0, ctx, n)]
            sc_b = (q4 @ kt.transpose(1, 2)).reshape(nkv, grp, T, T) + self.causal * qs
            sc_b = sc_b.reshape(nkv, grp * T, T)
            mx = sc_b.amax(-1, keepdim=True)
            for sc in scs: mx = torch.maximum(mx, sc.amax(-1, keepdim=True))
            es = [torch.exp((sc - mx) * (1 / qs)) for sc in scs]; eb = torch.exp((sc_b - mx) * (1 / qs))
            den = eb.sum(-1, keepdim=True); o = eb @ vt
            for b, e in enumerate(es):
                den = den + e.sum(-1, keepdim=True); o = o + e @ v_st[:, b * n:(b + 1) * n]
            o = o / den
        else:
            sc_h = (q4 @ k_st.transpose(1, 2)).reshape(nkv, grp, T, ctx) + mask.reshape(1, 1, T, ctx)
            sc_b = (q4 @ kt.transpose(1, 2)).reshape(nkv, grp, T, T) + self.causal
            pr = torch.softmax(torch.cat([sc_h, sc_b], -1), -1).reshape(nkv, grp * T, ctx + T)
            o = pr[:, :, :ctx] @ v_st + pr[:, :, ctx:] @ vt
        o = o.reshape(nkv, grp, T, hd).permute(2, 0, 1, 3).reshape(T, nh * hd)
        o = o.transpose(0, 1).reshape(1, nh * hd, 1, T)
        x = x + rms_hidden(self.o(o if self.rout is None else self.rout(o)), self.ln_post_attn)
        h = rms_hidden(x, self.ln_pre_ff)
        if self.rin is not None: h = self.rin(h)
        a = F.gelu(self.gate(h), approximate="tanh") * self.up(h)
        x = x + rms_hidden(self.down(a if self.rmid is None else self.rmid(a)), self.ln_post_ff)
        return x * self.scalar, kt, vt


class Chunk(nn.Module):
    def __init__(self, Ws, layers, T, ctx):
        super().__init__()
        self.ids, self.T = list(layers), T
        self.layers = nn.ModuleList(Layer(W, i, T, WIN if lcfg(i)[0] else ctx) for W, i in zip(Ws, layers))
        self.taps = [i for i in self.ids if i in TAPS and i != self.ids[-1]]
        self.has_s = any(l.sliding for l in self.layers); self.has_g = any(not l.sliding for l in self.layers)

    def input_names(self):
        n = ["x"] + (["cos_s", "sin_s", "mask_s"] if self.has_s else []) + (["cos_g", "sin_g", "mask_g"] if self.has_g else [])
        return n + [f"{s}{i}" for i in self.ids for s in ("k", "v")]

    def output_names(self):
        return ["y"] + [f"h{i}" for i in self.taps] + [f"{s}{i}" for i in self.ids for s in ("kn", "vn")]

    def forward(self, x, *a):
        it = iter(a)
        rs = (next(it), next(it), next(it)) if self.has_s else None
        rg = (next(it), next(it), next(it)) if self.has_g else None
        outs, hs = [], []
        for i, l in zip(self.ids, self.layers):
            c, s, m = rs if l.sliding else rg
            x, kt, vt = l(x, c, s, m, next(it), next(it))
            outs += [kt, vt]
            if i in self.taps: hs.append(x)
        return (x, *hs, *outs)

    def example(self):
        ex = [torch.zeros(1, hid, 1, self.T)]
        for flag, sl in ((self.has_s, True), (self.has_g, False)):
            if flag:
                l = next(l for l in self.layers if l.sliding == sl)
                ex += [torch.ones(self.T, l.hd), torch.zeros(self.T, l.hd), torch.zeros(self.T, l.ctx)]
        for l in self.layers:
            kt = KT_ATTN and not l.sliding
            TL = kt and kt_tile(l.ctx)
            ks = torch.zeros(l.nkv, l.ctx // TL, l.hd, TL) if TL else torch.zeros(l.nkv, l.hd, l.ctx) if kt else torch.zeros(l.nkv, l.ctx, l.hd)
            ex += [ks, torch.zeros(l.nkv, l.ctx, l.hd)]
        return tuple(e.to(torch.float16) for e in ex)


def quantize(prog, quant):
    if quant == "none":
        return prog
    from coreai_opt.coreai_utils.common import CompressionGranularity as CG, DType
    if quant == "lut":   # weights already hold <= 16 distinct fp16 values per tensor: exact LUT, no k-means
        from coreai_opt.coreai_utils.passes import weight_palettization as wp
        from coreai_opt.coreai_utils._utils.palettize_utils import LutParams
        orig = wp._blockwise_compress

        it = wp.IntegerType; bits = [None]

        class _IdxT:   # unsigned index width of the LUT just compressed (vector LUTs: log2 K)
            get_signed = staticmethod(it.get_signed)
            @staticmethod
            def get_unsigned(n): return it.get_unsigned(n if bits[0] is None else bits[0])

        GS = int(os.environ.get("LUT_GROUP", 0))   # >0: same per-tensor LUT repeated per group of GS output channels (ANE keeps
                                                   # grouped LUTs compact: 0.29 vs 0.40 GB wired per layer, 2026-11-24)
        def tile(lut, d):
            return lut if not GS else np.repeat(lut, d.shape[0] // GS, axis=0)

        def inj(data, mode, *a, **k):
            d = np.ascontiguousarray(np.asarray(data, f16))
            kn = KNOWN_LUTS.get(wkey(d))
            if kn is not None:
                lut, idx = kn; bits[0] = int(np.log2(lut.shape[0])); extra = (1,) * (d.ndim - 2)
                return LutParams(indices=idx.reshape(*idx.shape, *extra).astype(np.uint8), lut=tile(lut.reshape(*(1,) * d.ndim, *lut.shape), d),
                                 vector_axis=0 if lut.shape[1] > 1 else None)
            bits[0] = None
            vals = np.nonzero(np.bincount(d.view(np.uint16).ravel(), minlength=65536))[0].astype(np.uint16)
            assert len(vals) <= 2 ** LUT_NB, len(vals)
            code = np.zeros(65536, np.uint8); code[vals] = np.arange(len(vals))
            lut = np.zeros(2 ** LUT_NB, f16); lut[:len(vals)] = vals.view(f16)
            return LutParams(indices=code[d.view(np.uint16)], lut=tile(lut.reshape(*(1,) * d.ndim, 2 ** LUT_NB, 1), d), vector_axis=None)
        wp._blockwise_compress = inj; wp.IntegerType = _IdxT
        cdv = wp._is_cluster_dim_valid
        wp._is_cluster_dim_valid = lambda op, cluster_dim, channel_axis: list(op.result.type.shape)[channel_axis] % cluster_dim == 0
        try:
            return wp.palettize_weights(prog, lut_dtype=None, n_bits=LUT_NB, granularity=CG.PER_GROUPED_CHANNEL if GS else CG.PER_TENSOR,
                                        group_size=GS or 32, weight_num_threshold=100000, enable_fast_kmeans_mode=False)
        finally:
            wp._blockwise_compress = orig; wp.IntegerType = it; wp._is_cluster_dim_valid = cdv
    if quant.startswith("pal"):   # palg<gs> | palt : 4-bit k-means LUT per group of <gs> output channels / per tensor
        from coreai_opt.coreai_utils.passes.weight_palettization import palettize_weights
        if quant == "palt":
            return palettize_weights(prog, lut_dtype=None, n_bits=4, granularity=CG.PER_TENSOR, weight_num_threshold=100000,
                                     enable_fast_kmeans_mode=True, num_kmeans_workers=8)
        return palettize_weights(prog, lut_dtype=None, n_bits=4, granularity=CG.PER_GROUPED_CHANNEL, group_size=int(quant[4:]),
                                 weight_num_threshold=100000, enable_fast_kmeans_mode=True, num_kmeans_workers=8)
    from coreai_opt.coreai_utils.passes.weight_quantization import quantize_weights
    return quantize_weights(prog, dtype={"int8c": DType.INT8, "int4c": DType.INT4}[quant], granularity=CG.PER_CHANNEL, weight_num_threshold=100000)


def export(module_for_T, Ts, out, quant):
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision
    cv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    base = None
    for T in Ts:
        m = module_for_T(T, base)
        base = base or m
        ep = torch.export.export(m, m.example(), strict=False).run_decompositions(coreai_torch.get_decomp_table())
        cast_to_16_bit_precision(ep)
        cv.add_exported_program(ep, input_names=m.input_names(), output_names=m.output_names(), entrypoint_name=f"v{T}")
    prog = quantize(cv.to_coreai(), quant)   # compress before optimize()
    prog.optimize()
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    return sum(f.stat().st_size for f in Path(out).rglob("*") if f.is_file()) / 1e6


def build_chunk(out_dir, layers, ctx, quant, Ts=None):
    Ts = Ts or tuple(int(t) for t in os.environ.get("TS", "8,1").split(","))
    t0 = time.time()
    Ws = [layer_weights(i) for i in layers]

    # CTX_TIERS=4096[,8192]: extra entries v{T}c{tier} whose global layers attend over only <tier> KV rows (same weights)
    tiers = [int(c) for c in os.environ.get("CTX_TIERS", "").split(",") if c]
    tts = [int(t) for t in os.environ.get("CTX_TIER_TS", "").split(",") if t] or list(Ts)
    Ts = tuple(Ts) + tuple(f"{t}c{c}" for c in tiers for t in tts)
    if os.environ.get("TS_ORDER"): Ts = tuple(x if "c" in x else int(x) for x in os.environ["TS_ORDER"].split(","))   # explicit export order
    elif os.environ.get("TIER_FIRST"): Ts = Ts[len(Ts) - len([1 for c in tiers for t in tts]):] + Ts[:len(Ts) - len([1 for c in tiers for t in tts])]

    def mk(T, base):
        t, c = (int(a) for a in T.split("c")) if isinstance(T, str) else (T, ctx)
        m = Chunk(Ws, layers, t, c).eval().to(torch.float16)
        if base is not None:
            m.load_state_dict(base.state_dict(), assign=True)
        return m
    name = f"chunk_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
    global LUT_NB
    if QDIR is not None:
        n = max(len(v) for i in layers for k, v in np.load(Path(QDIR) / f"L{i}.npz").items() if k.endswith(".lut"))
        LUT_NB = next(b for b in (4, 6, 8) if n <= 2 ** b); print(f"{name}: LUT {LUT_NB} bit", flush=True)
    mb = export(mk, Ts, Path(out_dir) / name, quant)
    LUT_NB = 4
    print(f"{name}: {mb:.0f} MB ({time.time() - t0:.0f}s)", flush=True)
    return name


class Head(nn.Module):
    """final RMSNorm + tied-embedding logits in slabs of <= 65536 output channels; returns top-1 ids and logits."""
    SLAB = 32768

    def __init__(self, E, nw, T, S=None):
        super().__init__()
        self.T = T
        self.register_buffer("nw", torch.as_tensor(nw).reshape(1, -1, 1, 1))
        self.convs = nn.ModuleList(conv(E[a:a + self.SLAB], None if S is None else S[a:a + self.SLAB]) for a in range(0, VOCAB, self.SLAB))

    def input_names(self): return ["x"]
    def output_names(self): return ["logits"]
    def example(self): return (torch.zeros(1, hid, 1, self.T, dtype=torch.float16),)

    def forward(self, x):
        h = rms_hidden(x, self.nw)
        return torch.cat([c(h) for c in self.convs], 1).reshape(VOCAB, self.T)


def build_head(out_dir, quant, Ts=None):
    Ts = Ts or tuple(int(t) for t in os.environ.get("TS", "8,1").split(","))
    t0 = time.time()
    E = tensor("model.language_model.embed_tokens.weight").float().numpy()
    nw = tensor("model.language_model.norm.weight").float().numpy()
    np.save(Path(out_dir) / "embed_f16.npy", (E * hid ** 0.5).astype(f16))
    np.save(Path(out_dir) / "final_norm.npy", nw.astype(np.float32))   # MTP drafter input norm (runtime needs no checkpoint)
    S = None
    if QDIR is not None:
        z = np.load(Path(QDIR) / "head.npz"); E, S = z["lut"].astype(f16)[z["idx"]], z["scale"].astype(f16)

    def mk(T, base):
        m = Head(E, nw, T, S).eval().to(torch.float16)
        if base is not None:
            m.load_state_dict(base.state_dict(), assign=True)
        return m
    mb = export(mk, Ts, Path(out_dir) / "head.aimodel", quant)
    print(f"head.aimodel: {mb:.0f} MB ({time.time() - t0:.0f}s)", flush=True)


def inv_freqs():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    r = Gemma4TextRotaryEmbedding(Gemma4TextConfig(**CFG))
    return {"s": r.sliding_attention_inv_freq.numpy().astype(np.float64), "g": r.full_attention_inv_freq.numpy().astype(np.float64)}


# ---------------------------------------------------------------- runtime
def wired_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * page / 2**30


class GemmaANE:
    def __init__(self, d):
        self.d = Path(d); self.man = json.loads((self.d / "manifest.json").read_text())
        self.ctx = self.man["ctx"]

    async def load(self, unit="ane"):
        from coreai.runtime import AIModel, NDArray, ComputeUnitKind, SpecializationOptions
        from coreai.runtime._ndarray import StorageKind
        self.ND = NDArray
        kind = ComputeUnitKind.neural_engine() if unit == "ane" else ComputeUnitKind.gpu()
        opt = SpecializationOptions.from_preferred_compute_unit_kind(kind)

        def buf(shape):
            nd = NDArray(np.zeros(shape, f16), StorageKind.IO_SURFACE)
            mv = memoryview(nd._tensor)
            ptr = np.frombuffer(mv, dtype=np.uint8).__array_interface__["data"][0]
            return nd, np.ctypeslib.as_array((ctypes.c_uint8 * mv.nbytes).from_address(ptr)).view(f16).reshape(shape)
        self.buf = buf
        self.models, self.fns = [], []
        w0 = wired_gb()
        for c in self.man["chunks"]:
            t0 = time.time()
            m = await AIModel.load(self.d / c["path"], specialization_options=opt)
            self.models.append(m); self.fns.append({T: m.load_function(f"v{T}") for T in self.man["Ts"]})
            print(f"  loaded {c['path']} {time.time() - t0:.1f}s wired +{wired_gb() - w0:.2f} GB", flush=True)
        self.head = None
        if not (self.d / "head.aimodel").exists():
            self.inv = {k: np.array(v) for k, v in self.man["inv_freq"].items()}
            self.emb = None
            self.reset()
            return
        t0 = time.time()
        self.head_m = await AIModel.load(self.d / "head.aimodel", specialization_options=opt)
        self.head = {T: self.head_m.load_function(f"v{T}") for T in self.man["Ts"]}
        print(f"  loaded head {time.time() - t0:.1f}s wired +{wired_gb() - w0:.2f} GB", flush=True)
        self.emb = np.load(self.d / "embed_f16.npy", mmap_mode="r")
        self.inv = {k: np.array(v) for k, v in self.man["inv_freq"].items()}
        self.reset()

    def reset(self):
        self.pos = 0
        self.kv = {}
        for c in self.man["chunks"]:
            for i in c["layers"]:
                sl, nkv, hd = lcfg(i)
                cap = WIN if sl else self.ctx
                self.kv[i] = (self.buf((nkv, cap, hd)), self.buf((nkv, cap, hd)))
        self.rowpos = np.full(WIN, -10**9)
        if not hasattr(self, "io"):
            self.io = {}
            for T, f in self.fns[0].items():
                self.io[T] = {}
                for nm in ("x", "cos_s", "sin_s", "mask_s", "cos_g", "sin_g", "mask_g"):
                    shp = {"x": (1, hid, 1, T), "mask_s": (T, WIN), "mask_g": (T, self.ctx)}.get(nm)
                    if shp is None:
                        shp = (T, lcfg(0 if nm.endswith("_s") else 5)[2])
                    self.io[T][nm] = self.buf(shp)

    async def step(self, ids, want_logits=True, embeds=None):
        """Process len(ids) (<= 8) new tokens; returns logits (n, VOCAB) float32 (or None)."""
        n = len(ids); T = 1 if n == 1 else 8
        assert self.pos + n <= self.ctx
        ND = self.ND
        io = self.io[T]   # persistent IOSurface inputs written in place (fresh NDArrays per call leak IOSurfaces)
        x = np.zeros((T, hid), f16); x[:n] = self.emb[np.asarray(ids)] if embeds is None else embeds
        p = self.pos + np.arange(T)
        io["x"][1][:] = np.ascontiguousarray(x.T).reshape(1, hid, 1, T)
        for key in ("s", "g"):
            fr = np.outer(p, self.inv[key]); emb = np.concatenate([fr, fr], -1)
            io[f"cos_{key}"][1][:] = np.cos(emb).reshape(io[f"cos_{key}"][1].shape)
            io[f"sin_{key}"][1][:] = np.sin(emb).reshape(io[f"sin_{key}"][1].shape)
        d = p[:, None] - self.rowpos[None, :]
        io["mask_s"][1][:] = np.where((d > 0) & (d < WIN), 0, NEG).reshape(io["mask_s"][1].shape)
        io["mask_g"][1][:] = np.where(np.arange(self.ctx)[None, :] < self.pos, 0, NEG).repeat(T, 0).reshape(io["mask_g"][1].shape)
        feeds = {k: v[0] for k, v in io.items()}
        new = {}
        for c, fn in zip(self.man["chunks"], self.fns):
            f = fn[T]
            ins = {k: v for k, v in feeds.items() if k in f.desc.input_names}
            for i in c["layers"]:
                ins[f"k{i}"], ins[f"v{i}"] = self.kv[i][0][0], self.kv[i][1][0]
            out = await f(inputs=ins)
            io["x"][1][:] = out["y"].numpy().reshape(io["x"][1].shape)
            for i in c["layers"]:
                new[i] = (out[f"kn{i}"], out[f"vn{i}"])
        self.last_hidden = io["x"][1].copy()
        logits = None
        if want_logits and self.head is not None:
            logits = (await self.head[T](inputs={"x": feeds["x"]}))["logits"].numpy().astype(np.float32).T[:n]
        for i, (kn, vn) in new.items():   # commit rows of the real tokens
            sl = lcfg(i)[0]
            rows = (p[:n] % WIN) if sl else p[:n]
            self.kv[i][0][1][:, rows] = kn.numpy()[:, :n]
            self.kv[i][1][1][:, rows] = vn.numpy()[:, :n]
        self.rowpos[p[:n] % WIN] = p[:n]
        self.pos += n
        return logits


def parse_layers(s):
    a, b = s.split("-"); return list(range(int(a), int(b) + 1))


if __name__ == "__main__":
    args = sys.argv[1:]
    opt = {args[j][2:]: args[j + 1] for j in range(len(args)) if args[j].startswith("--")}
    out = Path(args[1]); out.mkdir(parents=True, exist_ok=True)
    quant, ctx, cs = opt.get("quant", "none"), int(opt.get("ctx", 4096)), int(opt.get("chunk", 6))
    if "qdir" in opt:
        QDIR = os.path.expanduser(opt["qdir"])
    if args[0] == "build":
        layers = parse_layers(opt.get("layers", f"0-{NL - 1}"))
        mp = out / "manifest.json"
        man = json.loads(mp.read_text()) if mp.exists() else {"ctx": ctx, "Ts": [int(t) for t in os.environ.get("TS", "8,1").split(",")], "quant": quant, "chunks": [], "inv_freq": {k: v.tolist() for k, v in inv_freqs().items()}}
        assert man["ctx"] == ctx
        for a in range(0, len(layers), cs):
            ls = layers[a:a + cs]
            if any(c["layers"] == ls for c in man["chunks"]):
                continue
            name = build_chunk(out, ls, ctx, quant)
            man["chunks"].append({"path": name, "layers": ls, "taps": [i for i in ls if i in TAPS and i != ls[-1]]}); man["chunks"].sort(key=lambda c: c["layers"][0])
            mp.write_text(json.dumps(man))
            gc.collect()
    elif args[0] == "head":
        build_head(out, quant)
