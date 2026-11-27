"""Gemma 4 31B ANE runtime over the ANEMLL Forge Swift bridge (libcoreai_bridge.dylib, MIT): every input, output and
KV buffer is an IOSurface bound once, chunk k output y is chunk k+1 input, calls allocate nothing. (The coreai Python
binding allocates each output per call and never releases them: crash after ~16K outputs, ~125 decode steps.)
Same API as gemma_ane.GemmaANE (async load/step kept so callers do not change)."""
import json, os, sys, threading, time
from pathlib import Path
import numpy as np
from paths import BRIDGE
sys.path.insert(0, str(BRIDGE))
import coreai_bridge as B
from gemma_ane import lcfg, WIN, NEG, hid, wired_gb

f16 = np.float16
SHARED = ("cos_s", "sin_s", "mask_s", "cos_g", "sin_g", "mask_g")


def _kput(K, dst, kn):
    """Write kn (nkv, m, hd) as global K rows dst..dst+m into K stored as (nkv, ctx/TL, hd, TL) tiles or (nkv, hd, ctx)."""
    m = kn.shape[1]
    _f = os.environ.get("KFILL")   # timing experiments only: zero | rand
    if _f == "zero": kn = np.zeros_like(kn)
    elif _f == "rand": kn = np.random.standard_normal(kn.shape).astype(np.float16)
    elif _f == "half": kn = (kn * 0.5).astype(np.float16)
    elif _f == "shuf": kn = np.random.permutation(np.asarray(kn).ravel()).reshape(kn.shape)
    elif _f == "stat": print("Kstat", float(np.abs(kn).max()), float(np.asarray(kn, np.float32).std()), flush=True)
    if K.ndim == 4:
        TL = K.shape[3]; q = 0
        while q < m:
            j, o = divmod(dst + q, TL); mm = min(m - q, TL - o)
            K[:, j, :, o:o + mm] = kn[:, q:q + mm].transpose(0, 2, 1); q += mm
    else: K[:, :, dst:dst + m] = kn.transpose(0, 2, 1)


def _kget(K, a, b):
    """Global K rows a..b-1 as (nkv, b-a, hd) (a copy for tiles, a view otherwise)."""
    if K.ndim == 4:
        TL = K.shape[3]; j0, j1 = a // TL, -(-b // TL)
        t = K[:, j0:j1].transpose(0, 1, 3, 2).reshape(K.shape[0], -1, K.shape[2])
        return t[:, a - j0 * TL:b - j0 * TL]
    return K[:, :, a:b].transpose(0, 2, 1)


class TiledK:
    """Global K rows 0..n-1 kept in tiles (nkv, nt, hd, TL): mtp_mlx turns it into (nkv, n, hd) on the GPU."""
    def __init__(self, K, n):
        self.tiles = K[:, :-(-n // K.shape[3])]; self.n = n
    def __array__(self, dtype=None, copy=None):
        a = _kget(self.tiles, 0, self.n); return a if dtype is None else a.astype(dtype)


class GemmaANE:
    def __init__(self, d):
        self.d = Path(d); self.man = json.loads((self.d / "manifest.json").read_text())
        self.ctx = self.man["ctx"]; self.Ts = self.man["Ts"]
        self.kT = bool(self.man.get("kT"))   # global K history stored transposed (nkv, hd, ctx) (KT_ATTN build)

    def _kt(self, i): return self.kT and not lcfg(i)[0]

    def kv_get(self, i, rows):
        """(K, V) rows of layer i as (nkv, len(rows), hd) whatever the storage layout (global rows: consecutive)."""
        k, v = self.kv[i]
        if not self._kt(i): return k.np[:, rows], v.np[:, rows]
        r = np.arange(self.cap)[rows] if isinstance(rows, slice) else np.asarray(rows)
        a, b = (int(r[0]), int(r[-1]) + 1) if len(r) else (0, 0)
        return _kget(k.np, a, b), v.np[:, a:b]

    def _put(self, i, p, kn, vn, n):
        """Commit rows 0..n-1 of the new K/V outputs for positions p[:n] (consecutive) with slice copies."""
        a = int(p[0]); K, Vb = self.kv[i]; kt = self._kt(i)
        if lcfg(i)[0]:
            s = a % WIN; parts = [(s, 0, min(n, WIN - s))]
            if s + n > WIN: parts.append((0, WIN - s, n - (WIN - s)))
        else: parts = [(a, 0, n)]
        for dst, src, m in parts:
            if kt: _kput(K.np, dst, kn.np[:, src:src + m])
            else: K.np[:, dst:dst + m] = kn.np[:, src:src + m]
            Vb.np[:, dst:dst + m] = vn.np[:, src:src + m]

    def kv_set(self, i, rows, k, v):
        if self._kt(i):
            a = int(np.asarray(rows).ravel()[0]) if np.size(rows) else 0
            _kput(self.kv[i][0].np, a, k); self.kv[i][1].np[:, a:a + k.shape[1]] = v
        else:
            self.kv[i][0].np[:, rows] = k; self.kv[i][1].np[:, rows] = v

    async def load(self, unit="ane", verbose=True):
        w0 = wired_gb(); self.models, self.fns = [], []
        # context tiers (2026-11-25): entries v{T}c{tier} attend over only <tier> global KV rows; the runtime starts in
        # the smallest tier and copies the global KV rows into the next tier's buffers when the context outgrows it
        self.tiers = sorted(self.man.get("tiers", [])) + [self.ctx]
        nm = lambda T, tier: f"v{T}" if tier == self.ctx else f"v{T}c{tier}"
        for c in self.man["chunks"]:
            t0 = time.time(); m = B.Model(self.d / c["path"], compute=unit)
            self.models.append(m)
            have = set(m.function_names)
            self.fns.append({(T, tier): m.function(nm(T, tier)) for tier in self.tiers for T in c.get("Ts", self.Ts) if nm(T, tier) in have})
            time.sleep(float(os.environ.get("LOAD_SLEEP", 0)))
            if verbose: print(f"  loaded {c['path']} {time.time() - t0:.1f}s wired +{wired_gb() - w0:.2f} GB", flush=True)
        Tsets = [set(c.get("Ts", self.Ts)) for c in self.man["chunks"]]   # entries of the full-context tier
        full = lambda tier: [T for T in self.Ts if all((T, tier) in fn for fn in self.fns)]
        self.full_Ts = full(self.ctx)
        self.head = None
        if (self.d / "head.aimodel").exists():
            self.head_m = B.Model(self.d / "head.aimodel", compute=unit)
            self.head = {T: self.head_m.function(f"v{T}") for T in sorted({T for t in self.tiers for T in full(t)})}
            self.emb = np.load(self.d / "embed_f16.npy", mmap_mode="r")
        self.inv = {k: np.array(v) for k, v in self.man["inv_freq"].items()}
        self._ane = threading.Lock(); self._ane_t = time.perf_counter(); self._ping = None; self._ka_thread = None; self._ka_ms = 0
        if os.environ.get("KEEPALIVE_MS"): self.keepalive(float(os.environ["KEEPALIVE_MS"]))
        self.feat_layers = self.man.get("feat_layers", [])
        self.S = {}; slide = {}
        for tier in self.tiers:
            st = {k: {} for k in ("kv", "io", "new", "plan", "logits", "ys", "hb", "binds")}
            st["full_Ts"] = full(tier); Tsets = [{T for (T, t) in fn if t == tier} for fn in self.fns]
            for c, fn, ts in zip(self.man["chunks"], self.fns, Tsets):
                f = fn[(8 if 8 in ts else min(ts), tier)]
                for i in c["layers"]:
                    if lcfg(i)[0]:   # sliding ring: same shape in every tier, one shared buffer
                        if i not in slide: slide[i] = (f.buffer("input", f"k{i}"), f.buffer("input", f"v{i}"))
                        st["kv"][i] = slide[i]
                    else:
                        st["kv"][i] = (f.buffer("input", f"k{i}"), f.buffer("input", f"v{i}"))
            for T in [T for T in self.Ts if any(T in ts for ts in Tsets)]:
                io = {}; binds = []; new = {}; ys = []; hb = {}
                x = next(fn[(T, tier)] for fn in self.fns if (T, tier) in fn).buffer("input", "x"); io["x"] = x
                for c, fn in zip(self.man["chunks"], self.fns):
                    if (T, tier) not in fn: continue
                    f = fn[(T, tier)]; ins = {"x": x}
                    for n in f.input_names:
                        if n in SHARED:
                            if n not in io: io[n] = f.buffer("input", n)
                            ins[n] = io[n]
                    outs = {"y": f.buffer("output", "y")}
                    for i in c.get("taps", []):
                        outs[f"h{i}"] = hb[i] = f.buffer("output", f"h{i}")
                    hb[c["layers"][-1]] = outs["y"]
                    for i in c["layers"]:
                        ins[f"k{i}"], ins[f"v{i}"] = st["kv"][i]
                        new[i] = (f.buffer("output", f"kn{i}"), f.buffer("output", f"vn{i}"))
                        outs[f"kn{i}"], outs[f"vn{i}"] = new[i]
                    binds.append(f.bind(ins, outs)); x = outs["y"]; ys.append(x)
                st["io"][T], st["new"][T], st["ys"][T], st["hb"][T] = io, new, ys, hb
                st["plan"][T] = {False: B.Plan(binds)}; st["binds"][T] = binds
                if self.head and T in self.head:
                    h = self.head[T]; st["logits"][T] = h.buffer("output", "logits")
                    st["plan"][T][True] = B.Plan(binds + [h.bind({"x": x}, {"logits": st["logits"][T]})])
            # mixed build: leading chunks only up to 32 rows, the rest also has v64 -> prefill64() runs the front twice
            st["split"] = st["plan_front"] = None
            part = [T for T in self.Ts if T not in st["full_Ts"] and any(T in ts for ts in Tsets)]
            k = sum(64 not in ts for ts in Tsets)
            if 64 in part and all(32 in ts for ts in Tsets[:k]) and k < len(Tsets):   # front chunks v32, back chunks v64 (8K tier: nothing else)
                assert all(64 not in ts for ts in Tsets[:k]), "v64 chunks must be the trailing ones"
                st["split"] = k; st["plan_front"] = B.Plan(st["binds"][32][:k])
            self.S[tier] = st
        if os.environ.get("MLOCK", "0") != "0": self._mlock()
        if verbose: print(f"  bound (tiers {self.tiers}); wired +{wired_gb() - w0:.2f} GB", flush=True)
        self.reset()

    def _mlock(self):
        """Pin every KV and I/O buffer (mlock): keeps the kernel from compressing pages the ANE reads next call."""
        import ctypes
        libc = ctypes.CDLL(None); libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]; seen = set(); tot = 0; bad = 0
        for st in self.S.values():
            bufs = [b for bs in st["kv"].values() for b in bs] + [b for io in st["io"].values() for b in io.values()]
            for b in bufs:
                a = b.np; p = a.ctypes.data
                if p in seen: continue
                seen.add(p); r = libc.mlock(ctypes.c_void_p(p), ctypes.c_size_t(a.nbytes)); tot += a.nbytes; bad += r != 0
        print(f"  mlock {tot / 1e9:.2f} GB in {len(seen)} buffers, failures {bad}", flush=True)

    def _use(self, tier):
        self.tier = self.cap = tier
        for k, v in self.S[tier].items(): setattr(self, k, v)

    def _ensure(self, need, step=False):
        """Switch to the smallest tier holding <need> positions, carrying the committed global KV rows over.
        step=True: the tier must have entries every chunk shares (prefill-only tiers such as the 8K one are skipped)."""
        ok = lambda t: (self.S[t]["full_Ts"] if step else (self.S[t]["split"] is not None or self.S[t]["full_Ts"]))
        if need <= self.cap and ok(self.tier): return
        old = self.kv; self._use(min(t for t in self.tiers if t >= need and ok(t)))
        for i, bs in self.kv.items():
            if lcfg(i)[0]: continue
            if self.kT: _kput(bs[0].np, 0, _kget(old[i][0].np, 0, self.pos))   # layouts may differ per tier
            else: bs[0].np[:, :self.pos] = old[i][0].np[:, :self.pos]
            bs[1].np[:, :self.pos] = old[i][1].np[:, :self.pos]

    def ping(self):
        """Keep-alive call: chunk 0, smallest entry of the smallest tier, on private input/output buffers (touches no
        runtime state). The first ANE call after >=0.3 s idle often costs +300-600 ms; a ping every ~100 ms while idle
        avoids most of that (2026-11-25, prof_switch K=8 GAP=1: first-block stalls 16/24 -> 1/24)."""
        if self._ping is None:
            t0 = self.tiers[0]; T = min(T for (T, t) in self.fns[0] if t == t0); f = self.fns[0][(T, t0)]
            ins = {n: f.buffer("input", n) for n in f.input_names}
            outs = {n: f.buffer("output", n) for n in f.output_names}
            self._ping = B.Plan([f.bind(ins, outs)])
        with self._ane:
            self._ping.run()

    def keepalive(self, ms):
        """Background thread: ping when the ANE has been idle for >= ms (0 = off)."""
        self._ka_ms = ms / 1e3
        if ms and self._ka_thread is None:
            def loop():
                while True:
                    time.sleep(self._ka_ms or 0.5)
                    if self._ka_ms and time.perf_counter() - self._ane_t >= self._ka_ms: self.ping()
            self._ka_thread = threading.Thread(target=loop, daemon=True); self._ka_thread.start()

    def _run(self, plan, times=False):
        with self._ane:
            r = plan.run(times)
        self._ane_t = time.perf_counter()
        return r

    def reset(self):
        self._use(self.tiers[0])
        self.pos = 0; self.hist = []   # hist: token ids whose KV rows are committed (prefix reuse)
        self.rowpos = np.full(WIN, -10**9)   # stale rows are masked, no need to zero the buffers

    async def step(self, ids, want_logits=True, embeds=None, times=False, commit=None):
        """Process len(ids) (<= max(Ts)) new tokens; returns logits (n, VOCAB) float32 (or None).
        commit=None: the KV rows of all n tokens are committed. commit=False: nothing is committed (speculative
        verify); call commit_rows(m) afterwards to keep the first m tokens. self.last_hidden: (n, hid) pre-norm."""
        n = len(ids); assert self.pos + n <= self.ctx
        self._ensure(self.pos + n, step=True)
        if n > max(self.full_Ts):   # e.g. a 32-token prompt tail in a tier whose largest shared entry is v8
            B = max(self.full_Ts); assert commit is None
            for a in range(0, n - B, B): await self.step(ids[a:a + B], want_logits=False, embeds=None if embeds is None else embeds[a:a + B])
            a = (n - 1) // B * B
            return await self.step(ids[a:], want_logits, None if embeds is None else embeds[a:], times)
        T = min(t for t in self.full_Ts if t >= n)   # smallest entry that fits (v1 / v8 decode, v32 prefill)
        io = self.io[T]
        x = np.zeros((T, hid), f16); x[:n] = self.emb[np.asarray(ids)] if embeds is None else embeds
        p = self.pos + np.arange(T)
        io["x"].np[...] = x.T.reshape(1, hid, 1, T)
        for key in ("s", "g"):
            if f"cos_{key}" not in io: continue
            fr = np.outer(p, self.inv[key]); emb = np.concatenate([fr, fr], -1)
            io[f"cos_{key}"].np[...] = np.cos(emb); io[f"sin_{key}"].np[...] = np.sin(emb)
        d = p[:, None] - self.rowpos[None, :]
        if "mask_s" in io: io["mask_s"].np[...] = np.where((d > 0) & (d < WIN), 0, NEG)
        if "mask_g" in io: self._mask_g(io, T, self.pos)
        wl = bool(want_logits and self.head)
        self.times = self._run(self.plan[T][wl], times)
        self.last_hidden = self.ys[T][-1].np.reshape(hid, T).T[:n].astype(np.float32)
        self._pend = (T, p, n); self._pend_ids = list(ids) if embeds is None else [None] * n
        if commit is False:
            return self.logits[T].np.astype(np.float32).T[:n].copy() if wl else None
        self.commit_rows(n)
        return self.logits[T].np.astype(np.float32).T[:n].copy() if wl else None

    def features(self, k):
        """(k, len(feat_layers)*hid) fp16: drafter features (outputs of feat_layers) of the first k rows of the last step."""
        T = self._pend[0]
        return np.concatenate([self.hb[T][l].np.reshape(hid, T).T[:k] for l in self.feat_layers], -1)

    def commit_rows(self, n):
        T, p, _ = self._pend
        for i, (kn, vn) in self.new[T].items():   # commit the rows of the real tokens
            self._put(i, p, kn, vn, n)
        self.rowpos[p[:n] % WIN] = p[:n]
        self.pos += n; self.hist += self._pend_ids[:n]

    def final_norm(self, h):
        """Final RMSNorm (host side, fp32) of pre-norm hidden rows: the drafter input."""
        if not hasattr(self, "_nw"):
            if (self.d / "final_norm.npy").exists(): self._nw = np.load(self.d / "final_norm.npy").astype(np.float32)
            else:   # older model dirs: read it from the original checkpoint
                from gemma_ane import tensor
                self._nw = tensor("model.language_model.norm.weight").float().numpy()
        h = h.astype(np.float32)
        return h / np.sqrt((h * h).mean(-1, keepdims=True) + 1e-6) * self._nw

    def shared_kv(self):
        """K/V rows of the last sliding / last global layer (what the MTP drafter attends to), oldest first."""
        ls = max(i for i in self.kv if lcfg(i)[0]); lg = max(i for i in self.kv if not lcfg(i)[0])
        ps = np.arange(max(0, self.pos - WIN), self.pos)
        return {"sliding_attention": tuple(b.np[:, ps % WIN] for b in self.kv[ls]),
                "full_attention": (TiledK(self.kv[lg][0].np, self.pos), self.kv[lg][1].np[:, :self.pos])
                                  if self.kv[lg][0].np.ndim == 4 else self.kv_get(lg, slice(0, self.pos))}

    # ---- prefix KV reuse (2026-11-24) ----
    def rollback(self, L):
        """Forget positions >= L. Possible when no sliding-window row a later query needs was overwritten:
        pos <= WIN (ring never wrapped) or pos - L <= 1. Returns False when not possible (caller re-prefills)."""
        if L >= self.pos: return L == self.pos
        if not (self.pos <= WIN or self.pos - L <= 1): return False
        self.rowpos[self.rowpos >= L] = -10**9   # stale sliding rows masked; global rows beyond pos are masked by mask_g
        self.pos = L; self.hist = self.hist[:L]; return True

    def reuse(self, ids, max_skip=8, check=16):
        """Reuse the committed KV for ids. Returns the index into ids from which tokens still have to be run.
        1. align the KV history with ids, allowing blocks of <= max_skip extra tokens in the KV that ids leave out
           (Gemma 4 puts an empty thought block in the generation prompt but drops it from history turns); if the
           whole history aligns, keep the KV (extra tokens included) and continue after it;
        2. else longest common prefix with rollback if allowed; 3. else start fresh."""
        h = self.hist; i = j = 0; self.skipped = 0
        while i < len(h) and j < len(ids):
            if h[i] == ids[j]: i += 1; j += 1; continue
            for k in range(1, max_skip + 1):
                n = min(check, len(h) - i - k, len(ids) - j)
                if n > 0 and h[i + k:i + k + n] == ids[j:j + n]: i += k; self.skipped += k; break
            else: break
        if i == len(h) and j < len(ids): return j
        L = 0
        for a, b in zip(h, ids):
            if a != b: break
            L += 1
        L = min(L, len(ids) - 1); self.skipped = 0
        if not self.rollback(L): self.reset(); L = 0
        return L

    def _snap_rows(self, i, P):
        ps = np.arange(P); return (ps[-WIN:] % WIN) if lcfg(i)[0] else ps

    def snapshot(self, path):
        """Save the KV rows of positions [0, pos) + hist to <path>: JSON header line, then raw fp16 K, V per layer.
        Streams one layer at a time and bypasses the page cache (F_NOCACHE): the 32 GB box has ~2.5 GB to spare."""
        import fcntl, json as J
        with open(path, "wb") as f:
            fcntl.fcntl(f.fileno(), 48, 1)   # F_NOCACHE
            f.write((J.dumps({"hist": self.hist}) + "\n").encode())
            for i in sorted(self.kv):
                r = self._snap_rows(i, self.pos)
                for a in self.kv_get(i, r): f.write(np.ascontiguousarray(a).tobytes())
            f.flush(); os.fsync(f.fileno())   # no dirty pages left behind: they are not reclaimable until written

    def restore(self, path):
        """Load a snapshot into the KV buffers (one layer at a time, no page cache); afterwards pos = snapshot length."""
        import fcntl, json as J
        self.reset()
        with open(path, "rb") as f:
            fcntl.fcntl(f.fileno(), 48, 1)
            hist = J.loads(f.readline()); hist = hist["hist"]; P = len(hist)
            self._ensure(P)
            for i in sorted(self.kv):
                r = self._snap_rows(i, P)
                shp = self.kv[i][1].np.shape; shp = (shp[0], len(r), shp[2])
                kv = [np.frombuffer(f.read(int(np.prod(shp)) * 2), f16).reshape(shp) for _ in range(2)]
                self.kv_set(i, r, *kv)
        ps = np.arange(P); self.rowpos[ps[-WIN:] % WIN] = ps[-WIN:]
        self.pos = P; self.hist = hist
        return P

    # ---- mixed prefill (2026-11-24): front chunks v32 twice, back chunks v64 once ----
    def _inputs(self, T, p, x=None, rowpos=None, pos=None):
        io = self.io[T]; rowpos = self.rowpos if rowpos is None else rowpos; pos = self.pos if pos is None else pos
        if x is not None: io["x"].np[...] = x.T.reshape(1, hid, 1, T)
        for key in ("s", "g"):
            if f"cos_{key}" not in io: continue
            fr = np.outer(p, self.inv[key]); emb = np.concatenate([fr, fr], -1)
            io[f"cos_{key}"].np[...] = np.cos(emb); io[f"sin_{key}"].np[...] = np.sin(emb)
        d = p[:, None] - rowpos[None, :]
        if "mask_s" in io: io["mask_s"].np[...] = np.where((d > 0) & (d < WIN), 0, NEG)
        if "mask_g" in io: self._mask_g(io, T, pos)

    def _mask_g(self, io, T, pos):
        row = np.where(np.arange(self.cap) < pos, 0, NEG)
        if self.kT: io["mask_g"].np[0] = row   # KT build reads row 0 only
        else: io["mask_g"].np[...] = row[None].repeat(T, 0)

    def _commit(self, T, p, layers):
        for i in layers:
            kn, vn = self.new[T][i]; self._put(i, p, kn, vn, len(p))

    def prefill64(self, ids):
        """64 prompt tokens, no logits. Front chunks (no v64) run as two v32 halves, their outputs are copied into the
        v64 input of the back chunks, which run once. Same math as two v32 steps."""
        assert len(ids) == 64 and self.split and self.pos + 64 <= self.ctx
        self._ensure(self.pos + 64)
        p0 = self.pos; rp0 = self.rowpos.copy(); k = self.split
        front = [i for c in self.man["chunks"][:k] for i in c["layers"]]
        back = [i for c in self.man["chunks"][k:] for i in c["layers"]]
        x64 = self.io[64]["x"].np.reshape(hid, 64)
        e = np.asarray(self.emb[np.asarray(ids)], f16)
        for h in range(2):
            p = p0 + 32 * h + np.arange(32)
            self._inputs(32, p, e[32 * h:32 * h + 32])
            self._run(self.plan_front)
            x64[:, 32 * h:32 * h + 32] = self.ys[32][k - 1].np.reshape(hid, 32)
            self._commit(32, p, front)
            self.rowpos[p % WIN] = p; self.pos += 32   # visible to the second half (front layers only)
        p = p0 + np.arange(64)
        self._inputs(64, p, rowpos=rp0, pos=p0)        # back layers: none of the 64 rows is history yet
        self._run(self.plan[64][False])
        self._commit(64, p, back)
        self.last_hidden = self.ys[64][-1].np.reshape(hid, 64).T.astype(np.float32)
        self.hist += list(ids)
