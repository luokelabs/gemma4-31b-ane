"""Gemma 4 MTP drafter on the GPU (MLX, mlx_vlm's Gemma4AssistantDraftModel, 4-bit QAT assistant) fed by the ANE target.
draft(tok, h, pos, kv, n): tok = last emitted token (not yet in the target), h = normed target hidden that predicted it,
pos = its position, kv = GemmaANE.shared_kv() (last sliding / last global layer K,V, (nkv, S, hd) each). q_len 1: no mask."""
import os, sys, time, numpy as np
if os.environ.get("MLX_SITE"): sys.path.append(os.environ["MLX_SITE"])   # optional: mlx_vlm from another venv (same mlx version)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import MTP_MLX
import mlx.core as mx
from mlx.utils import tree_map
from mlx_vlm.speculative.drafters import load_drafter
if os.environ.get("MLX_CACHE"):
    (getattr(mx, "set_cache_limit", None) or mx.metal.set_cache_limit)(int(os.environ["MLX_CACHE"]))
PATH = str(MTP_MLX)


def _mx(a):
    """numpy (nkv, S, hd) -> mx; a transposed view of (nkv, hd, S) storage (KT_ATTN builds) is transposed on the GPU."""
    if hasattr(a, "tiles"):   # gemma_rt.TiledK: (nkv, nt, hd, TL) tiles -> (nkv, n, hd)
        t = mx.array(np.ascontiguousarray(a.tiles, np.float16)); nk, nt, hd, tl = t.shape
        return mx.swapaxes(t, 2, 3).reshape(nk, nt * tl, hd)[:, :a.n]
    if a.ndim == 3 and a.strides[1] == a.itemsize and not a.flags.c_contiguous:
        return mx.swapaxes(mx.array(np.ascontiguousarray(a.transpose(0, 2, 1), np.float16)), 1, 2)
    return mx.array(np.ascontiguousarray(a, np.float16))


class MTPMLX:
    def __init__(self, emb):
        m, _ = load_drafter(PATH, kind="mtp")
        m.update(tree_map(lambda x: x.astype(mx.float16) if x.dtype == mx.bfloat16 else x, m.parameters()))
        mx.eval(m.parameters()); self.m, self.emb = m, emb; self.t_draft = 0.0
        self.warm_iters = int(os.environ.get("WARM", 0))
        if self.warm_iters:
            self._w = mx.random.normal((1024, 1024)).astype(mx.float16) * 0.03; self._x0 = mx.random.normal((1024, 1024)).astype(mx.float16)
            mx.eval(self._w, self._x0)

    def warm(self):
        if not self.warm_iters: return
        x = self._x0
        for _ in range(self.warm_iters): x = mx.tanh(x @ self._w)
        mx.async_eval(x); self._wx = x

    def _step(self, x, off, kvs):
        m = self.m; h = m.pre_projection(x)
        for lay in m.model.layers:
            h, _, _ = lay(h, mask=None, cache=None, per_layer_input=None, shared_kv=kvs[lay.layer_type], offset=off)
        h = m.model.norm(h)
        return m.post_projection(h), mx.argmax(m.model.embed_tokens.as_linear(h)[0, -1])

    def draft(self, tok, h, pos, kv, n):
        t0 = time.perf_counter()
        kvs = {t: (_mx(k)[None], _mx(v)[None]) for t, (k, v) in kv.items()}
        h = mx.array(np.asarray(h, np.float16)).reshape(1, 1, -1); off = mx.array(pos); out = []
        for _ in range(n):
            e = mx.array(np.asarray(self.emb[tok], np.float16)).reshape(1, 1, -1)
            h, t = self._step(mx.concatenate([e, h], -1), off, kvs)
            tok = int(t.item()); out.append(tok)
        self.t_draft += time.perf_counter() - t0
        return out
