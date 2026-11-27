"""Shared helpers for the 3-bit format: online block-Hadamard rotations (same construction as gemma_ane.Hadamard) and
dense reconstruction of scalar (.lut) / vector (.vlut) LUT matrices. Row-vector convention: x' = x M, W' = W M,
M = blockdiag(diag(signs) H_b) / sqrt(b); seeds per matrix: q/k/v rot_mix_in, o rot_mix_out, gate/up rot_in, down rot_mid."""
import numpy as np, torch
from scipy.linalg import hadamard

HB_HID = 256
ROT_KEY = {"q_proj": "rot_mix_in", "k_proj": "rot_mix_in", "v_proj": "rot_mix_in", "o_proj": "rot_mix_out",
           "gate_proj": "rot_in", "up_proj": "rot_in", "down_proj": "rot_mid"}
SEED0 = {"rot_in": 1000, "rot_mid": 2000, "rot_mix_in": 3000, "rot_mix_out": 4000}
_cache = {}


def _parts(n, seed, hid, device):
    b = HB_HID if n == hid else 1024
    k = (n, seed, b, str(device))
    if k not in _cache:
        signs = torch.from_numpy(np.random.default_rng(seed).choice([-1.0, 1.0], n)).float().to(device)
        h = torch.from_numpy(hadamard(b) / np.sqrt(b)).float().to(device)
        _cache[k] = (signs, h, b)
    return _cache[k]


def rot(x, seed, hid):
    """x (..., n) -> x M (also W -> W M for a weight (Cout, n))."""
    n = x.shape[-1]; s, h, b = _parts(n, seed, hid, x.device)
    return ((x.float() * s).reshape(*x.shape[:-1], n // b, b) @ h).reshape(x.shape).to(x.dtype)


def unrot(q, seed, hid):
    """Q (Cout, n) in the rotated basis -> Q M^T (acts on unrotated inputs)."""
    n = q.shape[-1]; s, h, b = _parts(n, seed, hid, q.device)
    return ((q.float().reshape(*q.shape[:-1], n // b, b) @ h.T).reshape(q.shape) * s).to(q.dtype)


def rot_h(hm, seed, hid):
    """H (n, n) -> M^T H M."""
    a = rot(hm, seed, hid)                 # H M
    return rot(a.T.contiguous(), seed, hid).T.contiguous()


def dense(z, name):
    """(lut-based weight (Cout, Cin) float32 without scale, scale (Cout,)) from an npz / dict."""
    idx = torch.from_numpy(np.asarray(z[name + ".idx"]).astype(np.int64))
    if name + ".vlut" in z:
        lut = torch.from_numpy(np.asarray(z[name + ".vlut"]).astype(np.float32))
        w = lut[idx].permute(0, 2, 1).reshape(idx.shape[0] * lut.shape[1], idx.shape[1])
    else:
        w = torch.from_numpy(np.asarray(z[name + ".lut"]).astype(np.float32))[idx]
    return w, torch.from_numpy(np.asarray(z[name + ".scale"]).astype(np.float32))


def seed_of(z, name):
    k = ROT_KEY[name]
    return int(z[k]) if k in z and int(z[k]) >= 0 else None
