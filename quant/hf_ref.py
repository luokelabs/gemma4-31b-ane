"""Layer-by-layer fp32 reference with the real transformers Gemma4 layer classes (streams one layer at a time).
Saves each layer's input / output for an 8-token prompt to ref/, plus cos/sin per layer type."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4 import modeling_gemma4 as G

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root (paths, gemma_ane, gemma_rt)
from paths import HF_MODEL, WORK
M = HF_MODEL
import os
OUT = WORK / ("ref" if os.environ.get("N", "8") == "8" else "ref" + os.environ["N"]); OUT.mkdir(parents=True, exist_ok=True)
import os
T = int(os.environ.get("N", 8))
cfg = Gemma4TextConfig(**json.loads((M / "config.json").read_text())["text_config"])
idx = json.loads((M / "model.safetensors.index.json").read_text())["weight_map"]
files = {}
def get(k):
    f = idx[k]
    if f not in files: files[f] = safe_open(str(M / f), "pt")
    return files[f].get_tensor(k).float()

tok = AutoTokenizer.from_pretrained(str(M))
ids = tok(os.environ.get("TEXT", "The Eiffel Tower is located in the city of Paris, the capital of France"), return_tensors="pt").input_ids; ids = (ids if ids[0,0]==2 else torch.cat([torch.tensor([[2]]), ids], 1))[:, :T]
print("ids", ids.tolist(), [tok.decode(i) for i in ids[0]])
emb = get("model.language_model.embed_tokens.weight")
x = emb[ids[0]][None] * cfg.hidden_size ** 0.5
rope = G.Gemma4TextRotaryEmbedding(cfg)
pos = torch.arange(T)[None]
pe = {lt: rope(x, pos, lt) for lt in set(cfg.layer_types)}
for lt, (c, s) in pe.items():
    np.savez(OUT / f"rope_{lt}.npz", cos=c[0].numpy(), sin=s[0].numpy())
mask = torch.full((T, T), float("-inf")).triu(1)[None, None]
stats = []
with torch.no_grad():
    for i in range(cfg.num_hidden_layers):
        t0 = time.time()
        layer = G.Gemma4TextDecoderLayer(cfg, i).eval()
        pre = f"model.language_model.layers.{i}."
        sd = {k[len(pre):]: get(k) for k in idx if k.startswith(pre)}
        missing = layer.load_state_dict(sd, strict=False)
        assert not missing.unexpected_keys and all("inv_freq" in m for m in missing.missing_keys), missing
        xin = x.clone()
        x = layer(x, position_embeddings=pe[cfg.layer_types[i]], attention_mask=mask, shared_kv_states={}, position_ids=pos)
        if T == 8 or i in (5, 29, 59): np.savez(OUT / f"L{i}.npz", x_in=xin[0].numpy(), x_out=x[0].numpy())
        stats.append((i, cfg.layer_types[i][:4], float(xin.abs().max()), float(xin.pow(2).mean().sqrt()), float(layer.layer_scalar)))
        print(f"L{i:02d} {stats[-1][1]} max|x_in| {stats[-1][2]:9.1f} rms {stats[-1][3]:8.2f} scalar {stats[-1][4]:.4f} ({time.time()-t0:.1f}s)", flush=True)
        del layer, sd
    h = G.Gemma4RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps); h.weight.data = get("model.language_model.norm.weight")
    logits = h(x)[0] @ emb.T
    logits = torch.tanh(logits / cfg.final_logit_softcapping) * cfg.final_logit_softcapping
    np.save(OUT / "logits.npy", logits.numpy()); np.save(OUT / "ids.npy", ids[0].numpy())
    for p in range(0, T, max(1, T // 8)):
        print(p, repr(tok.decode(ids[0, p])), "->", [tok.decode(j) for j in logits[p].topk(3).indices])
print("final max|x|", float(x.abs().max()))
