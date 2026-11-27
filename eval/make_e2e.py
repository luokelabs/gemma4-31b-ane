"""Everyday-chat end-to-end prompts (~500 and ~1000 prompt tokens incl. chat template) + long-generation prompts.
-> e2e_prompts.json [cat, prompt, max_new=512], longgen_prompts.json [cat, prompt, max_new=1024]"""
import json, os
import sys
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root (paths, gemma_ane, gemma_rt)
from paths import HF_MODEL, CALIB
HERE = Path(__file__).parent / "prompts"
tok = AutoTokenizer.from_pretrained(str(HF_MODEL))
def n(p):
    r = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True)
    return len(r["input_ids"] if hasattr(r, "keys") else r)

en = "".join(pq.read_table(CALIB / "wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet")["text"].to_pylist()[200:])
zh = []
for l in open(Path(os.path.expanduser(os.environ.get("ZH_JSONL", "~/dfdata/zh.jsonl")))):
    for c in json.loads(l)["conversations"]:
        if c["from"] == "gpt" and len(c["value"]) > 200: zh.append(c["value"])
    if len(zh) > 60: break
zh = "\n\n".join(zh)
import importlib.util; GF = importlib.util.find_spec("mlx_lm").submodule_search_locations[0] + "/generate.py"
code = open(GF).read()

def fit(head, body, tail, target, unit):
    lo, hi = 0, len(body)
    while hi - lo > unit:
        mid = (lo + hi) // 2
        if n(head + body[:mid] + tail) < target: lo = mid
        else: hi = mid
    p = head + body[:lo] + tail; return p, n(p)

out = []
for target in (500, 1000):
    p, k = fit("Read the following article and summarize it in about 150 words, then list three key facts.\n\n", en, "\n\nSummary:", target, 20)
    out.append([f"en{target}", p, 512]); print("en", target, k)
    p, k = fit("请阅读下面的材料，用大约200字概括主要内容，并列出三个要点。\n\n", zh, "\n\n概括：", target, 10)
    out.append([f"zh{target}", p, 512]); print("zh", target, k)
    p, k = fit("Review the following Python code. Briefly explain what it does and point out two possible improvements.\n\n```python\n", code, "\n```", target, 40)
    out.append([f"code{target}", p, 512]); print("code", target, k)
json.dump(out, open(HERE / "e2e_prompts.json", "w"), ensure_ascii=False)

lg = [
    ["code", "Implement a complete, production-quality Python module for an LRU cache with per-entry TTL and thread safety. Include full docstrings, type hints, and a comprehensive set of unittest test cases."],
    ["code", "Write a complete Flask REST API for a todo-list app backed by SQLite: models, CRUD endpoints, input validation, error handling, and pytest tests. Explain each part briefly."],
    ["math", "Solve the following five problems step by step, showing all work: (1) the sum of the first 100 odd numbers; (2) solve x^2 - 7x + 10 = 0; (3) the probability of getting exactly 3 heads in 6 fair coin tosses; (4) the derivative of x^3 * ln(x); (5) the area between y = x^2 and y = 2x."],
    ["math", "Explain in detail, with worked examples and proofs where appropriate, the main ideas of linear algebra: vectors, matrices, linear maps, determinants, eigenvalues and eigenvectors, and diagonalization."],
    ["en", "Write a detailed 1200-word essay on the history of the printing press and its impact on European society, science and religion."],
    ["en", "Write a long short story (about 1000 words) about a lighthouse keeper who discovers a message in a bottle."],
    ["zh", "写一篇约1500字的文章，详细介绍中国古代四大发明的起源、发展以及它们对世界文明的影响。"],
    ["zh", "详细讲解机器学习中的梯度下降算法，包括基本原理、数学推导、常见变种（SGD、Momentum、Adam）以及实践中的调参技巧，越详细越好。"],
]
json.dump([[c, p, 1024] for c, p in lg], open(HERE / "longgen_prompts.json", "w"), ensure_ascii=False)
print("longgen", len(lg))
