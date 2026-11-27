"""Long-context prompts: wikitext-2 test text with a passcode needle at ~50% depth; lengths ~1k/4k/8k/15k tokens.
    make_long.py -> long_prompts.json  [cat, prompt, max_new]"""
import json, pyarrow.parquet as pq
import sys
from pathlib import Path
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root (paths, gemma_ane, gemma_rt)
from paths import HF_MODEL, CALIB
HERE = Path(__file__).parent / "prompts"
tok = AutoTokenizer.from_pretrained(str(HF_MODEL))
txt = "".join(pq.read_table(CALIB / "wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet")["text"].to_pylist())
ids = tok(txt, add_special_tokens=False)["input_ids"]
NEEDLE = " The secret passcode for the archive room is 48217. "
Q = ("\n\nQuestion: Somewhere in the text above there is one sentence that states a secret passcode. What is the passcode? "
     "First reply exactly in the form 'Passcode: <number>'. Then write a three-bullet summary of the text.")
out = []
for name, L in [("L1k", 1000), ("L4k", 4000), ("L8k", 8000), ("L15k", 15000)]:
    body = ids[:L]; a = tok.decode(body[: L // 2]); b = tok.decode(body[L // 2:])
    p = a + NEEDLE + b + Q
    n = len(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])
    print(name, n, "tokens"); out.append([name, p, 128])
json.dump(out, open(HERE / "long_prompts.json", "w"), ensure_ascii=False)
