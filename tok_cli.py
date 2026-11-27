"""Tokenizer helper run as a separate process (keeps transformers, ~1 GB, out of the ANE runtime process).
    tok_cli.py enc <prompt>          -> JSON {"ids", "eos"} with the chat template and generation prompt
    tok_cli.py encm <messages json>  -> same for a list of chat messages
    tok_cli.py dec <ids json>        -> text
    tok_cli.py serve                 -> one JSON request per stdin line: {"op": "encm"|"dec", "arg": ...}, one JSON reply per line
Tokenizer: $GEMMA_TOKENIZER, else the original checkpoint (paths.HF_MODEL)."""
import json, os, sys
from pathlib import Path
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import HF_MODEL

tok = AutoTokenizer.from_pretrained(os.environ.get("GEMMA_TOKENIZER", str(HF_MODEL)))
EOS = [tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"), 106]


def encm(msgs):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    return {"ids": list(ids), "eos": EOS}


if sys.argv[1] == "enc":
    print(json.dumps(encm([{"role": "user", "content": sys.argv[2]}])))
elif sys.argv[1] == "encm":
    print(json.dumps(encm(json.loads(sys.argv[2]))))
elif sys.argv[1] == "serve":
    for line in sys.stdin:
        q = json.loads(line)
        r = encm(q["arg"]) if q["op"] == "encm" else {"text": tok.decode(q["arg"], skip_special_tokens=True)}
        print(json.dumps(r, ensure_ascii=False), flush=True)
else:
    print(tok.decode(json.loads(sys.argv[2])))
