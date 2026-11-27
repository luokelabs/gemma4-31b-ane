"""Self-generated chat calibration / eval rows: greedy answers of the current ANE build to a prompt list.
Saves calib/chat_ids.json = list of full token rows (template + prompt + answer). Rows 0..23 calib, the rest eval."""
import asyncio, sys, json, subprocess, time
import numpy as np
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parent.parent))   # repo root
from pathlib import Path
from paths import CALIB
from gemma_rt import GemmaANE

PROMPTS = [
 "用三句话解释为什么天空是蓝色的。", "Write a Python function that merges two sorted lists, with tests.",
 "Explain how a hash map works to a beginner.", "把下面这句话翻译成英文并解释语法:今天下雨了,所以我没有去公园。",
 "What are the main causes of the French Revolution?", "写一个 SQL 查询,找出每个部门工资最高的员工。",
 "Give me a 5-step plan to learn linear algebra.", "Summarize the plot of Romeo and Juliet in one paragraph.",
 "解释一下什么是梯度下降,并给出一个简单的例子。", "Write a bash script that counts lines in all .py files of a directory.",
 "What is the difference between TCP and UDP?", "请写一封礼貌的邮件,向客户说明项目要延期一周。",
 "Solve: a train travels 300 km in 2.5 hours. What is its average speed? Show steps.", "用 JavaScript 实现一个防抖函数并解释原理。",
 "Describe the water cycle.", "列出三种常见的排序算法并比较它们的时间复杂度。",
 "Write a short poem about autumn.", "What should I consider when choosing a database for a small web app?",
 "解释量子纠缠,尽量通俗。", "Write a Rust function that reverses a string, and explain ownership in it.",
 "How do vaccines work?", "给我讲一个关于勇气的短故事。",
 "Explain the difference between a process and a thread.", "Convert this JSON to YAML and explain the differences: {\"name\": \"a\", \"tags\": [1, 2]}",
 # eval
 "为什么海水是咸的?", "Write a Python class for a stack with push, pop and peek.",
 "Explain what a neural network is in simple terms.", "请比较一下电动车和燃油车的优缺点。",
 "What is the capital of Australia, and why is it not Sydney?", "Write a C function that computes the greatest common divisor.",
]

def tok_cli(*a):
    return subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "tok_cli.py"), *a], capture_output=True, text=True, check=True).stdout

async def main(d, n=220):
    encs = [json.loads(tok_cli("enc", p).strip().splitlines()[-1]) for p in PROMPTS]
    m = GemmaANE(d); await m.load(verbose=False); rows = []
    for e in encs:
        ids, eos = e["ids"], set(e["eos"]); m.reset(); t0 = time.time()
        for a in range(0, len(ids), 8):
            lg = await m.step(ids[a:a + 8], want_logits=a + 8 >= len(ids))
        out = [int(lg[-1].argmax())]
        while len(out) < int(n) and out[-1] not in eos:
            out.append(int((await m.step([out[-1]]))[-1].argmax()))
        rows.append(ids + out); print(len(rows), len(ids), len(out), f"{time.time() - t0:.0f}s", flush=True)
        json.dump(rows, open(CALIB / "chat_ids.json", "w"))

asyncio.run(main(*sys.argv[1:]))
