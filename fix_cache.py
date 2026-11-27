"""fix_cache.py <model_dir>...: load every .aimodel; on nilError, evict its Core AI cache entry (key = main.hash) and retry once.
Stale entries appear after other ANE work evicts the compiled programs (the Core AI cache entry stays, load fails at once)."""
import sys, time, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import BRIDGE, coreai_cache_dirs
sys.path.insert(0, str(BRIDGE)); import coreai_bridge as B

def tryload(p):
    try:
        m = B.Model(p, compute="ane")
        for n in m.function_names: m.function(n)   # every entry the model has (KT 16K back chunks have no v1)
        return True
    except Exception: return False

for d in sys.argv[1:]:
    for p in sorted(Path(d).glob("*.aimodel")):
        t = time.time(); ok = tryload(p); note = ""
        if not ok:
            h = (p / "main.hash").read_bytes().hex()
            for C in coreai_cache_dirs():
                if (C / h).exists(): shutil.rmtree(C / h); note = " evicted"
            ok = tryload(p)
        print(d.split("/")[-1], p.name, "ok" if ok else "FAIL", note, f"{time.time() - t:.0f}s", flush=True)
