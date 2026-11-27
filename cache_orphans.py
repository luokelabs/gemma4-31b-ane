"""Delete Core AI cache entries whose hash matches no *.aimodel/main.hash under ~ (excluding ~/Library), older than 15 min.
Every build adds 20-30 GB of compiled programs; with little free disk macOS evicts fresh entries and loads fail (nilError).
   cache_orphans.py [--dry]   (env MIN_AGE seconds, default 900)"""
import os, sys, time, shutil, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import coreai_cache_dirs
H = Path.home()
keep = set()
for dp, dn, fn in os.walk(H, followlinks=False):
    if dp.startswith(str(H / "Library")) or "/." in dp[len(str(H)):]: dn[:] = []; continue
    if "main.hash" in fn and dp.endswith(".aimodel"):
        try: keep.add(Path(dp, "main.hash").read_bytes().hex())
        except Exception: pass
orph = [e for C in coreai_cache_dirs() for e in C.iterdir() if e.name not in keep and time.time() - e.stat().st_mtime > float(os.environ.get("MIN_AGE", 900))]
du = lambda p: int(subprocess.check_output(["du", "-sk", str(p)]).split()[0])
gb = sum(du(e) for e in orph) / 1e6
print(f"keep {len(keep)} hashes; orphan entries {len(orph)} {gb:.1f} GB")
if "--dry" not in sys.argv:
    for e in orph: shutil.rmtree(e, ignore_errors=True)
