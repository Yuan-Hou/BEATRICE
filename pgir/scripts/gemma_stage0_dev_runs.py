"""Gemma retrain stage 0: norag + vanilla over the 3 canonical dev sets.

Their per-item EM diff is the real-retrieval "does retrieval help" signal that
replaces the PGIR dev traces the Qwen-era SKR-devreal / Adaptive dev-override
were built from (no Gemma PGIR traces exist yet; norag vs vanilla has the
identical round-0 vs round-1 semantics: closed book vs one e5 k=3 retrieval).
"""
import os
import sys

sys.path.insert(0, "<package_root>")

os.environ["PGIR_FLASHRAG_YAML"] = "pgir/configs/flashrag_gemma.yaml"
os.environ["PGIR_RUNS_ROOT"] = "pgir/runs/flashrag_gemma"

import json
import time
from pathlib import Path

from pgir.run_baselines import run_one

MARK = Path("pgir/runs/flashrag_gemma/.done")
MARK.mkdir(parents=True, exist_ok=True)

for m in ("norag", "vanilla"):
    for ds in ("hotpotqa", "twowiki", "musique"):
        mk = MARK / f"dev_{m}_{ds}.done"
        if mk.exists():
            print(f"=== skip dev {m}/{ds}", flush=True)
            continue
        print(f"=== start dev {m}/{ds} {time.strftime('%F %T')}", flush=True)
        s = run_one(m, ds, "dev", save_note=f"gemma4_{m}_{ds}_dev")
        mk.write_text(json.dumps(s))
        print(f"=== done dev {m}/{ds}", flush=True)
print("STAGE0 DEV RUNS DONE", flush=True)
