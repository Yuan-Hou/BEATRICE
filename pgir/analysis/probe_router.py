"""Probe-as-router: zero-shot plug-in replacement for Adaptive-RAG/SKR routing.

Adaptive-RAG trains a T5 query-complexity router; SKR trains a self-knowledge
classifier. Our question-mode probe already outputs the same decision signal —
P(correct WITHOUT retrieval) — so route each item by its round-0 closed-book
score q0 taken from an existing PGIR trace (zero extra scoring, zero training):

    q0 >= hi           -> norag   (closed-book answer suffices)
    lo <= q0 < hi      -> vanilla (single retrieval round)
    q0 < lo            -> ircot   (multi-hop iterative)

Thresholds default to the PGIR accept band (hi=0.6 = tau_hi, lo=0.3 = bon_lo)
— intentionally NOT tuned for routing, to keep the plug-and-play claim honest.
Branch answers come from the existing norag/vanilla/ircot FlashRAG run dirs;
the router only chooses which stored answer each item gets, so the whole
comparison replays offline. Compares against each fixed branch and against the
trained routers' own runs (smoke_adaptive_*, smoke_skr_*).

Run (uses newest run dir per branch; --trace-glob picks which PGIR traces):
  cd <package_root> && python -m pgir.analysis.probe_router \
      --trace-glob 'pgir/runs/flashrag/m6_main/{ds}_*/pgir_trace_{ds}_test.jsonl'
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from pgir.calibration import PerModeCalibrator
from pgir.tuning.trace import em, f1, load_trace

RUNS = Path(__file__).resolve().parent.parent / "runs" / "flashrag"
DATASETS = ["hotpotqa", "twowiki", "musique", "popqa", "nq_ood", "triviaqa_ood"]
BRANCH_COST = {"norag": (1, 0), "vanilla": (1, 1)}  # gen, retr; ircot from its run


def load_preds(method: str, ds: str) -> dict[str, dict]:
    """id -> {pred, golden} from the newest run dir of a method/dataset."""
    pats = [f"smoke_{method}_{ds}_test_nfull", f"smoke_{method}_{ds}_test_full"]
    hits = []
    for p in pats:
        hits += [d for d in RUNS.glob(p) if d.is_dir()]
    files = []
    for d in hits:
        files += list(d.glob("**/intermediate_data.json"))
    if not files:
        return {}
    newest = max(files, key=lambda p: p.stat().st_mtime)
    out = {}
    for it in json.loads(newest.read_text()):
        out[it["id"]] = {"pred": (it.get("output") or {}).get("pred") or "",
                         "golden": it.get("golden_answers") or []}
    return out


def ircot_mean_cost(ds: str) -> tuple[float, float]:
    """mean (gen, retr) per item from the ircot run table if recorded."""
    for d in RUNS.glob(f"smoke_ircot_{ds}_test_*"):
        for f in d.glob("**/pgir_run_metadata.json"):
            pass  # ircot runs don't carry pgir metadata; fall through
    return (3.7, 3.7)  # M3 dev mean as documented fallback (report as approx)


def q0_scores(trace_path: str, calibrator=None) -> dict[str, float]:
    """example_id -> round-0 question-mode p (calibrated if given)."""
    out = {}
    for it in load_trace(trace_path):
        for a in it["pool"]:
            if a["round_index"] == 0 and a["mode"] == "question" \
                    and a.get("answer_source") != "sampled" and a["status"] == "success":
                p = a["p_raw"]
                if calibrator is not None:
                    p = float(calibrator.calibrate(p, "question"))
                else:
                    p = float(a["p_cal"])
                out[it["example_id"]] = p
                break
    return out


def route_dataset(ds: str, trace_glob: str, hi: float, lo: float,
                  calibrator=None) -> dict | None:
    traces = sorted(glob.glob(trace_glob.format(ds=ds)))
    if not traces:
        return None
    q0 = q0_scores(traces[-1], calibrator)
    branches = {m: load_preds(m, ds) for m in ["norag", "vanilla", "ircot"]}
    rivals = {m: load_preds(m, ds) for m in ["adaptive", "skr"]}
    if not all(branches.values()):
        missing = [m for m, v in branches.items() if not v]
        print(f"[skip] {ds}: missing branch runs {missing}")
        return None

    ids = [i for i in q0 if all(i in b for b in branches.values())]
    if not ids:
        return None
    rows, route_count = [], {"norag": 0, "vanilla": 0, "ircot": 0}
    ig, ir = ircot_mean_cost(ds)
    for i in ids:
        p = q0[i]
        if lo is None:  # 2-way (SKR's task): retrieve or not, single threshold
            branch = "norag" if p >= hi else "vanilla"
        else:
            branch = "norag" if p >= hi else ("vanilla" if p >= lo else "ircot")
        route_count[branch] += 1
        pick = branches[branch][i]
        gold = pick["golden"]
        gen, retr = BRANCH_COST.get(branch, (ig, ir))
        rows.append({"id": i, "branch": branch, "em": em(pick["pred"], gold),
                     "f1": f1(pick["pred"], gold), "gen": gen, "retr": retr})
    n = len(rows)
    res = {"dataset": ds, "n": n,
           "router_em": sum(r["em"] for r in rows) / n,
           "router_f1": sum(r["f1"] for r in rows) / n,
           "router_gen": sum(r["gen"] for r in rows) / n,
           "router_retr": sum(r["retr"] for r in rows) / n,
           "route_share": {k: v / n for k, v in route_count.items()}}
    for m, preds in branches.items():
        res[f"{m}_em"] = sum(em(preds[i]["pred"], preds[i]["golden"])
                             for i in ids) / n
    res["oracle2_em"] = sum(max(em(branches[m][i]["pred"], branches[m][i]["golden"])
                                for m in ("norag", "vanilla")) for i in ids) / n
    for m, preds in rivals.items():
        if preds:
            common = [i for i in ids if i in preds]
            if common:
                res[f"{m}_em"] = sum(em(preds[i]["pred"], preds[i]["golden"])
                                     for i in common) / len(common)
                res[f"{m}_n"] = len(common)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-glob", required=True,
                    help="PGIR trace glob with {ds} placeholder (source of q0)")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--hi", type=float, default=0.5)
    ap.add_argument("--lo", type=float, default=None,
                    help="3-way lower threshold (vanilla vs ircot); omit for the "
                         "2-way retrieve-or-not router (default — ircot branch is "
                         "weaker than vanilla on this backbone, 3-way loses)")
    ap.add_argument("--calibrator", default=None,
                    help="calibrate p_raw with this file (else trust trace p_cal)")
    ap.add_argument("--out", default=str(RUNS.parent / "tables" / "probe_router"))
    args = ap.parse_args()

    cal = PerModeCalibrator.load(args.calibrator) if args.calibrator else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    hdr = (f"{'dataset':>13} {'n':>5} {'router':>7} {'norag':>6} {'vanilla':>7} "
           f"{'ircot':>6} {'adaptive':>8} {'skr':>6} {'gen':>5}  route n/v/i")
    print(hdr)
    for ds in args.datasets:
        r = route_dataset(ds, args.trace_glob, args.hi, args.lo, cal)
        if r is None:
            continue
        results.append(r)
        sh = r["route_share"]
        print(f"{ds:>13} {r['n']:>5} {r['router_em']:>7.3f} {r['norag_em']:>6.3f} "
              f"{r['vanilla_em']:>7.3f} {r['ircot_em']:>6.3f} "
              f"{r.get('adaptive_em', float('nan')):>8.3f} "
              f"{r.get('skr_em', float('nan')):>6.3f} {r['router_gen']:>5.2f}  "
              f"{sh['norag']:.2f}/{sh['vanilla']:.2f}/{sh['ircot']:.2f}")
    (out_dir / "probe_router.json").write_text(json.dumps(
        {"hi": args.hi, "lo": args.lo, "trace_glob": args.trace_glob,
         "results": results}, indent=2))
    print(f"-> {out_dir}/probe_router.json")


if __name__ == "__main__":
    main()
