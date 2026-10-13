"""M3 tuning driver: val traces -> calibration -> threshold sweep -> config.

Steps (all on the canonical dev split only — test is untouched):
  1. collect full-budget traces per dataset (run_baselines --method pgir
     --split dev --full-budget), or point --traces at existing ones;
  2. fit per-mode calibrators on (p_raw, EM-correctness) of all dev arms;
  3. distribution-shift checkup (spec §9 risk 1): AUROC of p_raw vs EM label
     on retrieved-context arms — compare against the oracle-context test AUROC
     (~.93) before trusting the probe on real retrieval;
  4. sweep tau_hi x budget on replayed traces, pick best EM subject to
     gen_calls <= --gen-budget (spec §6-5), write chosen config + calibrator.

Run: cd <package_root> && python -m pgir.tuning.run_tuning --traces runs/...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pgir.calibration import PerModeCalibrator, brier, ece, reliability_table
from pgir.tuning.trace import em, load_golds, load_trace, select_config, sweep

PGIR_DIR = Path(__file__).resolve().parent.parent


def auroc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    for v in np.unique(s):
        m = s == v
        ranks[m] = ranks[m].mean()
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * (~pos).sum()))


def arm_labels(traces, golds):
    """(p_raw, EM-label, mode) for every successful arm, plus the probe
    checkpoint seen per mode (for calibrator<->probe provenance)."""
    out = {"question": ([], []), "context": ([], [])}
    ckpts: dict[str, set] = {"question": set(), "context": set()}
    for it in traces:
        g = golds[it["example_id"]]
        for a in it["pool"]:
            if a["status"] != "success":
                continue
            out[a["mode"]][0].append(float(a["p_raw"]))
            out[a["mode"]][1].append(int(em(a["answer"], g)))
            if a.get("probe_checkpoint"):
                ckpts[a["mode"]].add(a["probe_checkpoint"])
    return out, ckpts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True,
                    help="full-budget dev trace jsonl files (all datasets)")
    ap.add_argument("--gen-budget", type=float, required=True,
                    help="max mean #gen_calls (e.g. IRCoT dev mean, §6-5)")
    ap.add_argument("--out", default=str(PGIR_DIR / "runs" / "tuning"))
    ap.add_argument("--scope", default="beatrice/configs/sampling/fixed_scope_ids.json",
                    help="scope-ids json defining the dev admission; trace rows "
                         "outside its validation list are DROPPED (not an error)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    golds = load_golds(PGIR_DIR / "datasets")

    # hard test-split guard: no trace id may leak in from test/OOD. Rows that are
    # simply outside this scope's dev admission are dropped, not fatal — but
    # anything that appears in a FORBIDDEN split still aborts the fit.
    scope = json.loads(Path(args.scope).read_text())
    dev_ids = set(scope["ids"]["validation"])
    forbidden = set(scope["ids"]["test"]) | {i for v in scope.get("extra_ids", {}).values()
                                             for i in v}
    traces, dropped = [], 0
    for t in args.traces:
        part = load_trace(t)
        for it in part:
            if it["example_id"] in forbidden:
                raise RuntimeError(
                    f"test/OOD id {it['example_id']} in tuning traces — tuning is dev-only!")
        keep = [it for it in part if it["example_id"] in dev_ids]
        dropped += len(part) - len(keep)
        traces.extend(keep)
    print(f"loaded {len(traces)} trace items from {len(args.traces)} files "
          f"(dropped {dropped} outside {Path(args.scope).name} dev admission)")

    # ---- calibration on dev arms (EM labels) --------------------------------
    labels, ckpts = arm_labels(traces, golds)
    cal = PerModeCalibrator()
    report = {}
    for mode, (p, y) in labels.items():
        if len(p) < 50:
            print(f"[warn] mode={mode}: only {len(p)} arms, skipping calibration")
            continue
        mode_ckpts = ckpts.get(mode, set())
        if len(mode_ckpts) > 1:
            raise RuntimeError(
                f"mode={mode} traces mix probe checkpoints {sorted(mode_ckpts)} — "
                f"recollect traces under a single probe before calibrating")
        probe_ckpt = next(iter(mode_ckpts), None)
        cal.fit(p, y, mode, "dev", probe_checkpoint=probe_ckpt)
        pc = cal.calibrate(np.asarray(p), mode)
        report[mode] = {
            "n": len(p), "kind": cal.meta[mode]["kind"],
            "auroc_raw": auroc(p, y),
            "ece_raw": ece(p, y), "ece_cal": ece(pc, y),
            "brier_raw": brier(p, y), "brier_cal": brier(pc, y),
            "reliability": reliability_table(pc, y),
        }
        print(f"[{mode}] n={len(p)} AUROC={report[mode]['auroc_raw']:.4f} "
              f"ECE {report[mode]['ece_raw']:.4f}->{report[mode]['ece_cal']:.4f}")
    cal.save(out / "calibrator.json")

    # ---- distribution-shift checkup (risk 1) --------------------------------
    ctx_auroc = report.get("context", {}).get("auroc_raw")
    if ctx_auroc is not None and ctx_auroc < 0.88:
        print(f"[ALERT] context-mode AUROC on retrieved passages = {ctx_auroc:.3f} "
              "(oracle-context test ~= .93). Distribution shift — consider mixing "
              "retrieved-context data into probe training (spec §9-1).")

    # ---- threshold sweep + constrained selection -----------------------------
    rows = sweep(traces, golds, calibrator=cal)
    chosen = select_config(rows, args.gen_budget)
    (out / "sweep.json").write_text(json.dumps(
        {"rows": rows, "chosen": chosen, "gen_budget": args.gen_budget,
         "calibration": {m: {k: v for k, v in r.items() if k != "reliability"}
                         for m, r in report.items()}}, indent=2))
    (out / "calibration_report.json").write_text(json.dumps(report, indent=2))
    print("chosen config:", json.dumps(chosen, indent=2))
    print(f"artifacts in {out}")


if __name__ == "__main__":
    main()
