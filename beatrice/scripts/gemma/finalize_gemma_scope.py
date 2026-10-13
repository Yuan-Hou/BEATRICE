"""Assemble the final Gemma scope at exactly the Qwen-era sizes.

Selection rule per cell (dataset x hf_split x scope group):
  1. keep every original scope id that is triple-unanimous (closed-book +
     with-context + rewritten-context all 3-vote unanimous), in scope order;
  2. fill the remainder with supplement candidates in manifest order that
     passed the same triple-unanimity bar.
nq/triviaqa (no rewritten condition) use double-unanimity and fill to the Qwen
feature-success sizes (1182/1277).

Outputs:
  configs/sampling/gemma4_scope_ids.json      authoritative Gemma split file
  data/gemma4/labels/final/*_labels.jsonl     labels for selected ids only
  data/gemma4/sources_final/*.jsonl           example sources for selected ids
  data/gemma4/labels/final/summary_final.{json,md}
Exits non-zero on any shortfall or verification failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "data" / "generations"
SUPP_SRC = ROOT / "data" / "gemma4" / "sources_supplement"
SRC = ROOT / "data" / "gemma4" / "sources"
LABELS = ROOT / "data" / "gemma4" / "labels"
FINAL_LABELS = LABELS / "final"
FINAL_SRC = ROOT / "data" / "gemma4" / "sources_final"

IN_DOMAIN = ("hotpotqa_supporting_only", "musique", "twowiki_supporting_only", "popqa")
NQ_TQA_TARGETS = {"nq": 1182, "triviaqa": 1277}


def jl(path: Path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def unanimous_verdict(run: dict):
    votes = [d.get("parsed") for d in (run.get("judge_vote") or {}).get("variants", [])]
    if len(votes) != 3 or any(v is None for v in votes) or len(set(votes)) != 1:
        return None
    return bool(votes[0])


def ds_key(i: str) -> str:
    p = i.split("-")
    if p[0] == "musique":
        return "musique", p[1]
    if p[0] in ("popqa", "nq", "triviaqa"):
        return p[0], p[1]
    name = {"hotpotqa": "hotpotqa_supporting_only", "twowiki": "twowiki_supporting_only"}[p[0]]
    return name, p[2]


def load_labels(name: str) -> dict[str, dict]:
    return {r["id"]: r for r in jl(LABELS / f"{name}_labels.jsonl")}


def labels_from_generations(path: Path, want_runs) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    for rec in jl(path):
        ex = rec.get("example") or {}
        runs_out = {}
        ok = True
        for rn in want_runs:
            run = (rec.get("runs") or {}).get(rn)
            v = unanimous_verdict(run) if run else None
            if v is None:
                ok = False
                break
            runs_out[rn] = {"is_correct": v}
        if ok:
            out[ex["id"]] = {"id": ex["id"], "dataset": ex.get("dataset"),
                             "split": ex.get("split"), "runs": runs_out}
    return out


def main() -> None:
    scope = json.load((ROOT / "configs/sampling/fixed_scope_ids.json").open())["ids"]
    manifest = json.load((SUPP_SRC / "manifest.json").open())["candidates"]

    # -- original keep sets --
    base_lab = {}
    for n in IN_DOMAIN:
        base_lab.update(load_labels(n))
    rew_lab = load_labels("rewritten_context")
    kept = set(base_lab) & set(rew_lab)

    # -- supplement pass sets --
    supp_base = {}
    for n in IN_DOMAIN:
        supp_base.update(labels_from_generations(
            GEN / f"supp_{n}_gemma4e4b_results.jsonl", ("without_context", "with_context")))
    supp_rew = labels_from_generations(
        GEN / "supp_rewritten_context_gemma4e4b_results.jsonl", ("with_context",))
    supp_pass = set(supp_base) & set(supp_rew)
    print(f"original triple-unanimous: {len(kept)}; supplement pass: {len(supp_pass)}"
          f" (base {len(supp_base)} ∩ rewritten {len(supp_rew)})")

    # -- fill in-domain cells --
    mkey_for = {
        ("hotpotqa_supporting_only", "train"): "hotpotqa/train",
        ("hotpotqa_supporting_only", "validation"): "hotpotqa/validation",
        ("musique", "train"): "musique/train",
        ("musique", "validation"): "musique/validation",
        ("twowiki_supporting_only", "train"): "twowiki/train",
        ("twowiki_supporting_only", "validation"): "twowiki/validation",
        ("popqa", "test"): "popqa/test",
    }
    cursor = {k: 0 for k in manifest}
    new_scope = {}
    fill_stats = {}
    shortfall = False
    for grp in ("train", "validation", "test"):
        sel = []
        cells: dict[tuple, list] = {}
        for i in scope[grp]:
            cells.setdefault(ds_key(i), []).append(i)
        for cell, ids in cells.items():
            kept_ids = [i for i in ids if i in kept]
            need = len(ids) - len(kept_ids)
            mk = mkey_for[cell]
            fill = []
            lst = manifest.get(mk, [])
            while need > len(fill) and cursor[mk] < len(lst):
                cand = lst[cursor[mk]]
                cursor[mk] += 1
                if cand in supp_pass:
                    fill.append(cand)
            fill_stats[f"{grp}:{cell[0]}/{cell[1]}"] = {
                "total": len(ids), "kept_original": len(kept_ids), "filled": len(fill)}
            if len(fill) < need:
                print(f"SHORTFALL {grp} {cell}: need {need}, filled {len(fill)}")
                shortfall = True
            sel.extend(kept_ids + fill)
        new_scope[grp] = sel

    # -- nq / triviaqa --
    nqtqa_sel = {}
    for n, target in NQ_TQA_TARGETS.items():
        orig = load_labels(n)
        supp = labels_from_generations(
            GEN / f"supp_{n}_gemma4e4b_results.jsonl", ("without_context", "with_context"))
        sel = list(orig)
        have = set(sel)  # manifest candidates may have become unanimous originals
        for cand in manifest.get(f"{n}/validation", []):
            if len(sel) >= target:
                break
            if cand in supp and cand not in have:
                sel.append(cand)
                have.add(cand)
        if len(set(sel)) != len(sel):
            raise RuntimeError(f"{n}: duplicate ids in selection")
        fill_stats[f"{n}"] = {"total": target, "kept_original": len(orig),
                              "filled": len(sel) - len(orig)}
        if len(sel) < target:
            print(f"SHORTFALL {n}: need {target - len(orig)}, filled {len(sel) - len(orig)}")
            shortfall = True
        nqtqa_sel[n] = sel
        supp_base.update(supp)
        supp_base.update(orig)  # original nq/tqa labels live outside base_lab (IN_DOMAIN only)

    if shortfall:
        print("FINALIZE_SHORTFALL — enlarge candidate buffers and rerun the supplement pipeline")
        sys.exit(1)

    # -- emit scope file --
    scope_out = {
        "seed": 20260713,
        "source": "gemma4_e4b_triple_unanimous_refill_of_fixed_scope_ids",
        "description": ("Gemma-4-E4B scope at exactly the Qwen sizes. Original fixed_scope_ids "
                        "positions kept where closed-book+withctx+rewritten labels are 3-vote "
                        "unanimous (Qwen3.5-27B judge); deficits refilled with freshly sampled "
                        "candidates passing the same bar. nq/triviaqa selections stored in "
                        "extra_ids."),
        "groups": 3,
        "ids": new_scope,
        "extra_ids": nqtqa_sel,
    }
    out_scope = ROOT / "configs/sampling/gemma4_scope_ids.json"
    out_scope.write_text(json.dumps(scope_out, indent=1))

    # -- emit final labels + sources --
    FINAL_LABELS.mkdir(parents=True, exist_ok=True)
    FINAL_SRC.mkdir(parents=True, exist_ok=True)
    all_sel = {i for ids in new_scope.values() for i in ids}

    def write_subset(name: str, selected: set[str], label_map: dict, src_paths: list[Path]):
        with (FINAL_LABELS / f"{name}_labels.jsonl").open("w") as f:
            n_lab = 0
            for i, lab in label_map.items():
                if i in selected:
                    f.write(json.dumps(lab, ensure_ascii=False) + "\n")
                    n_lab += 1
        seen = set()
        with (FINAL_SRC / f"{name}.jsonl").open("w") as f:
            for p in src_paths:
                if not p.exists():
                    continue
                for ex in jl(p):
                    if ex["id"] in selected and ex["id"] not in seen:
                        seen.add(ex["id"])
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        if n_lab != len(selected) or len(seen) != len(selected):
            print(f"VERIFY FAIL {name}: selected {len(selected)}, labels {n_lab}, sources {len(seen)}")
            return False
        return True

    ok = True
    merged_base_labels = dict(base_lab)
    merged_base_labels.update(supp_base)
    for n in IN_DOMAIN:
        sel_n = {i for i in all_sel if ds_key(i)[0] == n}
        ok &= write_subset(n, sel_n, merged_base_labels,
                           [SRC / f"{n}.jsonl", SUPP_SRC / f"{n}.jsonl"])
    merged_rew_labels = dict(rew_lab)
    merged_rew_labels.update(supp_rew)
    ok &= write_subset("rewritten_context", all_sel, merged_rew_labels,
                       [SRC / "rewritten_context.jsonl", SUPP_SRC / "rewritten_supplement.jsonl"])
    for n in NQ_TQA_TARGETS:
        ok &= write_subset(n, set(nqtqa_sel[n]), merged_base_labels,
                           [SRC / f"{n}.jsonl", SUPP_SRC / f"{n}.jsonl"])

    # -- verify counts --
    expect = {"train": 30000, "validation": 1500, "test": 2500}
    for grp, n in expect.items():
        if len(new_scope[grp]) != n or len(set(new_scope[grp])) != n:
            print(f"VERIFY FAIL group {grp}: {len(new_scope[grp])} (unique {len(set(new_scope[grp]))}) != {n}")
            ok = False
    if len(all_sel) != 34000:
        print(f"VERIFY FAIL total scope ids: {len(all_sel)} != 34000")
        ok = False

    summary = {"fill_stats": fill_stats, "scope_path": str(out_scope),
               "supplement_pass": len(supp_pass), "original_kept": len(kept)}
    (FINAL_LABELS / "summary_final.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Gemma scope finalization (exact Qwen sizes)", "",
             "| cell | total | kept original | refilled |", "|---|---|---|---|"]
    for k, s in sorted(fill_stats.items()):
        lines.append(f"| {k} | {s['total']} | {s['kept_original']} | {s['filled']} |")
    (FINAL_LABELS / "summary_final.md").write_text("\n".join(lines) + "\n")
    for k, s in sorted(fill_stats.items()):
        print(f"{k}: total {s['total']} kept {s['kept_original']} filled {s['filled']}")

    if not ok:
        print("FINALIZE_VERIFY_FAILED")
        sys.exit(1)
    print("FINALIZE_OK")


if __name__ == "__main__":
    main()
