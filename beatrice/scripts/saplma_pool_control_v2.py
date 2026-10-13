"""v2: full-train, decompose the hidden-only vs SAPLMA gap into POOLING vs TOKEN-SCOPE.

Fixes v1 confounds:
  - FULL train (no subsample) -> absolutes comparable to standalone SAPLMA (.9109 test)
    and the deployed hidden-only ConvBranch (.9293 test).
  - Adds ALL-TOKENS pooling (context+question+answer), since the ConvBranch pools the full
    sequence while SAPLMA uses answer tokens only.

Ladder (all MLP[256,128,64] + StandardScaler on L17 sidecar hidden):
  answer_last   : last answer token           (classic SAPLMA position)
  answer_mean   : mean over answer tokens      (== standalone SAPLMA-mean; sanity anchor)
  answer_maxmean: concat(max,mean) answer      (our _masked_pool, minus conv, answer-only)
  all_mean      : mean over ctx+q+ans tokens    (token-scope axis)
  all_maxmean   : concat(max,mean) over ctx+q+ans (our _masked_pool scope, minus conv)

Read: answer_mean should reproduce ~.9109 (validates pipeline). If all_* jumps toward
.929, the gap is TOKEN SCOPE; residual to .929 is the conv encoder.
LLM-free, CPU-only, streamed.
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

REPO = "."
sys.path.insert(0, os.path.join(REPO, "src"))
os.chdir(REPO)
from ragprobe.training.dataset import load_feature_meta, get_with_context_correct  # noqa
from ragprobe.baselines.faithful import _train_mlp, _mlp_predict, auroc  # noqa

SCOPE = "configs/sampling/fixed_scope_ids.json"
L = "data/features_judge_consensus/qwen3_5_9b_L17_rd"
RW = "data/features_judge_consensus/qwen3_5_9b_rewritten_context_L17_rd"
MULTIHOP = [f"{L}/musique_features_meta.jsonl", f"{L}/hotpotqa_supporting_only_features_meta.jsonl",
            f"{L}/twowiki_supporting_only_features_meta.jsonl"]
REWRITTEN = [f"{RW}/rewritten_context_cqa_features_meta.jsonl"]
POPQA = [f"{L}/popqa_features_meta.jsonl"]

def partition(group):
    return {"enabled": True, "ids_path": SCOPE, "group": group, "source_splits": None}

def collect(paths, group):
    out = []
    for p in paths:
        recs = load_feature_meta([p], partition=partition(group))
        k = 0
        for r in recs:
            sp = (r.get("feature_extraction", {}).get("rag", {}) or {}).get("sidecar_path")
            lab = get_with_context_correct(r)
            if sp and os.path.exists(sp) and lab is not None:
                out.append((sp, int(bool(lab)))); k += 1
        print(f"  [{group}] {os.path.basename(p)}: {k}", flush=True)
    return out

def pool_one(sp):
    o = torch.load(sp, map_location="cpu", weights_only=False)
    lh = o["last_hidden_states"]
    a = lh["answer"]
    if a is None or a.shape[0] == 0:
        return None
    a = a.float().numpy()
    segs = [lh[s].float().numpy() for s in ("context", "question", "answer")
            if lh.get(s) is not None and lh[s].shape[0] > 0]
    allt = np.concatenate(segs, axis=0)
    return np.concatenate([
        a[-1], a.mean(0), a.max(0),            # answer: last, mean, max  (0:3d)
        allt.mean(0), allt.max(0),             # all:    mean, max        (3d:5d)
    ]).astype(np.float32)

def build(items, workers=24):
    X = [None] * len(items); y = np.array([it[1] for it in items], dtype=np.float32)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, vec in enumerate(ex.map(pool_one, [it[0] for it in items])):
            X[i] = vec
            if (i + 1) % 5000 == 0:
                print(f"    {i+1}/{len(items)} ({(i+1)/(time.time()-t0):.0f}/s)", flush=True)
    keep = [i for i, v in enumerate(X) if v is not None]
    return np.stack([X[i] for i in keep]), y[keep]

def views(X, d):
    a_last, a_mean, a_max = X[:, :d], X[:, d:2*d], X[:, 2*d:3*d]
    t_mean, t_max = X[:, 3*d:4*d], X[:, 4*d:5*d]
    return {
        "answer_last": a_last,
        "answer_mean": a_mean,
        "answer_maxmean": np.concatenate([a_max, a_mean], 1),
        "all_mean": t_mean,
        "all_maxmean": np.concatenate([t_max, t_mean], 1),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp_pool_control_v2.json")
    args = ap.parse_args()
    print("== collect ==", flush=True)
    tr_items = collect(MULTIHOP + REWRITTEN, "train")
    va_items = collect(MULTIHOP, "validation")
    te_items = collect(MULTIHOP, "test")
    po_items = collect(POPQA, "test")
    print(f"== build ==  train={len(tr_items)} val={len(va_items)} test={len(te_items)} popqa={len(po_items)}", flush=True)
    Xtr, ytr = build(tr_items); print("  train", Xtr.shape, flush=True)
    Xva, yva = build(va_items); print("  val", Xva.shape, flush=True)
    Xte, yte = build(te_items); print("  test", Xte.shape, flush=True)
    Xpo, ypo = build(po_items); print("  popqa", Xpo.shape, flush=True)
    d = Xtr.shape[1] // 5
    tr, va, te, po = views(Xtr, d), views(Xva, d), views(Xte, d), views(Xpo, d)
    res = {"d": d, "n": {"train": len(ytr), "val": len(yva), "test": len(yte), "popqa": len(ypo)},
           "reference": {"convbranch_hidden_only_test": 0.9293, "standalone_saplma_test": 0.9109,
                         "champion_test": 0.9317}}
    for pool in ["answer_last", "answer_mean", "answer_maxmean", "all_mean", "all_maxmean"]:
        net, sc, pva = _train_mlp(tr[pool], ytr, va[pool], device="cpu")
        row = {"val": auroc(yva, pva), "test": auroc(yte, _mlp_predict(net, sc, te[pool])),
               "popqa": auroc(ypo, _mlp_predict(net, sc, po[pool]))}
        res[pool] = row
        print(f"[{pool:14s}] val={row['val']:.4f} test={row['test']:.4f} popqa={row['popqa']:.4f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=2)
    try:
        json.dump(res, open(os.path.join(REPO, "temp", "saplma_pool_control_v2.json"), "w"), indent=2)
    except Exception as e:
        print("temp copy failed:", e)
    print("WROTE", args.out, flush=True)

if __name__ == "__main__":
    main()
