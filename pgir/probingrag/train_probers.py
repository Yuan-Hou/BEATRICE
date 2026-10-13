"""Train Probing-RAG probers on the ready-made layer features.

Data: beatrice/data/layer_features/{noctx,withctx}/train__shard*.npz
  hidden_mean (N, 33, 4096) = mean-pooled hidden states, all layers, from the
  training-format forward (prompt + generated answer); index label = answer
  correct in that condition (1 = no further retrieval needed).
Both conditions pooled (paper trains on no-retrieval + retrieved versions),
classes balanced to 0.5 by subsampling (their README instruction).

Prober (paper spec): LayerNorm -> Linear(4096,256) -> ReLU -> Linear(256,2).
One prober per candidate layer (even layers 6..30); the top-K layers by
validation AUROC are selected for inference (logit margins summed).

Run: CUDA_VISIBLE_DEVICES=4 PYTORCH_NVML_BASED_CUDA_CHECK=0 \
       python -m pgir.probingrag.train_probers
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LF = Path("beatrice/data/layer_features")
OUT = Path(__file__).resolve().parent.parent / "assets" / "probingrag_probers"
CANDIDATE_LAYERS = list(range(6, 31, 2))  # even mid layers of 32 (index into 33 incl. embed)
DIM = 4096                                # hidden size (--gemma switches to 2560)
TOP_K = 5
SEED = 17


class Prober(nn.Module):
    def __init__(self, dim=4096, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def load_split(split_glob: str, overrides: dict | None = None,
               keep_ids: set | None = None):
    """overrides: {cond: {example_id: bool}} — optional label replacement map;
    rows missing from the map are dropped. keep_ids: optional id admission for
    this split. Both default to the stored index labels/rows."""
    xs, ys = [], []
    for cond in ("noctx", "withctx"):
        ov = (overrides or {}).get(cond)
        for npz_path in sorted((LF / cond).glob(f"{split_glob}.npz")):
            if "rewritten" in npz_path.name:
                continue
            idx_file = Path(str(npz_path)[:-4] + ".index.jsonl")
            rows = [json.loads(l) for l in open(idx_file)]
            arr = np.load(npz_path)["hidden_mean"]  # (N, 33, 4096) fp16
            assert arr.shape[0] == len(rows)
            sel, labels = [], []
            for i, r in enumerate(rows):
                eid = str(r["example_id"])
                if keep_ids is not None and eid not in keep_ids:
                    continue
                if ov is not None:
                    if eid not in ov:
                        continue  # not present in the override map
                    labels.append(int(ov[eid]))
                else:
                    labels.append(int(r["label"]))
                sel.append(i)
            if not sel:
                continue
            xs.append(arr[np.asarray(sel)])
            ys.append(np.asarray(labels, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def balance(x, y, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    keep = np.concatenate([rng.choice(pos, n, replace=False),
                           rng.choice(neg, n, replace=False)])
    rng.shuffle(keep)
    return x[keep], y[keep]


def auroc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):
        m = scores == v
        ranks[m] = ranks[m].mean()
    pos = labels == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * (~pos).sum()))


def train_one(layer, xtr, ytr, xva, yva, device, epochs=4, bs=512, lr=1e-3):
    torch.manual_seed(SEED)
    model = Prober(dim=DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    xt = torch.from_numpy(xtr[:, layer].astype(np.float32))
    yt = torch.from_numpy(ytr)
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xt[b].to(device)), yt[b].to(device))
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(xva[:, layer].astype(np.float32)).to(device))
        margin = (logits[:, 1] - logits[:, 0]).cpu().numpy()  # >0 => no retrieval needed
    return model, auroc(margin, yva), margin


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma", action="store_true",
                    help="train on the gemma-4-E4B-it layer features "
                         "(beatrice data/gemma4/layer_features, 43 layers x 2560, "
                         "labels baked into the index rows), gated by "
                         "gemma4_scope_ids; output -> probingrag_probers_gemma")
    args = ap.parse_args()
    global OUT, LF, CANDIDATE_LAYERS, DIM
    overrides = tr_ids = va_ids = None
    if args.gemma:
        LF = Path("beatrice/data/gemma4/layer_features")
        OUT = OUT.parent / "probingrag_probers_gemma"
        DIM = 2560
        # same relative depth band as the qwen 6..30 of 33 (mid layers, even)
        CANDIDATE_LAYERS = list(range(8, 41, 2))
        scope = json.loads(Path("beatrice/configs/sampling/"
                                "gemma4_scope_ids.json").read_text())
        tr_ids, va_ids = set(scope["ids"]["train"]), set(scope["ids"]["validation"])

    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xtr, ytr = load_split("train__shard*", overrides, tr_ids)
    xva, yva = load_split("validation__shard*", overrides, va_ids)
    xtr, ytr = balance(xtr, ytr, rng)
    print(f"train {xtr.shape} balanced pos-rate={ytr.mean():.3f} | val {xva.shape} pos={yva.mean():.3f}")

    results, models, margins = {}, {}, {}
    for layer in CANDIDATE_LAYERS:
        model, au, margin = train_one(layer, xtr, ytr, xva, yva, device)
        results[layer] = au
        models[layer] = model
        margins[layer] = margin
        print(f"layer {layer:2d}: val AUROC {au:.4f}", flush=True)

    top = sorted(results, key=results.get, reverse=True)[:TOP_K]
    print("selected layers:", top)
    summed = np.sum([margins[l] for l in top], axis=0)
    print(f"summed-margin val AUROC: {auroc(summed, yva):.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    for l in top:
        torch.save(models[l].state_dict(), OUT / f"prober_layer{l}.pt")
    (OUT / "config.json").write_text(json.dumps({
        "layers": top, "per_layer_val_auroc": {str(l): results[l] for l in results},
        "summed_val_auroc": auroc(summed, yva), "dim": DIM, "hidden": 256,
        "train_size": int(len(xtr)), "seed": SEED,
        "margin_semantics": "logit[no_retrieval] - logit[retrieval]; sum over layers; stop when > theta",
    }, indent=2))
    # persist val margins for theta tuning
    np.save(OUT / "val_summed_margin.npy", summed)
    np.save(OUT / "val_labels.npy", yva)
    print("->", OUT)


if __name__ == "__main__":
    main()
