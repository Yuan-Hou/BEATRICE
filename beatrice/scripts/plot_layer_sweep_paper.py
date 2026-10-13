"""Paper-final layer-sweep figure (single closed-book curve).

Reproduces the exact figure logic used for the AAAI submission
(Figures/layer_sweep.pdf, commit f959cc7): the closed-book (noctx)
saplma_mean *validation* AUROC curve, one line, with the selected layer
marked by a dot + dashed vline. Layer selection criterion = argmax of
that same curve (the script also reports the test_id argmax so agreement
can be checked before the caption claims it).

Usage:
  python scripts/plot_layer_sweep_paper.py \
      --curves outputs/reports/layer_sweep_curves.json \
      --out-pdf ~/paper/Figures/layer_sweep.pdf \
      --out-png outputs/reports/layer_sweep_final.png
  # Gemma:
  python scripts/plot_layer_sweep_paper.py \
      --curves outputs/reports/gemma_layer_sweep_curves.json \
      --out-png outputs/reports/gemma_layer_sweep_fig.png
"""
import argparse
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "font.family": "serif", "mathtext.fontset": "stix",
    "font.size": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
})
import matplotlib.pyplot as plt

BLUE, ROSE, GRID = "#0A96DB", "#C7376C", "#e8e7e4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--out-pdf", default=None)
    ap.add_argument("--out-png", default=None)
    ap.add_argument("--layer", type=int, default=None,
                    help="marked layer; default = val argmax of noctx saplma_mean")
    ap.add_argument("--ylim", default=None, help="e.g. 0.58,0.82; default auto")
    args = ap.parse_args()

    d = json.load(open(args.curves))
    val = np.array(d["noctx"]["saplma_mean"]["validation"], dtype=float)
    test = np.array(d["noctx"]["saplma_mean"]["test_id"], dtype=float)
    L = np.arange(len(val))

    k_val, k_test = int(val.argmax()), int(test.argmax())
    k = args.layer if args.layer is not None else k_val
    print(f"noctx saplma_mean: val argmax L{k_val} ({val[k_val]:.4f}), "
          f"test argmax L{k_test} ({test[k_test]:.4f}); marking L{k} ({val[k]:.4f})")
    if k_val != k_test:
        print("WARNING: val/test argmax disagree -- caption's "
              "'(also on held-out test)' claim needs revisiting")

    if args.ylim:
        y0, y1 = (float(x) for x in args.ylim.split(","))
    else:
        y0 = np.floor((val.min() - 0.01) * 50) / 50
        y1 = np.ceil((val.max() + 0.01) * 50) / 50

    fig, ax = plt.subplots(figsize=(3.35, 1.25))
    ax.plot(L, val, color=BLUE, lw=1.1)
    ax.scatter([k], [val[k]], s=11, color=ROSE, zorder=5)
    ax.axvline(k, color=ROSE, lw=0.7, ls=(0, (3, 2)), alpha=0.85)
    ax.text(k - 0.6, y0 + 0.17 * (y1 - y0), f"L{k}", color=ROSE,
            fontsize=6.5, ha="right", va="bottom")
    ax.set_xlabel("layer")
    ax.set_ylabel("val AUROC")
    ax.set_xlim(0, len(L) - 1)
    ax.set_ylim(y0, y1)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.subplots_adjust(left=0.13, right=0.985, top=0.95, bottom=0.25)
    for out in (args.out_pdf, args.out_png):
        if out:
            fig.savefig(out, **({"dpi": 200} if out.endswith(".png") else {}))
            print(f"saved {out}")


if __name__ == "__main__":
    main()
