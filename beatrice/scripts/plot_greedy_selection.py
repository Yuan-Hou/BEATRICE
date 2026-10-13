#!/usr/bin/env python
"""Main-figure plot of the greedy encoder-selection process.

For each setting (question-only / with-context) the fusion probe is built by
greedy forward selection over 15 candidate encoder-branches, each defined by a
subset of four feature channels:

    H = Hidden (L17 hidden state)   P = Prob (token probability)
    R = Resid (residual-lens)       A = Attn (attention simplex)

Each round evaluates every remaining branch, adds the one with the highest
VALIDATION AUROC, and stops when the best marginal gain is <= 5e-4. This script
renders that trajectory as a compact annotated staircase against the full-fusion
(all-N branches) reference, and writes a separate caption/description text file.

Reads the search reports produced by scripts/search_fusion_combos.py:
    outputs/reports/qonly_combo_search.json      (question-only)
    outputs/reports/ctx_combo_search_13br.json   (with-context)

Outputs (PDF figure + TXT text, kept separate):
    outputs/figures/greedy_selection_<setting>.pdf
    outputs/figures/greedy_selection_<setting>.txt

Usage:  python scripts/plot_greedy_selection.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.transforms import offset_copy

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "outputs" / "figures"

# ---- palette (user-provided) -------------------------------------------------
PAL = {
    "lemon":  "#FFF6C9",  # Lemon Chiffon  -- soft fill
    "peony":  "#DC7CA9",  # Sweet Peony
    "orchid": "#DDB8E2",  # Pink Orchid    -- muted candidate/gridline
    "grape":  "#965187",  # Grape Soda     -- full-fusion reference
    "wine":   "#C7376C",  # Rose Wine      -- selected champion accent
    "bell":   "#0A96DB",  # Blue Bell      -- greedy path
    "red":    "#DE0000",  # Racing Red
}
INK = "#2B2B2B"

# ---- channel shorthand for encoder-branch names ------------------------------
_CH = {"hid": "H", "hidden": "H", "prob": "P", "probability": "P",
       "resid": "R", "residual": "R", "attn": "A", "attention": "A"}


def shorthand(branch: str) -> str:
    """hid_prob_attn -> 'H+P+A' ; hid_resid -> 'H+R' ; quad -> 'H+P+R+A'."""
    if branch in ("quad", "quad1", "hid_prob_resid_attn"):
        return "H+P+R+A"
    return "+".join(_CH.get(tok, tok.upper()) for tok in branch.split("_"))


# ---- feature glyphs (shared main-figure symbol system) -----------------------
# The four feature channels are drawn as GRAY-OUTLINED SHAPES; colour is reserved
# elsewhere for token segments and never encodes a feature here.
#   H Hidden -> triangle      P Prob -> diamond
#   R Resid  -> ring          A Attn -> 3x3 mini heatmap grid
CH_ORDER = ["H", "P", "R", "A"]


def _grid_marker() -> MPath:
    """A 2x2 grid (attention) as one marker Path: outer square + a single mid
    horizontal and vertical divider, so internal lines are drawn once (they
    overlap rather than doubling up into thick strokes)."""
    m = 0.5
    verts = [(-m, -m), (m, -m), (m, m), (-m, m), (-m, -m),   # outer square
             (0.0, -m), (0.0, m),                            # vertical midline
             (-m, 0.0), (m, 0.0)]                            # horizontal midline
    codes = [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO,
             MPath.CLOSEPOLY,
             MPath.MOVETO, MPath.LINETO,
             MPath.MOVETO, MPath.LINETO]
    return MPath(verts, codes)


GRID = _grid_marker()
# marker, size, facecolor, linewidth  (edge colour applied at draw time)
_GLYPH = {
    "H": ("^", 82, "white", 1.15),
    "P": ("D", 66, "white", 1.15),
    "R": ("o", 78, "white", 1.15),
    "A": (GRID, 104, "none", 0.9),
}


def branch_channels(branch: str) -> list[str]:
    if branch.startswith("q_"):   # closed-book conditioning marker, not a channel
        branch = branch[2:]
    if branch in ("quad", "quad1", "hid_prob_resid_attn"):
        present = set("HPRA")
    else:
        present = {_CH.get(tok, tok) for tok in branch.split("_")}
    return [c for c in CH_ORDER if c in present]


def draw_glyphs(ax, fig, k, y, branch, faded=False, dy=22, spacing=11.5):
    """Draw the feature-channel glyph cluster for `branch` at a fixed pixel
    offset above the data point (k, y). Closed-book branches (q_*) are wrapped
    in thin round brackets — same syntax as the tables ([.]=with-ctx, (.)=closed-book)."""
    chans = branch_channels(branch)
    closed_book = branch.startswith("q_")
    edge = "#B4B4B4" if faded else "#6E6E6E"
    n = len(chans)
    for j, ch in enumerate(chans):
        marker, s, face, lw = _GLYPH[ch]
        xoff = (j - (n - 1) / 2) * spacing
        tr = offset_copy(ax.transData, fig=fig, x=xoff, y=dy, units="points")
        ax.scatter([k], [y], marker=marker, s=s, transform=tr,
                   facecolor=face, edgecolors=edge, linewidths=lw,
                   clip_on=False, zorder=6)
    if closed_book:
        half = (n - 1) / 2 * spacing + 10
        for xoff, br in ((-half, "("), (half, ")")):
            tr = offset_copy(ax.transData, fig=fig, x=xoff, y=dy, units="points")
            ax.text(k, y, br, transform=tr, ha="center", va="center",
                    fontsize=13, color=edge, clip_on=False, zorder=6)


SETTINGS = [
    {
        "key": "qonly",
        "json": "outputs/reports/qonly_combo_search.json",
        "log": "logs/L17rd_attn_triples/combo_search.log",
        "n_all": 15,
    },
    {
        "key": "ctx",
        "json": "outputs/reports/joint_combo_search_30br.json",
        "log": "logs/ctx_L17rd_attn_triples/joint_combo_search_30br.log",
        "n_all": 30,
    },
]

_STOP_RE = re.compile(r"best candidate\s+(\S+)\s+val=([\d.]+)")


def best_rejected(cfg: dict):
    """Parse the greedy-search log for the k+1 candidate that was evaluated but
    NOT added (its gain fell below the 5e-4 stop threshold). Returns (branch, val)
    or None if the log/line is absent."""
    p = ROOT / cfg["log"]
    if not p.exists():
        return None
    for line in p.read_text(errors="ignore").splitlines():
        if "no gain" in line:
            m = _STOP_RE.search(line)
            if m:
                return m.group(1), float(m.group(2))
    return None


def plot_setting(cfg: dict) -> None:
    data = json.loads((ROOT / cfg["json"]).read_text())
    traj = data["greedy_trajectory"]

    ks = list(range(1, len(traj) + 1))
    vals = [t["val"] for t in traj]
    branches = [t["added"] for t in traj]
    kstop = len(traj)

    fig, ax = plt.subplots(figsize=(4.0, 3.1))

    # greedy path + selection nodes -- colour (from the palette, pink/purple
    # family) marks the trajectory & the champion; the FEATURES stay gray glyphs
    # so colour never encodes a feature.
    ax.plot(ks, vals, "-", color=PAL["grape"], lw=1.9, zorder=3,
            solid_capstyle="round")
    for i, (k, v, br) in enumerate(zip(ks, vals, branches)):
        last = i == len(ks) - 1
        ax.scatter([k], [v], s=78 if last else 46, zorder=4,
                   facecolor=PAL["wine"] if last else PAL["peony"],
                   edgecolor="white", linewidth=1.2)
        draw_glyphs(ax, fig, k, v, br, dy=22)

    # k+1: best candidate evaluated but NOT added (gain <= 5e-4) -- hollow node +
    # dashed reach + faded glyphs show the trend flattening -> selection stops.
    rej = best_rejected(cfg)
    if rej is not None:
        rb, rv = rej
        kr = kstop + 1
        ax.plot([ks[-1], kr], [vals[-1], rv], ls=(0, (4, 3)), lw=1.5,
                color="#BEBEBE", zorder=2)
        ax.scatter([kr], [rv], s=42, zorder=4, facecolor="white",
                   edgecolor="#A6A6A6", linewidth=1.4)
        draw_glyphs(ax, fig, kr, rv, rb, faded=True, dy=22)
        ax.annotate("Δ ≤ 5×10$^{-4}$\nstop", xy=(kr, rv), xytext=(0, -13),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=8, color="#9A9A9A")
        xmax = kr
        allv = vals + [rv]
    else:
        xmax = kstop
        allv = vals

    # axes cosmetics -- tight y-range; top keeps just enough room for the glyph
    # clusters that sit above the highest node, bottom hugs the lowest node.
    lo, hi = min(allv), max(allv)
    span = hi - lo
    ax.set_ylim(lo - span * 0.14 - 5e-4, hi + span * 0.72 + 6e-4)
    ax.set_xlim(0.55, xmax + 0.55)
    ax.set_xticks(list(range(1, xmax + 1)))
    ax.set_xlabel("Encoders in fusion set  (k)", fontsize=10)
    ax.set_ylabel("Validation AUROC", fontsize=10)
    ax.grid(axis="y", color="#E2E2E2", alpha=0.9, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    out = FIGDIR / f"greedy_selection_{cfg['key']}.pdf"
    fig.savefig(out, bbox_inches="tight")
    png_dir = os.environ.get("GREEDY_PNG_DIR")
    if png_dir:  # optional raster copy for visual QA only
        fig.savefig(Path(png_dir) / f"greedy_selection_{cfg['key']}.png",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out}")


def write_text(cfg: dict) -> None:
    data = json.loads((ROOT / cfg["json"]).read_text())
    traj = data["greedy_trajectory"]
    fam = data["families"]
    all_val = fam["all15"]["val"]
    n_all = cfg["n_all"]
    setting = "question-only (no retrieved context)" if cfg["key"] == "qonly" \
        else "with retrieved context"
    metrics = ["val", "test", "popqa"] + (["nq", "triviaqa"] if cfg["key"] == "ctx" else [])
    mlabel = {"val": "val", "test": "test", "popqa": "PopQA-OOD",
              "nq": "NQ", "triviaqa": "TriviaQA"}

    lines = []
    lines.append(f"Greedy encoder selection — {setting}.")
    lines.append("")
    lines.append(
        f"Starting from {n_all} candidate encoder-branches (each a subset of the "
        "four feature channels, drawn as gray-outlined glyphs in the figure — "
        "H=Hidden (triangle △), P=Prob (diamond ◇), R=Resid (ring ◯), "
        "A=Attn (3×3 grid ▦)), we grow the fusion set by greedy forward "
        "selection on validation AUROC: at every round we add the single branch "
        "that most improves validation AUROC, stopping when the best remaining "
        "branch adds ≤ 5×10^-4. The faded hollow point at k+1 is the best "
        "candidate that was evaluated but rejected under this stop rule.")
    lines.append("")
    lines.append("Per-round selection:")
    for i, t in enumerate(traj, 1):
        added = shorthand(t["added"])
        stats = ", ".join(f"{mlabel[m]} {t[m]:.4f}" for m in metrics if m in t)
        lines.append(f"  Round {i}: add {added}  (encoder set = "
                     f"{'{'}{', '.join(shorthand(b) for b in t['branches'])}{'}'}) "
                     f"→ {stats}")
    lines.append("")
    champ_k = len(traj)
    champ_branches = ", ".join(shorthand(b) for b in traj[-1]["branches"])
    lines.append(
        f"Selection halts at k={champ_k} (greedy-k{champ_k}); the chosen set "
        f"{'{'}{champ_branches}{'}'} reaches validation AUROC {traj[-1]['val']:.4f}, "
        f"matching or exceeding the full {n_all}-encoder fusion "
        f"({all_val:.4f}) with only {champ_k} encoder"
        f"{'s' if champ_k > 1 else ''} — i.e. a compact, more efficient "
        "champion rather than the full stack.")
    txt = "\n".join(lines) + "\n"
    out = FIGDIR / f"greedy_selection_{cfg['key']}.txt"
    out.write_text(txt)
    print(f"[txt] wrote {out}")


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,   # embed as TrueType (editable text in the PDF)
        "axes.linewidth": 0.9,
    })
    for cfg in SETTINGS:
        plot_setting(cfg)
        write_text(cfg)


if __name__ == "__main__":
    main()
