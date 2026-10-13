"""Pareto (Gemma-4 backbone): LLM generations/question vs F1 — companion to
scripts/plot_pareto.py (Qwen), same single-column style so the two figures pair.
Broken x-axis: competitive 1-5 gen region left, SeaKR alone right (17.5 gen).

PGIR frontier = offline replay of the tau=0.6/B=3 test traces (exact prefix
replay), from runs/tables/pgir_pareto_points_gemma.json; here the whole
trajectory B=1/2/3 is drawn (main config B=3). Baseline macro F1 = mean of the
six per-dataset F1 in runs/tables/gemma_trainfree_baselines.md.

Outputs runs/tables/pareto_f1_gen_gemma.{pdf,png}.
Run:  python scripts/plot_pareto_gemma.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.size": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE = "#C7376C"   # Rose Wine accent, same as the Qwen figure
GRAY = "#52514e"
INK = "#0b0b0b"
MUTED = "#8a8985"
SURFACE = "#ffffff"
GRID = "#e8e7e4"

OUT = "pgir/runs/tables/pareto_f1_gen_gemma"

# PGIR frontier: (gen/question, F1, tag) — replayed 2026-07-20 from the
# gemma4_pgir_*_test traces (pgir_pareto_points_gemma.json).
pgir = [(1.96, 0.240, "B=1"), (3.39, 0.383, "B=2"), (4.63, 0.396, "B=3")]
# baselines: name, gen/q, F1, label-x, label-y, ha, va
# vanilla RAG (.3507) and SKR (.3508) coincide at x=1 — one visible dot,
# two stacked labels.
baselines = [
    ("no-RAG",       1.00, 0.088, 1.06, 0.088, "left",  "center"),
    ("vanilla RAG",  1.00, 0.351, 1.06, 0.347, "left",  "top"),
    ("SKR",          1.00, 0.351, 1.06, 0.355, "left",  "bottom"),
    ("Adaptive-RAG", 1.70, 0.361, 1.70, 0.368, "center", "bottom"),
    ("DRAGIN",       2.64, 0.282, 2.64, 0.275, "center", "top"),
    # Probing-RAG (.3843) sits on top of PGIR B=2 (3.39, .3831): draw its dot
    # in FRONT of the pink marker (zorder bump below) so the near-tie is visible.
    ("Probing-RAG",  3.43, 0.384, 3.52, 0.373, "left",  "top"),
]
seakr = ("SeaKR", 17.48, 0.307)


def main():
    fig = plt.figure(figsize=(3.42, 2.85))
    fig.patch.set_facecolor(SURFACE)
    gs = GridSpec(1, 2, width_ratios=[7.0, 1.0], wspace=0.05)
    axL = fig.add_subplot(gs[0])
    axR = fig.add_subplot(gs[1], sharey=axL)

    for ax in (axL, axR):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=6.5)

    for name, x, y, lx, ly, ha, va in baselines:
        z = 6 if name == "Probing-RAG" else 3
        axL.scatter([x], [y], s=22, color=GRAY, zorder=z,
                    edgecolors=SURFACE, linewidths=1.0)
        axL.annotate(name, (lx, ly), ha=ha, va=va, fontsize=6.3, color=GRAY)

    px = [p[0] for p in pgir]; py = [p[1] for p in pgir]
    axL.plot(px, py, "-", color=BLUE, lw=1.5, zorder=4)
    axL.scatter(px, py, s=30, color=BLUE, zorder=5,
                edgecolors=SURFACE, linewidths=1.1)
    for x, y, tag in pgir:
        # keep the B=2 tag clear of the Probing-RAG dot sitting on that marker
        dx = -0.10 if tag == "B=2" else 0.0
        axL.annotate(tag, (x + dx, y + 0.008), ha="center", va="bottom",
                     fontsize=6.3, color=BLUE, fontweight="bold")
    bx, by, _ = pgir[-1]
    axL.annotate("main config", xy=(bx, by), xytext=(bx + 0.28, by - 0.044),
                 ha="center", va="top", fontsize=6.3, color=BLUE,
                 fontstyle="italic",
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.8,
                                 shrinkA=1.5, shrinkB=3))
    axL.annotate("PGIR", (2.45, 0.318), ha="center", va="bottom",
                 fontsize=8, color=BLUE, fontweight="bold")

    axL.set_xlim(0.6, 5.35)
    axL.set_ylim(0.05, 0.435)
    axL.set_xticks([1, 2, 3, 4, 5])
    axL.spines["right"].set_visible(False)
    axL.spines[["left", "bottom", "top"]].set_color(MUTED)

    name, x, y = seakr
    axR.scatter([x], [y], s=22, color=GRAY, zorder=3,
                edgecolors=SURFACE, linewidths=1.0)
    axR.annotate(name, (x, y + 0.008), ha="center", va="bottom",
                 fontsize=6.3, color=GRAY)
    axR.set_xlim(16.6, 18.4)
    axR.set_xticks([17.5])
    axR.set_xticklabels(["17.5"])
    axR.spines["left"].set_visible(False)
    axR.spines[["bottom", "top", "right"]].set_color(MUTED)
    axR.tick_params(axis="y", left=False, labelleft=False)

    d = 0.02
    axL.plot([1 - d, 1 + d], [-d, d], transform=axL.transAxes,
             color=MUTED, clip_on=False, lw=1.0)
    axL.plot([1 - d, 1 + d], [1 - d, 1 + d], transform=axL.transAxes,
             color=MUTED, clip_on=False, lw=1.0)
    axR.plot([-d * 7.0, d * 7.0], [-d, d], transform=axR.transAxes,
             color=MUTED, clip_on=False, lw=1.0)
    axR.plot([-d * 7.0, d * 7.0], [1 - d, 1 + d], transform=axR.transAxes,
             color=MUTED, clip_on=False, lw=1.0)

    axL.set_ylabel("F1 score", fontsize=7.5, color=INK, labelpad=3)
    fig.text(0.5, 0.02, "LLM generations per question",
             ha="center", fontsize=7.5, color=INK)

    fig.subplots_adjust(left=0.155, right=0.985, top=0.98, bottom=0.15)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", facecolor=SURFACE, dpi=300)
    print(f"saved {OUT}.pdf + .png")


if __name__ == "__main__":
    main()
