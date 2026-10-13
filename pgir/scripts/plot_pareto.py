"""Pareto: efficiency (LLM generations / question) vs F1 — the PGIR sub-experiment
figure. Sized for a SINGLE-COLUMN slot (~3.4in wide) in the paper. Broken x-axis:
the 1-4 gen competitive region gets the width, SeaKR sits alone in a narrow right
panel (16.6 gen cost outlier). No title. PGIR frontier tagged B=2/B=3/B=4 with
'main config' by B=4; labels hug their marks.

Outputs runs/tables/pareto_f1_gen.{pdf,png}. Run:  python scripts/plot_pareto.py
Edit the `pgir` / `baselines` / `seakr` tables below to refresh numbers.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.size": 7,
    "pdf.fonttype": 42,   # embed TrueType (editor-friendly, no Type-3)
    "ps.fonttype": 42,
})

BLUE = "#C7376C"   # 'Rose Wine' accent for the PGIR protagonist (user palette)
GRAY = "#52514e"
INK = "#0b0b0b"
MUTED = "#8a8985"
SURFACE = "#ffffff"
GRID = "#e8e7e4"

OUT = "pgir/runs/tables/pareto_f1_gen"

# PGIR frontier: (gen/question, F1, tag) — m7_fused (dual q+ctx fused probe,
# 2026-07-10); refresh from runs/tables/pgir_pareto_points.json. B=1 anchor
# (closed-book only, F1 .3135 @ 1.00 gen) recorded in the json, not plotted.
pgir = [(1.93, 0.464, "B=2"), (2.76, 0.481, "B=3"), (3.56, 0.484, "B=4")]
# baselines in the main region: name, gen/q, F1, label-x, label-y, ha, va
# label offsets kept small so each word hugs its mark.
baselines = [
    ("no-RAG",       1.00, 0.303, 1.05, 0.303, "left",  "center"),
    ("vanilla RAG",  1.00, 0.451, 1.05, 0.454, "left",  "bottom"),
    ("SKR",          1.00, 0.445, 1.05, 0.442, "left",  "top"),
    ("Adaptive-RAG", 1.63, 0.424, 1.63, 0.418, "center","top"),
    ("Probing-RAG",  2.56, 0.462, 2.56, 0.455, "center","top"),
    ("DRAGIN",       3.55, 0.425, 3.48, 0.425, "right", "center"),
]
seakr = ("SeaKR", 13.63, 0.434)  # cost outlier -> right panel (faithful repro, 2026-07-10)


def main():
    fig = plt.figure(figsize=(3.42, 2.85))   # single-column width
    fig.patch.set_facecolor(SURFACE)
    gs = GridSpec(1, 2, width_ratios=[7.0, 1.0], wspace=0.05)
    axL = fig.add_subplot(gs[0])
    axR = fig.add_subplot(gs[1], sharey=axL)

    for ax in (axL, axR):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=6.5)

    # left panel: main competitive region
    for name, x, y, lx, ly, ha, va in baselines:
        axL.scatter([x], [y], s=22, color=GRAY, zorder=3,
                    edgecolors=SURFACE, linewidths=1.0)
        axL.annotate(name, (lx, ly), ha=ha, va=va, fontsize=6.3, color=GRAY)

    px = [p[0] for p in pgir]; py = [p[1] for p in pgir]
    axL.plot(px, py, "-", color=BLUE, lw=1.5, zorder=4)
    axL.scatter(px, py, s=30, color=BLUE, zorder=5,
                edgecolors=SURFACE, linewidths=1.1)
    for x, y, tag in pgir:
        axL.annotate(tag, (x, y + 0.007), ha="center", va="bottom",
                     fontsize=6.3, color=BLUE, fontweight="bold")
    bx, by, _ = pgir[-1]   # B=4 is the main config -> point at it explicitly
    axL.annotate("main config", xy=(bx, by), xytext=(bx - 0.02, by - 0.028),
                 ha="center", va="top", fontsize=6.3, color=BLUE,
                 fontstyle="italic",
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.8,
                                 shrinkA=1.5, shrinkB=3))
    # 'PGIR' hugs the curve, above the B2-B3 segment
    axL.annotate("PGIR", (2.28, 0.474), ha="center", va="bottom",
                 fontsize=8, color=BLUE, fontweight="bold")

    axL.set_xlim(0.6, 4.0)
    axL.set_ylim(0.29, 0.495)
    axL.set_xticks([1, 2, 3, 4])
    axL.spines["right"].set_visible(False)
    axL.spines[["left", "bottom", "top"]].set_color(MUTED)

    # right panel: SeaKR outlier
    name, x, y = seakr
    axR.scatter([x], [y], s=22, color=GRAY, zorder=3,
                edgecolors=SURFACE, linewidths=1.0)
    axR.annotate(name, (x, y + 0.007), ha="center", va="bottom",
                 fontsize=6.3, color=GRAY)
    axR.set_xlim(12.9, 14.3)
    axR.set_xticks([13.6])
    axR.set_xticklabels(["13.6"])
    axR.spines["left"].set_visible(False)
    axR.spines[["bottom", "top", "right"]].set_color(MUTED)
    axR.tick_params(axis="y", left=False, labelleft=False)

    # diagonal break marks between panels (bottom and top edges)
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
