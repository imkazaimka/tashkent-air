"""
Conceptual schematics (not data plots) to explain the two mechanisms under test:
(a) the wind-direction transport quasi-experiment, and (b) boundary-layer trapping.

Run:  python src/figures_schematic.py   ->  figures/fig_schematic.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse, Rectangle, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

FIG = C.ROOT / "figures"
NAVY, ACCENT, COOL, WARM = "#16314f", "#c0392b", "#2980b9", "#e8a33d"
RNG = np.random.default_rng(7)

plt.rcParams.update({"font.family": "Helvetica Neue, Arial, sans-serif",
                     "font.size": 10})


def panel_transport(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off"); ax.set_aspect("equal")
    ax.set_title("(a)  Regional transport — the wind-direction test",
                 fontweight="bold", color=NAVY, fontsize=12, loc="left")

    # upwind polluted region (E/NE)
    ax.add_patch(Ellipse((7.7, 5.7), 3.4, 2.0, color=ACCENT, alpha=0.16))
    ax.add_patch(Rectangle((7.3, 5.4), 0.7, 0.7, color="#5b6470"))
    ax.text(7.7, 6.7, "upwind city + air mass\n(Fergana / Almaty / Bishkek)",
            ha="center", va="bottom", fontsize=9, color="#5b6470")

    # Tashkent
    ax.plot(2.6, 3.0, "*", ms=30, color=WARM, mec=NAVY, mew=1.3, zorder=5)
    ax.text(2.6, 2.2, "Tashkent", ha="center", fontsize=10.5,
            fontweight="bold", color=NAVY)

    # inbound wind = transport
    ax.add_patch(FancyArrowPatch((6.9, 5.1), (3.4, 3.4), arrowstyle="-|>",
                 mutation_scale=26, lw=3.2, color=ACCENT, zorder=4))
    ax.text(5.2, 4.7, "wind FROM the city\n→ pollution arrives", ha="center",
            fontsize=9.5, color=ACCENT, fontweight="bold")

    # outbound wind = control (no effect)
    ax.add_patch(FancyArrowPatch((3.6, 2.4), (6.6, 1.2), arrowstyle="-|>",
                 mutation_scale=18, lw=1.6, color="#9aa3ad", ls="--", zorder=3))
    ax.text(5.2, 1.0, "wind TOWARD the city → no effect (control)",
            ha="center", fontsize=8.6, color="#7a828c")

    # compass
    ax.annotate("N", xy=(0.7, 7.4), ha="center", fontsize=9, color=NAVY)
    ax.add_patch(FancyArrowPatch((0.7, 6.6), (0.7, 7.2), arrowstyle="-|>",
                 mutation_scale=12, color=NAVY))

    ax.text(5.0, 0.1, "Only genuine transport makes Tashkent dirtier on inbound-wind "
            "days; shared weather cannot.", ha="center", fontsize=8.4,
            style="italic", color="#5b6470")


def panel_dispersion(ax):
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("(b)  Atmospheric dispersion — the boundary-layer lid",
                 fontweight="bold", color=NAVY, fontsize=12, loc="left")

    def scene(x0, x1, lid, label, dense, lidlabel, col):
        ax.plot([x0, x1], [1, 1], color="#5b6470", lw=2)                 # ground
        ax.add_patch(Rectangle((x0, 1), x1 - x0, lid - 1, color=col, alpha=0.18))
        ax.plot([x0, x1], [lid, lid], ls="--", lw=1.6, color=NAVY)       # the lid
        ax.text((x0 + x1) / 2, lid + 0.15, lidlabel, ha="center", fontsize=8.4,
                color=NAVY)
        # pollution particles
        n = 60 if dense else 26
        ymax = 1.7 if dense else lid - 0.2
        xs = RNG.uniform(x0 + 0.3, x1 - 0.3, n)
        ys = RNG.uniform(1.1, ymax, n)
        ax.scatter(xs, ys, s=12, color="#6b4a2b", alpha=0.8, zorder=4)
        # emission source (house + chimney)
        hx = (x0 + x1) / 2
        ax.add_patch(Rectangle((hx - 0.4, 1), 0.8, 0.6, color="#8a8f98"))
        ax.add_patch(Rectangle((hx + 0.1, 1.6), 0.18, 0.5, color="#8a8f98"))
        ax.text((x0 + x1) / 2, 0.45, label, ha="center", fontsize=9.2,
                fontweight="bold", color=NAVY)
        return ymax

    # summer: deep layer, diluted
    scene(0.6, 5.4, 6.6, "Summer: deep mixing\n→ diluted (low PM2.5)", False,
          "mixing height ≈ 1–2 km", COOL)
    for x in (1.5, 3.0, 4.5):                                            # convection
        ax.add_patch(FancyArrowPatch((x, 1.6), (x, 5.6), arrowstyle="-|>",
                     mutation_scale=10, color=COOL, alpha=0.6))
    ax.add_patch(plt.Circle((1.2, 6.9), 0.35, color=WARM))              # sun

    # winter: shallow inversion, trapped
    scene(6.6, 11.4, 3.0, "Winter: shallow inversion\n→ trapped (high PM2.5)", True,
          "inversion — warm air aloft", "#7f8c8d")
    ax.annotate("warm air", xy=(9.0, 3.4), fontsize=8, color=ACCENT, ha="center")


def main():
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    panel_transport(ax[0])
    panel_dispersion(ax[1])
    fig.suptitle("Conceptual framework: the two mechanisms tested",
                 fontweight="bold", fontsize=13, color=NAVY, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig_schematic.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/fig_schematic.png")


if __name__ == "__main__":
    main()
