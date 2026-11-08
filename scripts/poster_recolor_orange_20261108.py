"""Re-draw the section-7 poster figures in the poster's palette (Amazon-style orange, greys, navy text).

Outputs <name>_orange.png/.svg next to the originals in results/figures/poster_20261108/:
fig15 funnel flow, fig19 waffles, fig17 heatmap, fig21 upgrade list, fig18 staircase, fig22 future titles, fig12 iPhone 13 launch.
Best system = orange, original = grey, 'old phone' tags = dark grey, highlight boxes = peach, text = navy.

Usage: .venv/bin/python scripts/poster_recolor_orange_20261108.py
"""
from __future__ import annotations
import shutil, sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import poster_diagrams_20261108 as D  # noqa: E402
import poster_launch_decay_20261108 as L  # noqa: E402

ORANGE, GREY, DARK, PEACH, NAVY = "#E8710A", "#9A9A9A", "#5F5F5F", "#F3A15C", "#232F3E"
for m in (D, L):
    m.BEST, m.ORIG, m.INK = ORANGE, GREY, NAVY
    m.plt.rcParams.update({"axes.labelcolor": NAVY})
D.REC, D.WARN, D.DEEP = DARK, PEACH, "#8A3B00"


def save(fig, name):
    fig.savefig(D.OUT / f"{name}_orange.png", dpi=300, bbox_inches="tight"); fig.savefig(D.OUT / f"{name}_orange.svg", bbox_inches="tight"); plt.close(fig)


D.save = save
for f in (D.fig19, D.fig17, D.fig21, D.fig18, D.fig22, D.fig15):
    f()
tmp = L.OUT / "_orange_tmp"; tmp.mkdir(exist_ok=True); L.OUT = tmp
L.WARN_BG = "#fde3cc"
L.main()
for ext in ("png", "svg"):
    shutil.move(tmp / f"fig12_iphone13_launch.{ext}", D.OUT / f"fig12_iphone13_launch_orange.{ext}")
tmp.rmdir()
print("done")
