"""Side-by-side visual sanity check of the parser output vs the source PNG."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maze_parser import ParsedMaze, GRID_SIZE


def _cell_center(pos, n=GRID_SIZE):
    r, c = pos
    return (c + 0.5, n - (r + 0.5))


def _draw_walls(ax, parsed: ParsedMaze):
    n = GRID_SIZE
    ax.plot([0, n, n, 0, 0], [0, 0, n, n, 0], color="black", lw=1.2)
    for r in range(n):
        for c in range(n - 1):
            if parsed.right_blocked[r, c]:
                x = c + 1
                ax.plot([x, x], [n - (r + 1), n - r], color="black", lw=0.6)
    for r in range(n - 1):
        for c in range(n):
            if parsed.down_blocked[r, c]:
                y = n - (r + 1)
                ax.plot([c, c + 1], [y, y], color="black", lw=0.6)


def plot(parsed: ParsedMaze, hazards_png: str, out_path: str):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    img = Image.open(hazards_png)
    axs[0].imshow(img)
    axs[0].set_title(f"{parsed.name} - source hazards.png")
    axs[0].axis("off")

    ax = axs[1]
    _draw_walls(ax, parsed)

    # Fires
    for g_idx, group in enumerate(parsed.fire_groups):
        xs, ys = zip(*[_cell_center(p) for p in group])
        ax.scatter(xs, ys, s=22, c="orangered", marker="s", zorder=3)
        pivot = group[0]
        px, py = _cell_center(pivot)
        ax.scatter([px], [py], s=55, facecolors="none", edgecolors="black", lw=1.2, zorder=4)

    # Confusion
    if parsed.confusion_pads:
        xs, ys = zip(*[_cell_center(p) for p in parsed.confusion_pads])
        ax.scatter(xs, ys, s=26, c="yellow", edgecolors="black", marker="o", zorder=3)

    # Teleport sources and destinations
    for src, dst in zip(parsed.teleport_sources, parsed.teleport_dests):
        sx, sy = _cell_center(src)
        dx, dy = _cell_center(dst)
        ax.scatter([sx], [sy], s=45, c="purple", marker="^", zorder=4)
        ax.scatter([dx], [dy], s=45, c="magenta", marker="v", zorder=4)
        ax.plot([sx, dx], [sy, dy], color="purple", alpha=0.3, lw=0.8, zorder=2)

    # Wind arrows
    if parsed.wind_cells:
        for pos, direction in parsed.wind_cells.items():
            x, y = _cell_center(pos)
            dx, dy = {"UP": (0, 0.35), "DOWN": (0, -0.35),
                      "LEFT": (-0.35, 0), "RIGHT": (0.35, 0)}[direction]
            ax.arrow(x - dx / 2, y - dy / 2, dx, dy, head_width=0.2,
                     color="blue", length_includes_head=True, zorder=4)

    sx, sy = _cell_center(parsed.start)
    gx, gy = _cell_center(parsed.goal)
    ax.scatter([sx], [sy], s=80, c="limegreen", marker="o", zorder=5)
    ax.scatter([gx], [gy], s=80, c="red", marker="*", zorder=5)

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal")
    ax.set_title(f"{parsed.name} - parsed structure")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("alpha", "beta", "gamma"):
        parsed = ParsedMaze.load(os.path.join(here, "data", name, "parsed.npz"))
        plot(parsed, os.path.join(here, "data", name, "hazards.png"),
             os.path.join(here, "figures", f"parse_check_{name}.png"))
