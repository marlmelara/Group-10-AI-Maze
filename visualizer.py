from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

Position = Tuple[int, int]


def draw_maze_base(
    right_blocked: np.ndarray,
    down_blocked: np.ndarray,
    start: Position,
    goal: Position,
    death_pits: Optional[Set[Position]] = None,
    confusion_pads: Optional[Set[Position]] = None,
    teleport_map: Optional[Dict[Position, Position]] = None,
    path: Optional[List[Position]] = None,
    explored: Optional[Iterable[Position]] = None,
    save_path: Optional[str] = None,
    title: str = "Maze",
) -> None:
    n = right_blocked.shape[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("white")

    # Outer border
    ax.plot([0, n], [0, 0], color="black", linewidth=1.5)
    ax.plot([0, n], [n, n], color="black", linewidth=1.5)
    ax.plot([0, 0], [0, n], color="black", linewidth=1.5)
    ax.plot([n, n], [0, n], color="black", linewidth=1.5)

    # Internal vertical walls
    for r in range(n):
        for c in range(n - 1):
            if right_blocked[r, c]:
                x = c + 1
                y0 = n - (r + 1)
                y1 = n - r
                ax.plot([x, x], [y0, y1], color="black", linewidth=1.0)

    # Internal horizontal walls
    for r in range(n - 1):
        for c in range(n):
            if down_blocked[r, c]:
                y = n - (r + 1)
                x0 = c
                x1 = c + 1
                ax.plot([x0, x1], [y, y], color="black", linewidth=1.0)

    def cell_center(pos: Position) -> Tuple[float, float]:
        r, c = pos
        return (c + 0.5, n - (r + 0.5))

    if explored:
        xs, ys = zip(*[cell_center(p) for p in explored])
        ax.scatter(xs, ys, s=12, c="#d9ecff", marker="s", linewidths=0)

    if death_pits:
        xs, ys = zip(*[cell_center(p) for p in death_pits])
        ax.scatter(xs, ys, s=12, c="#ff9900", marker="s", linewidths=0)

    if confusion_pads:
        xs, ys = zip(*[cell_center(p) for p in confusion_pads])
        ax.scatter(xs, ys, s=12, c="#00bcd4", marker="s", linewidths=0)

    if teleport_map:
        xs, ys = zip(*[cell_center(p) for p in teleport_map.keys()])
        ax.scatter(xs, ys, s=12, c="#7b1fa2", marker="s", linewidths=0)

    if path:
        coords = [cell_center(p) for p in path]
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        ax.plot(xs, ys, color="red", linewidth=2.0)

    sx, sy = cell_center(start)
    gx, gy = cell_center(goal)
    ax.scatter([sx], [sy], s=36, c="green", marker="o")
    ax.scatter([gx], [gy], s=36, c="darkred", marker="o")

    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(-0.5, n + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches="tight")

    plt.show()