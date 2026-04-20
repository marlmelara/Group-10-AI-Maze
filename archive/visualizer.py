from typing import Dict, Iterable, List, Optional, Set, Tuple
import time

import matplotlib.pyplot as plt
import numpy as np

Position = Tuple[int, int]


def _cell_center(pos: Position, n: int) -> Tuple[float, float]:
    r, c = pos
    return (c + 0.5, n - (r + 0.5))


def _draw_static_maze(
    ax,
    right_blocked: np.ndarray,
    down_blocked: np.ndarray,
    start: Position,
    goal: Position,
    death_pits: Optional[Set[Position]] = None,
    confusion_pads: Optional[Set[Position]] = None,
    teleport_map: Optional[Dict[Position, Position]] = None,
    explored: Optional[Iterable[Position]] = None,
) -> int:
    """
    Draw the maze walls and static markers onto an axes.
    Returns maze size n.
    """
    n = right_blocked.shape[0]
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

    if explored:
        explored_list = list(explored)
        if explored_list:
            xs, ys = zip(*[_cell_center(p, n) for p in explored_list])
            ax.scatter(xs, ys, s=12, c="#d9ecff", marker="s", linewidths=0, zorder=2)

    if death_pits:
        pit_list = list(death_pits)
        if pit_list:
            xs, ys = zip(*[_cell_center(p, n) for p in pit_list])
            ax.scatter(xs, ys, s=18, c="#ff9900", marker="s", linewidths=0, zorder=3)

    if confusion_pads:
        confusion_list = list(confusion_pads)
        if confusion_list:
            xs, ys = zip(*[_cell_center(p, n) for p in confusion_list])
            ax.scatter(xs, ys, s=18, c="#00bcd4", marker="s", linewidths=0, zorder=3)

    if teleport_map:
        tele_list = list(teleport_map.keys())
        if tele_list:
            xs, ys = zip(*[_cell_center(p, n) for p in tele_list])
            ax.scatter(xs, ys, s=18, c="#7b1fa2", marker="s", linewidths=0, zorder=3)

    # Start / Goal
    sx, sy = _cell_center(start, n)
    gx, gy = _cell_center(goal, n)
    ax.scatter([sx], [sy], s=40, c="green", marker="o", zorder=5)
    ax.scatter([gx], [gy], s=40, c="darkred", marker="o", zorder=5)

    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(-0.5, n + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    return n


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
    fig, ax = plt.subplots(figsize=(8, 8))

    n = _draw_static_maze(
        ax=ax,
        right_blocked=right_blocked,
        down_blocked=down_blocked,
        start=start,
        goal=goal,
        death_pits=death_pits,
        confusion_pads=confusion_pads,
        teleport_map=teleport_map,
        explored=explored,
    )

    if path:
        coords = [_cell_center(p, n) for p in path]
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        ax.plot(xs, ys, color="red", linewidth=2.0, zorder=4)

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches="tight")

    plt.show()


def animate_path(
    right_blocked: np.ndarray,
    down_blocked: np.ndarray,
    start: Position,
    goal: Position,
    path: List[Position],
    death_pits: Optional[Set[Position]] = None,
    confusion_pads: Optional[Set[Position]] = None,
    teleport_map: Optional[Dict[Position, Position]] = None,
    explored: Optional[Iterable[Position]] = None,
    title: str = "Maze Animation",
    delay: float = 0.03,
    save_path: Optional[str] = None,
    reverse: bool = False,
    close_to_skip: bool = True,
) -> bool:
    """
    Live animation of an agent traversing a path through the maze.

    Parameters
    ----------
    delay:
        Time in seconds between animation frames. Smaller = faster.
    reverse:
        If True, display the path in reverse order.
    close_to_skip:
        If True, closing the figure immediately stops the animation.

    Returns
    -------
    bool
        True if the animation completed, False if it was interrupted/closed.
    """
    if not path:
        return True

    display_path = list(reversed(path)) if reverse else list(path)

    fig, ax = plt.subplots(figsize=(8, 8))

    n = _draw_static_maze(
        ax=ax,
        right_blocked=right_blocked,
        down_blocked=down_blocked,
        start=start,
        goal=goal,
        death_pits=death_pits,
        confusion_pads=confusion_pads,
        teleport_map=teleport_map,
        explored=explored,
    )

    ax.set_title(title)

    animation_running = {"open": True}

    def on_close(_event):
        animation_running["open"] = False

    fig.canvas.mpl_connect("close_event", on_close)

    # Agent marker starts at first displayed position
    first_x, first_y = _cell_center(display_path[0], n)
    agent_dot = ax.scatter([first_x], [first_y], s=70, c="blue", marker="o", zorder=6)

    # Trail line
    trail_x: List[float] = []
    trail_y: List[float] = []
    trail_line, = ax.plot([], [], color="red", linewidth=2.0, zorder=4)

    plt.tight_layout()
    plt.ion()
    plt.show()

    for pos in display_path:
        if close_to_skip and (not animation_running["open"] or not plt.fignum_exists(fig.number)):
            plt.close(fig)
            plt.ioff()
            return False

        x, y = _cell_center(pos, n)
        trail_x.append(x)
        trail_y.append(y)

        trail_line.set_data(trail_x, trail_y)
        agent_dot.set_offsets(np.array([[x, y]]))

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        # Sleep in tiny chunks so closing the window interrupts quickly
        waited = 0.0
        while waited < delay:
            if close_to_skip and (not animation_running["open"] or not plt.fignum_exists(fig.number)):
                plt.close(fig)
                plt.ioff()
                return False
            step = min(0.01, delay - waited)
            time.sleep(step)
            waited += step

    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches="tight")

    plt.ioff()
    plt.show()
    return True