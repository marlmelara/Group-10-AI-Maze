"""Visualizations for the maze solver: static maps, trajectories, live demo,
and learning curves."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from maze_parser import GRID_SIZE, ParsedMaze
from agent import HybridRLAgent

Position = Tuple[int, int]


def _cell_xy(pos: Position, n: int = GRID_SIZE) -> Tuple[float, float]:
    r, c = pos
    return (c + 0.5, n - (r + 0.5))


def draw_maze(ax, parsed: ParsedMaze, *,
              show_hazards: bool = True,
              show_start_goal: bool = True,
              alpha: float = 1.0) -> None:
    n = GRID_SIZE
    ax.plot([0, n, n, 0, 0], [0, 0, n, n, 0], color="black", lw=1.2, alpha=alpha)
    for r in range(n):
        for c in range(n - 1):
            if parsed.right_blocked[r, c]:
                x = c + 1
                ax.plot([x, x], [n - (r + 1), n - r], color="black", lw=0.6, alpha=alpha)
    for r in range(n - 1):
        for c in range(n):
            if parsed.down_blocked[r, c]:
                y = n - (r + 1)
                ax.plot([c, c + 1], [y, y], color="black", lw=0.6, alpha=alpha)

    if show_hazards:
        if parsed.death_pits:
            xs, ys = zip(*[_cell_xy(p) for p in parsed.death_pits])
            ax.scatter(xs, ys, s=22, c="orangered", marker="s", zorder=3,
                       label="Fire / Death pit")
        if parsed.confusion_pads:
            xs, ys = zip(*[_cell_xy(p) for p in parsed.confusion_pads])
            ax.scatter(xs, ys, s=22, c="gold", marker="o",
                       edgecolors="black", zorder=3, label="Confusion pad")
        for src, dst in zip(parsed.teleport_sources, parsed.teleport_dests):
            sx, sy = _cell_xy(src)
            dx, dy = _cell_xy(dst)
            ax.scatter([sx], [sy], s=36, c="purple", marker="^", zorder=4)
            ax.scatter([dx], [dy], s=36, c="magenta", marker="v", zorder=4)
            ax.plot([sx, dx], [sy, dy], color="purple", alpha=0.25, lw=0.6, zorder=2)
        if parsed.wind_cells:
            for pos, direction in parsed.wind_cells.items():
                x, y = _cell_xy(pos)
                dx, dy = {"UP": (0, 0.4), "DOWN": (0, -0.4),
                          "LEFT": (-0.4, 0), "RIGHT": (0.4, 0)}[direction]
                ax.arrow(x - dx / 2, y - dy / 2, dx, dy, head_width=0.22,
                         color="royalblue", length_includes_head=True, zorder=4)

    if show_start_goal:
        sx, sy = _cell_xy(parsed.start)
        gx, gy = _cell_xy(parsed.goal)
        ax.scatter([sx], [sy], s=80, c="limegreen", marker="o", zorder=5,
                   edgecolors="black")
        ax.scatter([gx], [gy], s=120, c="red", marker="*", zorder=5,
                   edgecolors="black")

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_static_map(parsed: ParsedMaze, save_path: str, title: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(9, 9))
    draw_maze(ax, parsed)
    if title:
        ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(parsed: ParsedMaze, trajectory: List[Position],
                     save_path: str, title: Optional[str] = None,
                     explored: Optional[Iterable[Position]] = None):
    fig, ax = plt.subplots(figsize=(9, 9))
    draw_maze(ax, parsed)
    if explored:
        xs, ys = zip(*[_cell_xy(p) for p in explored])
        ax.scatter(xs, ys, s=4, c="#cfe4ff", marker="s", zorder=1.5)
    if trajectory:
        xs = [_cell_xy(p)[0] for p in trajectory]
        ys = [_cell_xy(p)[1] for p in trajectory]
        ax.plot(xs, ys, color="crimson", lw=1.0, zorder=3.5)
    if title:
        ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(episode_records, save_path: str,
                         title: str = "Learning curve"):
    """Plot turns-to-goal vs episode and deaths vs episode."""
    eps = [r.episode_index for r in episode_records]
    turns = [r.turns_taken if r.goal_reached else None for r in episode_records]
    deaths = [r.deaths for r in episode_records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(eps, [t if t is not None else 0 for t in turns], "-o", color="navy")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Turns to goal")
    ax1.set_title("Turns to reach goal")
    ax1.grid(True, alpha=0.3)

    ax2.plot(eps, deaths, "-o", color="crimson")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Deaths")
    ax2.set_title("Deaths per episode")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_visit_heatmap(agent: HybridRLAgent, save_path: str, parsed: ParsedMaze,
                        title: str = "Agent exploration heatmap"):
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for pos, cnt in agent.visit_counts.items():
        grid[pos[0], pos[1]] = cnt

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_maze(ax, parsed, show_hazards=False, show_start_goal=False, alpha=0.4)
    # Show heatmap flipped to match maze coords.
    flipped = grid[::-1, :]
    extent = (0, GRID_SIZE, 0, GRID_SIZE)
    ax.imshow(np.log1p(flipped), cmap="hot", extent=extent, alpha=0.55, origin="lower")
    sx, sy = _cell_xy(parsed.start)
    gx, gy = _cell_xy(parsed.goal)
    ax.scatter([sx], [sy], s=80, c="limegreen", marker="o", zorder=5, edgecolors="black")
    ax.scatter([gx], [gy], s=120, c="red", marker="*", zorder=5, edgecolors="black")
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def animate_episode(parsed: ParsedMaze, trajectory: List[Position],
                    fire_phase_snapshots: List[Set[Position]],
                    save_path: Optional[str] = None,
                    interval_ms: int = 30,
                    title: str = "Agent trajectory"):
    """Animate the agent's trajectory with rotating fire snapshots.

    fire_phase_snapshots[i] = set of fire cells at the i-th action.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_maze(ax, parsed, show_hazards=False)
    # Non-rotating markers (teleport, confusion) drawn persistently.
    for src, dst in zip(parsed.teleport_sources, parsed.teleport_dests):
        sx, sy = _cell_xy(src); dx, dy = _cell_xy(dst)
        ax.scatter([sx], [sy], s=36, c="purple", marker="^", zorder=4)
        ax.scatter([dx], [dy], s=36, c="magenta", marker="v", zorder=4)
    if parsed.confusion_pads:
        xs, ys = zip(*[_cell_xy(p) for p in parsed.confusion_pads])
        ax.scatter(xs, ys, s=22, c="gold", marker="o", edgecolors="black", zorder=3)

    sx, sy = _cell_xy(parsed.start); gx, gy = _cell_xy(parsed.goal)
    ax.scatter([sx], [sy], s=80, c="limegreen", marker="o", zorder=5, edgecolors="black")
    ax.scatter([gx], [gy], s=120, c="red", marker="*", zorder=5, edgecolors="black")

    fire_scatter = ax.scatter([], [], s=42, c="orangered", marker="s", zorder=4)
    trail_line, = ax.plot([], [], color="crimson", lw=1.2, zorder=5)
    agent_dot = ax.scatter([], [], s=80, c="dodgerblue", marker="o",
                            edgecolors="black", zorder=6)
    ax.set_title(title)

    trail_x: List[float] = []
    trail_y: List[float] = []

    def update(frame):
        # Fire snapshot for this frame.
        fires = fire_phase_snapshots[min(frame, len(fire_phase_snapshots) - 1)]
        if fires:
            fx, fy = zip(*[_cell_xy(p) for p in fires])
        else:
            fx, fy = [], []
        fire_scatter.set_offsets(np.array(list(zip(fx, fy))) if fires else np.zeros((0, 2)))

        pos = trajectory[frame]
        x, y = _cell_xy(pos)
        trail_x.append(x); trail_y.append(y)
        trail_line.set_data(trail_x, trail_y)
        agent_dot.set_offsets(np.array([[x, y]]))
        return fire_scatter, trail_line, agent_dot

    anim = FuncAnimation(fig, update, frames=len(trajectory),
                         interval=interval_ms, blit=False, repeat=False)
    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer=PillowWriter(fps=max(1, int(1000 / interval_ms))))
        else:
            anim.save(save_path, fps=max(1, int(1000 / interval_ms)))
    else:
        plt.show()
    plt.close(fig)
