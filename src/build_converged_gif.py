"""Build a 'converged' beta demo GIF from the tail end of the ep 1
pickle — i.e., the final stretch where the agent successfully
navigates to the goal.  Much shorter and cleaner than the full
exploration clip.
"""
from __future__ import annotations
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from environment import MazeEnvironment
from maze_parser import ParsedMaze
from visualizer import draw_maze, _cell_xy


PKL = ROOT / "results" / "beta_frames_ep1.pkl"
OUT_GIF = ROOT / "figures" / "beta_demo_converged.gif"


def main():
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    frames = data["frames"]
    print(f"Loaded {len(frames)} frames from {PKL}")

    # Take the last 60 frames — that's roughly the final 2400 turns
    # (frame_stride was 40) where the agent actually reaches the goal.
    tail = frames[-60:]
    # Ensure the goal frame is the last one.
    print(f"Using tail of {len(tail)} frames; "
          f"last turn = {tail[-1][3]}, goal_reached={tail[-1][4]}")

    env = MazeEnvironment("beta")
    parsed = env.parsed

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_maze(ax, parsed, show_hazards=False)
    if parsed.confusion_pads:
        xs, ys = zip(*[_cell_xy(p) for p in parsed.confusion_pads])
        ax.scatter(xs, ys, s=22, c="gold", marker="o",
                   edgecolors="black", zorder=3)
    for src, dst in zip(parsed.teleport_sources, parsed.teleport_dests):
        sx, sy = _cell_xy(src); dx, dy = _cell_xy(dst)
        ax.scatter([sx], [sy], s=36, c="purple", marker="^", zorder=4)
        ax.scatter([dx], [dy], s=36, c="magenta", marker="v", zorder=4)
    sx, sy = _cell_xy(parsed.start)
    gx, gy = _cell_xy(parsed.goal)
    ax.scatter([sx], [sy], s=120, c="limegreen", marker="o",
               zorder=5, edgecolors="black")
    ax.scatter([gx], [gy], s=180, c="red", marker="*",
               zorder=5, edgecolors="black")

    fire_scatter = ax.scatter([], [], s=48, c="orangered", marker="s",
                              zorder=4)
    trail_line, = ax.plot([], [], color="crimson", lw=1.8, zorder=5)
    agent_dot = ax.scatter([], [], s=120, c="dodgerblue", marker="o",
                           edgecolors="black", linewidth=1.4, zorder=6)
    status = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                     fontsize=11, verticalalignment="top",
                     bbox=dict(facecolor="white", alpha=0.88,
                               edgecolor="gray"))
    ax.set_title("maze-beta  •  goal-approach (endgame of solved ep)  "
                 "•  Group 15")
    plt.tight_layout()

    trail_x: list = []
    trail_y: list = []

    def update(i):
        pos, fires, deaths, turn, goal = tail[i]
        x, y = _cell_xy(pos)
        trail_x.append(x); trail_y.append(y)
        trail_line.set_data(trail_x, trail_y)
        agent_dot.set_offsets(np.array([[x, y]]))
        if fires:
            fx, fy = zip(*[_cell_xy(p) for p in fires])
            fire_scatter.set_offsets(np.array(list(zip(fx, fy))))
        else:
            fire_scatter.set_offsets(np.zeros((0, 2)))
        tag = "  GOAL!" if goal else ""
        status.set_text(f"Turn {turn}  Deaths {deaths}  Pos {pos}{tag}")
        return fire_scatter, trail_line, agent_dot, status

    ani = animation.FuncAnimation(fig, update, frames=len(tail),
                                  interval=100, blit=False,
                                  repeat=False)
    ani.save(str(OUT_GIF), writer="pillow", fps=10, dpi=100)
    still = str(OUT_GIF).replace(".gif", "_final.png")
    fig.savefig(still, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_GIF}")
    print(f"Wrote {still}")


if __name__ == "__main__":
    main()
