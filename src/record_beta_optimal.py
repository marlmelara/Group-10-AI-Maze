"""Record a single cold-start beta episode as a GIF.

Simplified: runs ONE episode of beta (cold-start, which we know
solves in ~5000 turns based on prior verified runs), subsamples
every 40 turns to keep the frame count manageable, then renders
a GIF.  No warmup, no monkey-patching, no multi-episode.

Usage:  python3 src/record_beta_optimal.py
Output: figures/beta_demo.gif + figures/beta_demo_final.png
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from environment import MazeEnvironment
from agent import HybridRLAgent
from maze_parser import ParsedMaze
from visualizer import draw_maze, _cell_xy


FRAME_STRIDE = 40      # keep every 40th turn (= ~125 frames for 5000 turns)
MAX_TURNS = 15000


def record_single_episode(maze_id: str = "beta"):
    """Run 1 cold-start episode, capture subsampled frames."""
    agent = HybridRLAgent()
    env = MazeEnvironment(maze_id)
    agent.reset_for_new_maze()
    agent.set_start_goal(env.start_pos, env.goal_pos)
    env.reset(); agent.reset_episode(); agent.pos = env.start_pos

    frames: List[Tuple] = []
    # First frame.
    frames.append((env.agent_pos, tuple(env.snapshot_fires()),
                   env.death_counter, 0, False))
    last = None
    t0 = time.time()
    for turn in range(1, MAX_TURNS + 1):
        actions = agent.plan_turn(last)
        last = env.step(actions)
        if (turn % FRAME_STRIDE == 0) or last.is_goal_reached:
            frames.append((env.agent_pos, tuple(env.snapshot_fires()),
                           env.death_counter, turn,
                           last.is_goal_reached))
            if turn % 500 == 0:
                print(f"t={turn} pos={env.agent_pos} "
                      f"deaths={env.death_counter} "
                      f"frames={len(frames)} "
                      f"el={time.time()-t0:.1f}s", flush=True)
        if last.is_goal_reached:
            break

    stats = env.get_episode_stats()
    print(f"DONE ep: goal={stats['goal_reached']} "
          f"turns={stats['turns_taken']} deaths={stats['deaths']} "
          f"frames={len(frames)} wall_s={time.time()-t0:.1f}",
          flush=True)
    return frames, env.parsed, stats


def render_gif(frames: List[Tuple], parsed: ParsedMaze, out_gif: str,
               fps: int = 15):
    """Render captured frames directly to a GIF."""
    print(f"Rendering {len(frames)} frames -> {out_gif}", flush=True)
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
    trail_line, = ax.plot([], [], color="crimson", lw=1.6, zorder=5)
    agent_dot = ax.scatter([], [], s=120, c="dodgerblue", marker="o",
                           edgecolors="black", linewidth=1.4, zorder=6)
    status = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                     fontsize=11, verticalalignment="top",
                     bbox=dict(facecolor="white", alpha=0.88,
                               edgecolor="gray"))
    ax.set_title("maze-beta  •  agent solving live  •  Group 15")
    plt.tight_layout()

    trail_x: List[float] = []
    trail_y: List[float] = []

    def update(i):
        pos, fires, deaths, turn, goal = frames[i]
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

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=1000 // fps, blit=False,
                                  repeat=False)
    ani.save(out_gif, writer="pillow", fps=fps, dpi=100)
    still = out_gif.replace(".gif", "_final.png")
    fig.savefig(still, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    out_gif = os.path.join(ROOT, "figures", "beta_demo.gif")
    pickle_path = os.path.join(ROOT, "results", "beta_frames.pkl")
    t_start = time.time()

    frames, parsed, stats = record_single_episode()
    with open(pickle_path, "wb") as f:
        pickle.dump({"frames": frames, "stats": dict(stats)}, f)
    print(f"Saved {len(frames)} frames -> {pickle_path}", flush=True)

    render_gif(frames, parsed, out_gif)
    print(f"All done. Wall time: {time.time() - t_start:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
