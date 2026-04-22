"""Record the CONVERGED beta episode as a GIF.

Runs a standard 5-episode beta evaluation (matching run_experiments.py
semantics so beta converges to the ~80-turn optimal path by ep 3+).
Captures per-turn frames for EVERY episode so we can pick the shortest
(most converged) one afterwards.  Renders a GIF of that converged
episode at a faster frame rate since the path is short.

Usage:  python3 src/record_beta_converged.py
Output: figures/beta_demo_converged.gif +
        figures/beta_demo_converged_final.png +
        results/beta_frames_converged.pkl
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from typing import Dict, List, Tuple

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


MAX_TURNS = 15000
NUM_EPS = 5


def record_all_episodes(maze_id: str = "beta") -> Tuple[Dict[int, list], ParsedMaze]:
    """Run NUM_EPS cold-to-converged episodes on maze_id, capturing
    EVERY turn's state for each episode.  Returns dict keyed by
    episode number -> list of (pos, fires, deaths, turn, goal) frames.
    """
    agent = HybridRLAgent()
    env = MazeEnvironment(maze_id)
    agent.reset_for_new_maze()
    agent.set_start_goal(env.start_pos, env.goal_pos)

    all_frames: Dict[int, list] = {}
    for ep in range(1, NUM_EPS + 1):
        env.reset(); agent.reset_episode(); agent.pos = env.start_pos
        frames: list = [(env.agent_pos, tuple(env.snapshot_fires()),
                          env.death_counter, 0, False)]
        last = None
        t0 = time.time()
        for turn in range(1, MAX_TURNS + 1):
            actions = agent.plan_turn(last)
            last = env.step(actions)
            frames.append((env.agent_pos, tuple(env.snapshot_fires()),
                           env.death_counter, turn,
                           last.is_goal_reached))
            if last.is_goal_reached:
                break
        stats = env.get_episode_stats()
        el = time.time() - t0
        print(f"ep{ep}: goal={stats['goal_reached']} "
              f"turns={stats['turns_taken']} deaths={stats['deaths']} "
              f"frames={len(frames)} wall_s={el:.1f}", flush=True)
        all_frames[ep] = frames
    return all_frames, env.parsed


def pick_converged(all_frames: Dict[int, list]) -> Tuple[int, list]:
    """Pick the SHORTEST successful episode — that's the converged one."""
    best_ep, best_frames = None, None
    for ep, frames in all_frames.items():
        if not frames:
            continue
        if not frames[-1][4]:  # not goal-reached
            continue
        if best_frames is None or len(frames) < len(best_frames):
            best_ep, best_frames = ep, frames
    return best_ep, best_frames


def subsample(frames: list, target_max: int = 120) -> list:
    """Subsample to keep GIF compact.  Always keep first and last."""
    if len(frames) <= target_max:
        return frames
    step = max(1, len(frames) // target_max)
    out = frames[::step]
    if out[-1] is not frames[-1]:
        out.append(frames[-1])
    return out


def render_gif(frames: list, parsed: ParsedMaze, out_gif: str,
               title: str, fps: int = 15):
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
    ax.set_title(title)
    plt.tight_layout()

    trail_x: list = []
    trail_y: list = []

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
    out_gif = os.path.join(ROOT, "figures", "beta_demo_converged.gif")
    pickle_path = os.path.join(ROOT, "results",
                               "beta_frames_converged.pkl")
    t_start = time.time()

    all_frames, parsed = record_all_episodes()
    best_ep, best_frames = pick_converged(all_frames)
    if best_ep is None:
        print("No successful episode captured; aborting.", flush=True)
        return
    print(f"Converged episode chosen: ep{best_ep} with "
          f"{len(best_frames)} frames.", flush=True)
    with open(pickle_path, "wb") as f:
        pickle.dump({"all_frames": all_frames, "chosen_ep": best_ep}, f)
    print(f"Saved full pickle -> {pickle_path}", flush=True)

    # The converged path is short (~80 frames); plot every frame at
    # 10 fps for a smooth ~8-second GIF.
    display_frames = subsample(best_frames, target_max=120)
    render_gif(display_frames, parsed, out_gif,
               title=f"maze-beta  •  converged ep{best_ep} "
                     f"(BFS-optimal path)  •  Group 15",
               fps=10)
    print(f"Rendered GIF -> {out_gif}", flush=True)
    print(f"All done. Wall time: {time.time() - t_start:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
