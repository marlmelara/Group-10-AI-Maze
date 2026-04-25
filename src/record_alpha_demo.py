"""Record maze-alpha demo GIFs (episode 1 exploration + converged).

Mirrors src/record_beta_converged_v2.py exactly, but runs on maze-alpha
and emits BOTH the episode-1 exploration GIF and the converged-episode
GIF from a single 5-episode run.

Usage:  python3 src/record_alpha_demo.py
Outputs:
  figures/alpha_demo_exploration.gif       (+ _final.png)
  figures/alpha_demo_converged.gif         (+ _final.png)
  results/alpha_frames.pkl                 (pickled all-episode frames)
"""
from __future__ import annotations
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from environment import MazeEnvironment, Action
from agent import HybridRLAgent
from maze_parser import ParsedMaze
from visualizer import draw_maze, _cell_xy


MAX_TURNS = 15000
NUM_EPS = 5


def run_episode_capturing(env: MazeEnvironment, agent: HybridRLAgent,
                           episode_index: int,
                           max_turns: int = MAX_TURNS) -> Tuple[dict, list]:
    """EXACT copy of metrics.run_episode plus per-turn fire capture.
    Returns (stats_dict, frames_list)."""
    env.reset()
    agent.reset_episode()
    agent.pos = env.start_pos

    last = None
    t_start = time.time()
    frames = [(env.agent_pos, tuple(env.snapshot_fires()),
               env.death_counter, 0, False)]
    last_log = time.time()

    for t in range(max_turns):
        actions = agent.plan_turn(last)
        last = env.step(actions)
        frames.append((env.agent_pos, tuple(env.snapshot_fires()),
                       env.death_counter, t + 1,
                       last.is_goal_reached))

        if time.time() - last_log > 30:
            print(f"  ep{episode_index} t={t+1} pos={env.agent_pos} "
                  f"deaths={env.death_counter} "
                  f"explored={len(env.cells_explored)} "
                  f"wall_s={time.time() - t_start:.1f}", flush=True)
            last_log = time.time()

        if last.is_goal_reached:
            try:
                agent._integrate(last)
            except Exception:
                pass
            break

    stats = env.get_episode_stats()
    stats["wall_s"] = time.time() - t_start
    return stats, frames


def subsample(frames: list, target_max: int = 120) -> list:
    if len(frames) <= target_max:
        return frames
    step = max(1, len(frames) // target_max)
    out = frames[::step]
    if out[-1] is not frames[-1]:
        out.append(frames[-1])
    return out


def render_gif(frames: list, parsed: ParsedMaze, out_gif: str,
               title: str, fps: int = 10):
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
    out_expl = str(ROOT / "figures" / "alpha_demo_exploration.gif")
    out_conv = str(ROOT / "figures" / "alpha_demo_converged.gif")
    pkl_path = str(ROOT / "results" / "alpha_frames.pkl")

    t0 = time.time()
    agent = HybridRLAgent()
    env = MazeEnvironment("alpha")
    agent.reset_for_new_maze()
    agent.set_start_goal(env.start_pos, env.goal_pos)

    all_eps: Dict[int, dict] = {}
    for ep in range(1, NUM_EPS + 1):
        stats, frames = run_episode_capturing(env, agent, ep)
        print(f"ep{ep}: goal={stats['goal_reached']} "
              f"turns={stats['turns_taken']} deaths={stats['deaths']} "
              f"frames={len(frames)} wall_s={stats['wall_s']:.1f}",
              flush=True)
        all_eps[ep] = {"stats": dict(stats), "frames": frames}

    # Pickle everything for reproducibility.
    with open(pkl_path, "wb") as f:
        pickle.dump({"all": all_eps}, f)
    print(f"Pickled -> {pkl_path}", flush=True)

    # Pick the SHORTEST successful episode = most converged.
    best_ep, best_frames, best_turns = None, None, 10 ** 12
    for ep, data in all_eps.items():
        f = data["frames"]
        if not f or not f[-1][4]:
            continue
        if len(f) < best_turns:
            best_turns = len(f)
            best_ep = ep
            best_frames = f
    if best_ep is None:
        print("No converged episode; aborting GIFs.", flush=True)
        return

    # Episode 1 (exploration GIF).
    ep1_data = all_eps.get(1)
    if ep1_data is not None:
        ep1_frames = ep1_data["frames"]
        ep1_stats = ep1_data["stats"]
        display = subsample(ep1_frames, target_max=120)
        title = (f"maze-alpha  •  episode 1 (RL exploration)  •  "
                 f"{ep1_stats['turns_taken']} turns  •  "
                 f"{ep1_stats['deaths']} deaths  •  Group 15")
        render_gif(display, env.parsed, out_expl, title=title, fps=10)
        print(f"Rendered exploration GIF -> {out_expl}", flush=True)

    # Converged GIF.
    print(f"\nConverged = ep{best_ep} ({len(best_frames)} frames)",
          flush=True)
    display = subsample(best_frames, target_max=120)
    conv_stats = all_eps[best_ep]["stats"]
    title = (f"maze-alpha  •  converged ep{best_ep} (BFS-optimal)  •  "
             f"{conv_stats['turns_taken']} turns  •  "
             f"{conv_stats['deaths']} deaths  •  Group 15")
    render_gif(display, env.parsed, out_conv, title=title, fps=10)
    print(f"Rendered converged GIF -> {out_conv}", flush=True)
    print(f"Total wall time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
