"""Record a GIF of the agent solving maze-beta in converged mode.

Runs 2 silent warmup episodes to build the agent's map, then records
the 3rd episode (converged BFS-optimal path) as an animated GIF for
use in the slide deck.  Output is figures/beta_demo.gif and
figures/beta_demo_first_frame.png (a still that you can show if the
GIF doesn't play for some reason).

Usage:  python3 src/record_demo_gif.py [--maze beta] [--warmup 2]
"""
from __future__ import annotations

import argparse
import os
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


def build_figure(parsed: ParsedMaze, maze_id: str):
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
    trail_line, = ax.plot([], [], color="crimson", lw=1.4, zorder=5)
    agent_dot = ax.scatter([], [], s=120, c="dodgerblue", marker="o",
                           edgecolors="black", linewidth=1.4, zorder=6)
    status = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                     fontsize=11, verticalalignment="top",
                     bbox=dict(facecolor="white", alpha=0.88,
                               edgecolor="gray"))
    ax.set_title(f"maze-{maze_id}  •  converged demo (post-warmup)  •  Group 15")
    plt.tight_layout()
    return fig, ax, fire_scatter, trail_line, agent_dot, status


def run_silent_episode(env: MazeEnvironment, agent: HybridRLAgent,
                       max_turns: int = 15000) -> dict:
    env.reset(); agent.reset_episode(); agent.pos = env.start_pos
    last = None
    for _ in range(max_turns):
        actions = agent.plan_turn(last)
        last = env.step(actions)
        if last.is_goal_reached: break
    return env.get_episode_stats()


def record_converged(env: MazeEnvironment, agent: HybridRLAgent,
                     parsed: ParsedMaze, maze_id: str, out_gif: str,
                     max_turns: int = 15000,
                     frame_stride: int = 30) -> dict:
    """Run one episode and record every `frame_stride`-th turn as a
    GIF frame.  Keeps the final GIF short regardless of episode length.
    """
    env.reset(); agent.reset_episode(); agent.pos = env.start_pos
    fig, ax, fire_scatter, trail_line, agent_dot, status = build_figure(
        parsed, maze_id)

    trail_x: List[float] = []
    trail_y: List[float] = []
    frames: list = []
    last = None
    for turn in range(1, max_turns + 1):
        actions = agent.plan_turn(last)
        last = env.step(actions)
        pos = env.agent_pos
        # Subsample: keep first frame, every frame_stride-th, and last.
        if (turn == 1) or (turn % frame_stride == 0) or last.is_goal_reached:
            fires = env.snapshot_fires()
            frames.append((pos, fires, env.death_counter, turn,
                           last.is_goal_reached))
        if last.is_goal_reached:
            break

    def init():
        return fire_scatter, trail_line, agent_dot, status

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

    ani = animation.FuncAnimation(fig, update, init_func=init,
                                  frames=len(frames), interval=90,
                                  blit=False, repeat=False)
    ani.save(out_gif, writer="pillow", fps=12, dpi=100)
    # Also save the final frame as a still.
    still = out_gif.replace(".gif", "_final.png")
    fig.savefig(still, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return env.get_episode_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maze", default="beta")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--output",
                    default=os.path.join(ROOT, "figures", "beta_demo.gif"))
    args = ap.parse_args()

    agent = HybridRLAgent()
    env = MazeEnvironment(args.maze)
    agent.reset_for_new_maze()
    agent.set_start_goal(env.start_pos, env.goal_pos)

    t0 = time.time()
    for ep in range(args.warmup):
        stats = run_silent_episode(env, agent)
        print(f"warmup ep{ep+1}: goal={stats['goal_reached']} "
              f"turns={stats['turns_taken']} deaths={stats['deaths']} "
              f"wall_s={time.time() - t0:.1f}", flush=True)

    print(f"Recording converged episode to {args.output} ...", flush=True)
    parsed = env.parsed
    stats = record_converged(env, agent, parsed, args.maze, args.output)
    print(f"Recorded: goal={stats['goal_reached']} "
          f"turns={stats['turns_taken']} deaths={stats['deaths']} "
          f"wall_s={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
