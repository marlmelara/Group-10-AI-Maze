"""Live demo - animates the agent solving a maze in real time.

Usage:  python live_demo.py [--maze beta] [--episodes 1] [--save-gif output.gif]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from environment import MazeEnvironment, Action
from agent import HybridRLAgent
from maze_parser import ParsedMaze, GRID_SIZE
from visualizer import draw_maze, _cell_xy

Position = Tuple[int, int]


def run_live(maze_id: str, pretrain_maze: str = "alpha",
             pretrain_episodes: int = 3,
             demo_warmup: int = 2,
             save_gif: str = None,
             frame_interval_ms: int = 20,
             skip_frames: int = 2):
    """Pretrain on alpha, then show the agent solving `maze_id` live.

    Fire positions are refreshed every action to reflect rotation.
    """
    agent = HybridRLAgent()

    # Pretrain.
    if pretrain_maze and pretrain_episodes > 0:
        print(f"Pretraining on {pretrain_maze} for {pretrain_episodes} episodes...", flush=True)
        env = MazeEnvironment(pretrain_maze)
        agent.reset_for_new_maze()
        agent.set_start_goal(env.start_pos, env.goal_pos)
        for ep in range(pretrain_episodes):
            env.reset(); agent.reset_episode(); agent.pos = env.start_pos
            last = None
            for _ in range(10000):
                actions = agent.plan_turn(last)
                last = env.step(actions)
                if last.is_goal_reached: break
            stats = env.get_episode_stats()
            print(f"  train ep{ep+1}: goal={stats['goal_reached']} turns={stats['turns_taken']} "
                  f"deaths={stats['deaths']}", flush=True)

    # Evaluation.
    env = MazeEnvironment(maze_id)
    agent.reset_for_new_maze()
    agent.set_start_goal(env.start_pos, env.goal_pos)

    # Silent warmup on the DEMO maze: the first cold episode is the
    # exploration-heavy one (many thousands of turns) which is too slow
    # to watch live.  Running a few silent episodes here lets the agent
    # build its map; the subsequent animated episode then walks the
    # converged BFS-optimal path in ~80 turns.  The agent's map,
    # fires_by_phase, teleport_map, and successful_paths all persist
    # across reset_episode() calls within the same maze, so the next
    # episode benefits from what these silent runs discovered.
    if demo_warmup > 0:
        print(f"Warming up the agent's map of {maze_id} with "
              f"{demo_warmup} silent episode(s) before the animation...",
              flush=True)
        for ep in range(demo_warmup):
            env.reset(); agent.reset_episode(); agent.pos = env.start_pos
            last = None
            for _ in range(15000):
                actions = agent.plan_turn(last)
                last = env.step(actions)
                if last.is_goal_reached: break
            stats = env.get_episode_stats()
            print(f"  warmup ep{ep+1}: goal={stats['goal_reached']} "
                  f"turns={stats['turns_taken']} deaths={stats['deaths']}",
                  flush=True)

    env.reset(); agent.reset_episode(); agent.pos = env.start_pos

    parsed: ParsedMaze = env.parsed

    # Set up live plot.
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 9))
    draw_maze(ax, parsed, show_hazards=False)

    # Persistent hazards.
    if parsed.confusion_pads:
        xs, ys = zip(*[_cell_xy(p) for p in parsed.confusion_pads])
        ax.scatter(xs, ys, s=22, c="gold", marker="o", edgecolors="black", zorder=3)
    for src, dst in zip(parsed.teleport_sources, parsed.teleport_dests):
        sx, sy = _cell_xy(src); dx, dy = _cell_xy(dst)
        ax.scatter([sx], [sy], s=36, c="purple", marker="^", zorder=4)
        ax.scatter([dx], [dy], s=36, c="magenta", marker="v", zorder=4)
    if parsed.wind_cells:
        for pos, direction in parsed.wind_cells.items():
            x, y = _cell_xy(pos)
            dx, dy = {"UP": (0, 0.4), "DOWN": (0, -0.4),
                      "LEFT": (-0.4, 0), "RIGHT": (0.4, 0)}[direction]
            ax.arrow(x - dx / 2, y - dy / 2, dx, dy, head_width=0.22,
                     color="royalblue", length_includes_head=True, zorder=4)
    sx, sy = _cell_xy(parsed.start)
    gx, gy = _cell_xy(parsed.goal)
    ax.scatter([sx], [sy], s=100, c="limegreen", marker="o", zorder=5, edgecolors="black")
    ax.scatter([gx], [gy], s=140, c="red", marker="*", zorder=5, edgecolors="black")

    fire_scatter = ax.scatter([], [], s=48, c="orangered", marker="s", zorder=4)
    trail_line, = ax.plot([], [], color="crimson", lw=1.2, zorder=5)
    agent_dot = ax.scatter([], [], s=90, c="dodgerblue", marker="o",
                            edgecolors="black", linewidth=1.2, zorder=6)
    status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                           fontsize=10, verticalalignment="top",
                           bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

    ax.set_title(f"maze-{maze_id} live demo")
    plt.tight_layout()

    trail_x: List[float] = []
    trail_y: List[float] = []
    last_result = None
    frame_idx = 0

    import matplotlib.animation as animation
    frames_recorded = []

    def update_display(pos, fires, deaths, turn):
        x, y = _cell_xy(pos)
        trail_x.append(x); trail_y.append(y)
        trail_line.set_data(trail_x, trail_y)
        agent_dot.set_offsets(np.array([[x, y]]))
        if fires:
            fx, fy = zip(*[_cell_xy(p) for p in fires])
            fire_scatter.set_offsets(np.array(list(zip(fx, fy))))
        else:
            fire_scatter.set_offsets(np.zeros((0, 2)))
        status_text.set_text(f"Turn {turn}  Deaths {deaths}  Pos {pos}")

    turn = 0
    while turn < 10000:
        actions = agent.plan_turn(last_result)
        last_result = env.step(actions)
        turn += 1
        frame_idx += 1

        if frame_idx % skip_frames == 0 or last_result.is_goal_reached:
            update_display(env.agent_pos, env.snapshot_fires(),
                           env.death_counter, turn)
            plt.pause(frame_interval_ms / 1000.0)

        if last_result.is_goal_reached:
            break
        if not plt.fignum_exists(fig.number):
            break

    stats = env.get_episode_stats()
    print(f"Live demo done: goal={stats['goal_reached']} turns={stats['turns_taken']} "
          f"deaths={stats['deaths']} actions={stats['actions_taken']}")

    if plt.fignum_exists(fig.number):
        status_text.set_text(f"DONE  Turn {stats['turns_taken']}  "
                             f"Deaths {stats['deaths']}  Goal={stats['goal_reached']}")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze", default="beta")
    parser.add_argument("--pretrain-maze", default="alpha")
    parser.add_argument("--pretrain-episodes", type=int, default=3)
    parser.add_argument("--demo-warmup", type=int, default=2,
                        help="Silent episodes on the DEMO maze before "
                             "the animation (default 2); lets the live "
                             "episode show the converged optimal path.")
    parser.add_argument("--interval-ms", type=int, default=20)
    parser.add_argument("--skip-frames", type=int, default=2)
    args = parser.parse_args()

    run_live(args.maze, pretrain_maze=args.pretrain_maze,
             pretrain_episodes=args.pretrain_episodes,
             demo_warmup=args.demo_warmup,
             frame_interval_ms=args.interval_ms,
             skip_frames=args.skip_frames)
