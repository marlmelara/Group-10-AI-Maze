"""Top-level experiment runner.

Trains the agent on maze-alpha then evaluates the SAME agent (same learned
hyperparameters) on maze-beta and (optionally) maze-gamma.  Produces a JSON
report with all required primary + secondary metrics and saves the per-maze
visualizations.

Usage
-----
    python run_experiments.py [--episodes 5] [--training-episodes 10]
                               [--mazes alpha beta gamma]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from environment import MazeEnvironment
from agent import HybridRLAgent
from metrics import AggregateMetrics, EpisodeRecord, run_episode
from maze_parser import ParsedMaze
from visualizer import (plot_static_map, plot_trajectory,
                         plot_learning_curve, plot_visit_heatmap)


def evaluate(maze_id: str, agent: HybridRLAgent, episodes: int,
             max_turns: int, record_trajectories: bool = True) -> AggregateMetrics:
    env = MazeEnvironment(maze_id)
    agent.reset_for_new_maze()
    agent.set_start_goal(env.start_pos, env.goal_pos)
    records = []
    for i in range(episodes):
        rec = run_episode(env, agent, i + 1, max_turns=max_turns,
                          record_trajectory=record_trajectories)
        records.append(rec)
        print(f"  ep {i+1}: goal={rec.goal_reached} turns={rec.turns_taken} "
              f"deaths={rec.deaths} actions={rec.actions_taken} "
              f"explored={rec.cells_explored}", flush=True)
    return AggregateMetrics(episodes=records)


def train(maze_id: str, agent: HybridRLAgent, episodes: int, max_turns: int) -> AggregateMetrics:
    """Run `episodes` training runs on `maze_id`.  Agent retains its learned
    state (walls, fires, Q decay, etc.) across episodes."""
    env = MazeEnvironment(maze_id)
    agent.reset_for_new_maze()
    agent.set_start_goal(env.start_pos, env.goal_pos)
    records = []
    for i in range(episodes):
        rec = run_episode(env, agent, i + 1, max_turns=max_turns,
                          record_trajectory=(i == episodes - 1))
        records.append(rec)
        print(f"  train ep {i+1}: goal={rec.goal_reached} turns={rec.turns_taken} "
              f"deaths={rec.deaths} actions={rec.actions_taken}", flush=True)
    return AggregateMetrics(episodes=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5,
                        help="Test episodes per maze (default 5, matching spec)")
    parser.add_argument("--training-episodes", type=int, default=10,
                        help="Training episodes on maze-alpha before eval")
    parser.add_argument("--max-turns", type=int, default=10000)
    parser.add_argument("--mazes", nargs="+",
                        default=["alpha", "beta", "gamma"])
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "metrics.json"))
    parser.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    agent = HybridRLAgent()

    report: dict = {"config": vars(args), "results": {}}

    # Training phase on alpha.
    if "alpha" in args.mazes:
        print("=" * 72)
        print(f"TRAINING on maze-alpha for {args.training_episodes} episodes")
        print("=" * 72)
        train_metrics = train("alpha", agent, args.training_episodes, args.max_turns)
        nav_alpha = MazeEnvironment("alpha").total_navigable_cells()
        report["results"]["alpha_training"] = train_metrics.summary(nav_alpha)
        # Save learning curve.
        plot_learning_curve(train_metrics.episodes,
                            os.path.join(args.fig_dir, "learning_curve_alpha.png"),
                            title="Maze-alpha training learning curve")

    # Evaluation on each maze.
    for maze_id in args.mazes:
        print("=" * 72)
        print(f"EVALUATING on maze-{maze_id} for {args.episodes} episodes")
        print("=" * 72)
        metrics = evaluate(maze_id, agent, args.episodes, args.max_turns)
        env = MazeEnvironment(maze_id)
        nav = env.total_navigable_cells()
        summary = metrics.summary(nav)
        report["results"][f"{maze_id}_eval"] = summary
        print(f"  summary: success={summary['success_rate']:.2%} "
              f"avg_turns={summary['avg_turns']:.1f} "
              f"avg_path={summary['avg_path_length']:.1f} "
              f"death_rate={summary['death_rate']:.4f}")

        # Figures.
        parsed = ParsedMaze.load(os.path.join(ROOT, "data", maze_id, "parsed.npz"))
        plot_static_map(parsed, os.path.join(args.fig_dir, f"map_{maze_id}.png"),
                        title=f"maze-{maze_id}: walls + hazards")
        # Trajectory figure uses last successful episode's trajectory if any.
        traj = None
        for rec in reversed(metrics.episodes):
            if rec.goal_reached and rec.trajectory:
                traj = rec.trajectory
                break
        if traj:
            plot_trajectory(parsed, traj,
                            os.path.join(args.fig_dir, f"trajectory_{maze_id}.png"),
                            title=f"maze-{maze_id}: agent trajectory")
        plot_visit_heatmap(agent, os.path.join(args.fig_dir, f"heatmap_{maze_id}.png"),
                           parsed,
                           title=f"maze-{maze_id}: visit count heatmap")

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote metrics report -> {args.output}")


if __name__ == "__main__":
    main()
