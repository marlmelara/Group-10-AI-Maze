"""Quick per-maze evaluation with per-episode time cap.

Writes metrics.json including partial results when episodes time out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from environment import MazeEnvironment
from agent import HybridRLAgent
from metrics import EpisodeRecord, AggregateMetrics, run_episode
from maze_parser import ParsedMaze
from visualizer import (plot_static_map, plot_trajectory,
                         plot_learning_curve, plot_visit_heatmap)


def run_time_capped_episode(env: MazeEnvironment, agent: HybridRLAgent, ep: int,
                             max_turns: int = 10_000,
                             time_cap_s: float = 45.0) -> EpisodeRecord:
    """Like run_episode but caps wall time per episode."""
    env.reset()
    agent.reset_episode()
    agent.pos = env.start_pos

    last = None
    wall_start = time.time()
    replan_time = 0.0
    replans = 0
    path_length = 1
    trajectory = [env.start_pos]

    for t in range(max_turns):
        if time.time() - wall_start > time_cap_s:
            break
        t_start = time.time()
        actions = agent.plan_turn(last)
        replan_time += time.time() - t_start
        replans += 1
        last = env.step(actions)
        path_length += last.actions_executed
        trajectory.append(env.agent_pos)
        if last.is_goal_reached:
            try:
                agent._integrate(last)
            except Exception:
                pass
            break

    stats = env.get_episode_stats()
    return EpisodeRecord(
        episode_index=ep,
        goal_reached=stats["goal_reached"],
        turns_taken=stats["turns_taken"],
        actions_taken=stats["actions_taken"],
        deaths=stats["deaths"],
        confused=stats["confused"],
        teleports=stats["teleports"],
        wall_hits=stats["wall_hits"],
        wind_hits=stats["wind_hits"],
        cells_explored=stats["cells_explored"],
        total_cells_visited=stats["actions_taken"],
        wall_time_seconds=time.time() - wall_start,
        replans=replans,
        replan_wall_time=replan_time,
        path_length=path_length,
        trajectory=trajectory,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--training", type=int, default=3)
    parser.add_argument("--time-cap", type=float, default=45.0)
    parser.add_argument("--mazes", nargs="+", default=["alpha", "beta", "gamma"])
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "metrics.json"))
    parser.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    report = {"config": vars(args), "results": {}}

    # Training on alpha.
    print("=" * 72)
    print(f"TRAINING on maze-alpha for {args.training} episodes")
    print("=" * 72, flush=True)
    agent = HybridRLAgent()
    env = MazeEnvironment("alpha")
    agent.reset_for_new_maze()
    agent.set_start_goal(env.start_pos, env.goal_pos)
    training_records = []
    for i in range(args.training):
        rec = run_time_capped_episode(env, agent, i + 1,
                                      time_cap_s=args.time_cap)
        training_records.append(rec)
        print(f"  train ep {i+1}: goal={rec.goal_reached} turns={rec.turns_taken} "
              f"deaths={rec.deaths} actions={rec.actions_taken} elapsed={rec.wall_time_seconds:.1f}s",
              flush=True)
    report["results"]["alpha_training"] = AggregateMetrics(training_records).summary(
        env.total_navigable_cells())
    plot_learning_curve(training_records,
                        os.path.join(args.fig_dir, "learning_curve_alpha.png"),
                        title="Maze-alpha training")

    # Evaluation on each maze (fresh agent per maze, shared hyperparameters).
    for maze_id in args.mazes:
        print("=" * 72)
        print(f"EVALUATING on maze-{maze_id} for {args.episodes} episodes "
              f"(time cap {args.time_cap:.0f}s per ep)")
        print("=" * 72, flush=True)
        env = MazeEnvironment(maze_id)
        agent.reset_for_new_maze()
        agent.set_start_goal(env.start_pos, env.goal_pos)
        records = []
        for i in range(args.episodes):
            rec = run_time_capped_episode(env, agent, i + 1,
                                          time_cap_s=args.time_cap)
            records.append(rec)
            print(f"  ep {i+1}: goal={rec.goal_reached} turns={rec.turns_taken} "
                  f"deaths={rec.deaths} actions={rec.actions_taken} "
                  f"explored={rec.cells_explored} elapsed={rec.wall_time_seconds:.1f}s",
                  flush=True)
        metrics = AggregateMetrics(records)
        nav = env.total_navigable_cells()
        summary = metrics.summary(nav)
        report["results"][f"{maze_id}_eval"] = summary
        print(f"  summary: success={summary['success_rate']:.2%} "
              f"avg_turns={summary['avg_turns']:.1f} "
              f"avg_path={summary['avg_path_length']:.1f} "
              f"death_rate={summary['death_rate']:.4f}", flush=True)

        # Figures
        parsed = ParsedMaze.load(os.path.join(ROOT, "data", maze_id, "parsed.npz"))
        plot_static_map(parsed, os.path.join(args.fig_dir, f"map_{maze_id}.png"),
                        title=f"maze-{maze_id}: walls + hazards")
        traj = None
        for rec in reversed(records):
            if rec.goal_reached and rec.trajectory:
                traj = rec.trajectory; break
        if traj is None:
            # Use last trajectory even if partial
            traj = records[-1].trajectory if records else None
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
