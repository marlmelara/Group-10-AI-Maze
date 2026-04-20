"""Metrics utilities matching the spec's Evaluator interface.

Primary metrics:
  success_rate, avg_path_length, avg_turns, death_rate

Secondary (bonus) metrics:
  exploration_efficiency = unique_cells_discovered / total_cells_visited
  map_completeness       = known_cells / total_navigable_cells
  replanning_efficiency  = mean_time_per_replan (seconds)
  learning_efficiency    = episodes_until_first_success
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from environment import MazeEnvironment, Action, TurnResult
from agent import HybridRLAgent


@dataclass
class EpisodeRecord:
    episode_index: int
    goal_reached: bool
    turns_taken: int
    actions_taken: int
    deaths: int
    confused: int
    teleports: int
    wall_hits: int
    wind_hits: int
    cells_explored: int
    total_cells_visited: int
    wall_time_seconds: float
    replans: int
    replan_wall_time: float
    path_length: int              # cells visited (including duplicates) before reaching goal
    trajectory: List[tuple] = field(default_factory=list)

    @property
    def exploration_efficiency(self) -> float:
        if self.total_cells_visited == 0:
            return 0.0
        return self.cells_explored / self.total_cells_visited

    @property
    def replanning_efficiency(self) -> float:
        if self.replans == 0:
            return 0.0
        return self.replan_wall_time / self.replans


@dataclass
class AggregateMetrics:
    episodes: List[EpisodeRecord]

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.goal_reached) / len(self.episodes)

    @property
    def avg_path_length(self) -> float:
        succ = [e.path_length for e in self.episodes if e.goal_reached]
        return float(sum(succ) / len(succ)) if succ else 0.0

    @property
    def avg_turns(self) -> float:
        succ = [e.turns_taken for e in self.episodes if e.goal_reached]
        return float(sum(succ) / len(succ)) if succ else 0.0

    @property
    def death_rate(self) -> float:
        total_deaths = sum(e.deaths for e in self.episodes)
        total_turns = sum(e.turns_taken for e in self.episodes)
        return total_deaths / total_turns if total_turns else 0.0

    @property
    def avg_exploration_efficiency(self) -> float:
        vals = [e.exploration_efficiency for e in self.episodes]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def map_completeness(self, navigable_cells: int) -> float:
        if not self.episodes:
            return 0.0
        # Use the final (largest) exploration size across episodes.
        max_known = max(e.cells_explored for e in self.episodes)
        return min(1.0, max_known / max(1, navigable_cells))

    @property
    def avg_replanning_time(self) -> float:
        vals = [e.replanning_efficiency for e in self.episodes if e.replans > 0]
        return float(sum(vals) / len(vals)) if vals else 0.0

    @property
    def learning_efficiency(self) -> Optional[int]:
        for idx, e in enumerate(self.episodes):
            if e.goal_reached:
                return idx + 1
        return None

    def summary(self, navigable_cells: int) -> Dict:
        return {
            "episodes": len(self.episodes),
            "success_rate": round(self.success_rate, 4),
            "avg_path_length": round(self.avg_path_length, 2),
            "avg_turns": round(self.avg_turns, 2),
            "death_rate": round(self.death_rate, 4),
            "exploration_efficiency": round(self.avg_exploration_efficiency, 4),
            "map_completeness": round(self.map_completeness(navigable_cells), 4),
            "replanning_efficiency_ms": round(self.avg_replanning_time * 1000, 2),
            "learning_efficiency": self.learning_efficiency,
            "per_episode": [
                {
                    "ep": e.episode_index,
                    "goal": e.goal_reached,
                    "turns": e.turns_taken,
                    "actions": e.actions_taken,
                    "deaths": e.deaths,
                    "path_len": e.path_length,
                    "cells_explored": e.cells_explored,
                    "expl_eff": round(e.exploration_efficiency, 3),
                    "wall_seconds": round(e.wall_time_seconds, 2),
                } for e in self.episodes
            ],
        }


def run_episode(env: MazeEnvironment, agent: HybridRLAgent, episode_index: int,
                max_turns: int = 10000, record_trajectory: bool = False) -> EpisodeRecord:
    env.reset()
    agent.reset_episode()
    agent.pos = env.start_pos

    last: Optional[TurnResult] = None
    wall_start = time.time()
    replan_time = 0.0
    replans = 0
    path_length = 1                # starting cell counts
    trajectory: List[tuple] = [env.start_pos] if record_trajectory else []

    for t in range(max_turns):
        t_start = time.time()
        actions = agent.plan_turn(last)
        replan_time += time.time() - t_start
        replans += 1

        last = env.step(actions)
        path_length += last.actions_executed
        if record_trajectory:
            trajectory.append(env.agent_pos)
        if last.is_goal_reached:
            # Feed the final TurnResult to the agent so its internal state
            # (successful_paths, fires, Q, etc.) captures the win.
            try:
                agent._integrate(last)
            except Exception:
                pass
            break

    stats = env.get_episode_stats()
    wall_time = time.time() - wall_start

    return EpisodeRecord(
        episode_index=episode_index,
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
        wall_time_seconds=wall_time,
        replans=replans,
        replan_wall_time=replan_time,
        path_length=path_length,
        trajectory=trajectory,
    )


def run_multi_episode(maze_id: str, agent: HybridRLAgent, episodes: int,
                      max_turns: int = 10000, record_trajectory: bool = False,
                      share_agent: bool = True) -> AggregateMetrics:
    """Run `episodes` on the given maze and collect metrics."""
    env = MazeEnvironment(maze_id)
    if share_agent:
        # Tell the agent about start/goal so it can initialise its plan.
        agent.set_start_goal(env.start_pos, env.goal_pos)

    records: List[EpisodeRecord] = []
    for idx in range(episodes):
        rec = run_episode(env, agent, idx + 1, max_turns=max_turns,
                          record_trajectory=record_trajectory)
        records.append(rec)
    return AggregateMetrics(episodes=records)
