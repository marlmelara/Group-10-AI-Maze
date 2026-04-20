"""
Spec-compliant MazeEnvironment for COSC 4368 project.

Implements the API from Section 6 of the project specification:
  - Action enum (MOVE_UP/DOWN/LEFT/RIGHT/WAIT)
  - TurnResult dataclass
  - MazeEnvironment(maze_id).reset(), .step(actions) -> TurnResult, .get_episode_stats()

Additional mechanics from the spec:
  - Rotating death-pit clusters: every 5 cumulative actions the V-shaped fires
    rotate 90 degrees CW around the pivot (tip of the V).
  - Teleport pads move the agent to a deterministic destination; chains are
    followed but bounded.
  - Confusion pads invert movement for the remainder of the current turn and
    the whole next turn.
  - Deaths respawn the agent at the start next turn and increment death count.
  - Wind hazards (maze-gamma): impassable cells.  Attempting to step onto a
    wind cell pushes the agent one step in the arrow's direction instead, then
    forces any remaining actions of that turn to be the wind direction.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from maze_parser import ParsedMaze, GRID_SIZE

Position = Tuple[int, int]


class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4


_MOVE_DELTAS: Dict[Action, Tuple[int, int]] = {
    Action.MOVE_UP: (-1, 0),
    Action.MOVE_DOWN: (1, 0),
    Action.MOVE_LEFT: (0, -1),
    Action.MOVE_RIGHT: (0, 1),
    Action.WAIT: (0, 0),
}

_INVERT: Dict[Action, Action] = {
    Action.MOVE_UP: Action.MOVE_DOWN,
    Action.MOVE_DOWN: Action.MOVE_UP,
    Action.MOVE_LEFT: Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
    Action.WAIT: Action.WAIT,
}

_WIND_DIR_TO_ACTION: Dict[str, Action] = {
    "UP": Action.MOVE_UP,
    "DOWN": Action.MOVE_DOWN,
    "LEFT": Action.MOVE_LEFT,
    "RIGHT": Action.MOVE_RIGHT,
}


@dataclass
class TurnResult:
    wall_hits: int = 0
    current_position: Position = (0, 0)
    is_dead: bool = False
    is_confused: bool = False
    is_goal_reached: bool = False
    teleported: bool = False
    actions_executed: int = 0
    # Extra fields we track for metrics; not required by the spec but helpful.
    hit_wind: bool = False


class MazeEnvironment:
    """Maze environment matching the project spec API."""

    ROTATION_PERIOD_ACTIONS = 5        # fires rotate every 5 cumulative actions
    MAX_TURNS_PER_EPISODE = 10_000

    def __init__(self, maze_id: str, enable_rotation: bool = True,
                 enable_wind: Optional[bool] = None):
        """
        Args:
            maze_id: 'alpha' | 'beta' | 'gamma'
            enable_rotation: whether fire V-clusters rotate every 5 actions
            enable_wind: whether to use wind hazards (default = True iff gamma)
        """
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        parsed_path = os.path.join(root, "data", maze_id, "parsed.npz")
        self.parsed = ParsedMaze.load(parsed_path)
        self.maze_id = maze_id
        self.enable_rotation = enable_rotation
        self.enable_wind = (maze_id == "gamma") if enable_wind is None else enable_wind

        self.rows = self.cols = GRID_SIZE
        self.start_pos: Position = self.parsed.start
        self.goal_pos: Position = self.parsed.goal

        # Pre-compute rotations for each fire cluster.
        # self._fire_rotations[cluster_idx] is a list of 4 frozensets, one per
        # 90-degree rotation step (0, 90, 180, 270 CW).
        self._fire_rotations: List[List[Set[Position]]] = []
        for cluster in self.parsed.fire_groups:
            if not cluster:
                continue
            pivot = cluster[0]
            offsets = [(p[0] - pivot[0], p[1] - pivot[1]) for p in cluster]
            rotations: List[Set[Position]] = []
            for step in range(4):
                positions: Set[Position] = set()
                for dr, dc in offsets:
                    for _ in range(step):
                        dr, dc = dc, -dr    # 90 CW
                    nr = pivot[0] + dr
                    nc = pivot[1] + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        positions.add((nr, nc))
                rotations.append(positions)
            self._fire_rotations.append(rotations)

        # Non-rotating death pits (fires not part of any V cluster).
        rotating_cells = {p for g in self.parsed.fire_groups for p in g}
        self._static_death_pits: Set[Position] = set(self.parsed.death_pits) - rotating_cells

        self.teleport_map: Dict[Position, Position] = dict(zip(
            self.parsed.teleport_sources, self.parsed.teleport_dests
        ))
        self.confusion_pads: Set[Position] = set(self.parsed.confusion_pads)
        self.wind_cells: Dict[Position, str] = dict(self.parsed.wind_cells) if self.enable_wind else {}

        self.reset()

    # ------------------------------------------------------------------
    # Spec-required API
    # ------------------------------------------------------------------

    def reset(self) -> Position:
        self.agent_pos: Position = self.start_pos
        self.turn_counter: int = 0
        self.action_counter: int = 0
        self.death_counter: int = 0
        self.confused_counter: int = 0
        self.teleport_counter: int = 0
        self.wall_hits_total: int = 0
        self.wind_hits: int = 0
        self.goal_reached: bool = False
        self.cells_explored: Set[Position] = {self.start_pos}
        self.confusion_turns_remaining: int = 0
        self._pending_wind_dir: Optional[Action] = None
        return self.agent_pos

    def step(self, actions: List[Action]) -> TurnResult:
        if not actions or len(actions) > 5:
            raise ValueError("Actions list must contain between 1 and 5 actions.")

        result = TurnResult(current_position=self.agent_pos)
        turn_started_confused = self.confusion_turns_remaining > 0

        for i, requested in enumerate(actions, start=1):
            # Rotating hazards tick on every executed action.
            current_pits = self._active_death_pits()

            # Determine the effective action after confusion + wind overrides.
            action = requested
            if self._pending_wind_dir is not None:
                action = self._pending_wind_dir
            if self.confusion_turns_remaining > 0:
                action = _INVERT[action]
                result.is_confused = True

            moved_pos = self._try_move(self.agent_pos, action, result)
            if moved_pos is None:
                # WAIT action: no movement, no wall hit, still counts.
                pass
            else:
                self.agent_pos = moved_pos
                self.cells_explored.add(self.agent_pos)

            result.current_position = self.agent_pos
            result.actions_executed = i
            self.action_counter += 1

            if self.agent_pos == self.goal_pos:
                self.goal_reached = True
                result.is_goal_reached = True
                break

            # Death pit check.
            if self.agent_pos in current_pits:
                self.death_counter += 1
                result.is_dead = True
                break

            # Teleport chain (deterministic destinations).
            visited_tele: Set[Position] = set()
            while self.agent_pos in self.teleport_map and self.agent_pos not in visited_tele:
                visited_tele.add(self.agent_pos)
                self.agent_pos = self.teleport_map[self.agent_pos]
                self.cells_explored.add(self.agent_pos)
                self.teleport_counter += 1
                result.teleported = True
                result.current_position = self.agent_pos
                if self.agent_pos == self.goal_pos:
                    self.goal_reached = True
                    result.is_goal_reached = True
                    break
                if self.agent_pos in current_pits:
                    self.death_counter += 1
                    result.is_dead = True
                    break

            if result.is_goal_reached or result.is_dead:
                break

            # Confusion pad trigger.
            if self.agent_pos in self.confusion_pads:
                self.confused_counter += 1
                result.is_confused = True
                # rest of this turn + next turn inverted
                self.confusion_turns_remaining = max(self.confusion_turns_remaining, 2)

        self.wall_hits_total += result.wall_hits
        self.turn_counter += 1

        # Decrement confusion counter once per turn end.
        if self.confusion_turns_remaining > 0:
            self.confusion_turns_remaining -= 1

        # Wind override only lasts for the single turn that triggered it.
        self._pending_wind_dir = None

        if result.is_dead:
            self.agent_pos = self.start_pos
            self.cells_explored.add(self.start_pos)

        return result

    def get_episode_stats(self) -> dict:
        return {
            "turns_taken": self.turn_counter,
            "actions_taken": self.action_counter,
            "deaths": self.death_counter,
            "confused": self.confused_counter,
            "teleports": self.teleport_counter,
            "wall_hits": self.wall_hits_total,
            "wind_hits": self.wind_hits,
            "cells_explored": len(self.cells_explored),
            "goal_reached": self.goal_reached,
        }

    # ------------------------------------------------------------------
    # Helpers for agents (read-only views).  An agent does NOT have to
    # call these; they just mirror what the spec already gives the agent
    # via TurnResult.
    # ------------------------------------------------------------------

    def total_navigable_cells(self) -> int:
        """Number of cells not permanently blocked (used for map_completeness)."""
        blocked = len(self.wind_cells)
        # All 64x64 cells minus permanent wind cells.
        return GRID_SIZE * GRID_SIZE - blocked

    def snapshot_fires(self) -> Set[Position]:
        return self._active_death_pits()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _active_death_pits(self) -> Set[Position]:
        pits = set(self._static_death_pits)
        if not self.enable_rotation:
            for cluster in self.parsed.fire_groups:
                pits.update(cluster)
            return pits
        step = (self.action_counter // self.ROTATION_PERIOD_ACTIONS) % 4
        for rotations in self._fire_rotations:
            pits.update(rotations[step])
        return pits

    def _can_move_through_walls(self, pos: Position, action: Action) -> bool:
        r, c = pos
        if action == Action.WAIT:
            return True
        if action == Action.MOVE_UP:
            return r > 0 and self.parsed.down_blocked[r - 1, c] == 0
        if action == Action.MOVE_DOWN:
            return r < self.rows - 1 and self.parsed.down_blocked[r, c] == 0
        if action == Action.MOVE_LEFT:
            return c > 0 and self.parsed.right_blocked[r, c - 1] == 0
        if action == Action.MOVE_RIGHT:
            return c < self.cols - 1 and self.parsed.right_blocked[r, c] == 0
        return False

    def _target_of(self, pos: Position, action: Action) -> Position:
        dr, dc = _MOVE_DELTAS[action]
        return (pos[0] + dr, pos[1] + dc)

    def _try_move(self, pos: Position, action: Action, result: TurnResult) -> Optional[Position]:
        """Attempt a single-step action.  Returns the new position if the agent
        moved, or None if this was a WAIT / wall bounce / wind deflection that
        kept the agent in place (or was already handled)."""
        if action == Action.WAIT:
            return None

        if not self._can_move_through_walls(pos, action):
            result.wall_hits += 1
            return None

        target = self._target_of(pos, action)

        # Wind hazard: target cell is impassable, agent gets deflected.
        if target in self.wind_cells:
            wind_dir = _WIND_DIR_TO_ACTION[self.wind_cells[target]]
            result.hit_wind = True
            self.wind_hits += 1
            # For the rest of this turn, all actions are forced to wind_dir.
            self._pending_wind_dir = wind_dir
            # Immediate push one step in wind direction if possible.
            if self._can_move_through_walls(pos, wind_dir):
                new_target = self._target_of(pos, wind_dir)
                if new_target not in self.wind_cells:
                    return new_target
            # Blocked push = stay in place (bumping into wall).
            result.wall_hits += 1
            return None

        return target
