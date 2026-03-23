from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Tuple

import numpy as np

from maze_config import START_POS, GOAL_POS
from hazards import validate_open_cell, generate_hazard_layout

Position = Tuple[int, int]


class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4


@dataclass
class TurnResult:
    wall_hits: int = 0
    current_position: Position = (0, 0)
    is_dead: bool = False
    is_confused: bool = False
    is_goal_reached: bool = False
    teleported: bool = False
    actions_executed: int = 0


class MazeEnvironment:
    def __init__(self, hazards_enabled: bool = False):
        self.cells = np.load("maze_cells.npy")
        self.right_blocked = np.load("maze_walls_right.npy")
        self.down_blocked = np.load("maze_walls_down.npy")

        self.rows, self.cols = self.cells.shape
        self.start_pos = START_POS
        self.goal_pos = GOAL_POS
        self.hazards_enabled = hazards_enabled

        validate_open_cell(self.cells, self.start_pos, "START_POS")
        validate_open_cell(self.cells, self.goal_pos, "GOAL_POS")

        layout = generate_hazard_layout(self.cells, self.right_blocked, self.down_blocked)
        self.death_pits: Set[Position] = set(layout["death_pits"])
        self.confusion_pads: Set[Position] = set(layout["confusion_pads"])
        self.teleport_map: Dict[Position, Position] = dict(layout["teleport_map"])
        self.safe_path: List[Position] = list(layout["safe_path"])
        self.hazard_summary: dict = dict(layout["summary"])
        self.featured_hazards: dict = dict(layout["featured"])

        self.reset()

    def reset(self) -> Position:
        self.agent_pos: Position = self.start_pos
        self.turn_counter: int = 0
        self.death_counter: int = 0
        self.confused_counter: int = 0
        self.goal_reached: bool = False
        self.cells_explored: Set[Position] = {self.start_pos}
        self.confusion_turns_remaining: int = 0
        return self.agent_pos

    def in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_wall_cell(self, pos: Position) -> bool:
        r, c = pos
        return self.cells[r, c] == 1

    def _invert_action(self, action: Action) -> Action:
        if action == Action.MOVE_UP:
            return Action.MOVE_DOWN
        if action == Action.MOVE_DOWN:
            return Action.MOVE_UP
        if action == Action.MOVE_LEFT:
            return Action.MOVE_RIGHT
        if action == Action.MOVE_RIGHT:
            return Action.MOVE_LEFT
        return Action.WAIT

    def can_move(self, pos: Position, action: Action) -> bool:
        r, c = pos

        if action == Action.WAIT:
            return True

        if action == Action.MOVE_UP:
            if r == 0:
                return False
            return self.down_blocked[r - 1, c] == 0

        if action == Action.MOVE_DOWN:
            if r == self.rows - 1:
                return False
            return self.down_blocked[r, c] == 0

        if action == Action.MOVE_LEFT:
            if c == 0:
                return False
            return self.right_blocked[r, c - 1] == 0

        if action == Action.MOVE_RIGHT:
            if c == self.cols - 1:
                return False
            return self.right_blocked[r, c] == 0

        return False

    def next_position(self, pos: Position, action: Action) -> Position:
        r, c = pos
        if action == Action.MOVE_UP:
            return (r - 1, c)
        if action == Action.MOVE_DOWN:
            return (r + 1, c)
        if action == Action.MOVE_LEFT:
            return (r, c - 1)
        if action == Action.MOVE_RIGHT:
            return (r, c + 1)
        return (r, c)

    def step(self, actions: List[Action]) -> TurnResult:
        if not actions or len(actions) > 5:
            raise ValueError("Actions list must contain between 1 and 5 actions.")

        result = TurnResult(current_position=self.agent_pos)

        for i, action in enumerate(actions, start=1):
            currently_confused = self.confusion_turns_remaining > 0
            actual_action = self._invert_action(action) if currently_confused else action

            if currently_confused:
                result.is_confused = True

            if not self.can_move(self.agent_pos, actual_action):
                result.wall_hits += 1
                result.actions_executed = i
                result.current_position = self.agent_pos
                continue

            self.agent_pos = self.next_position(self.agent_pos, actual_action)
            self.cells_explored.add(self.agent_pos)
            result.current_position = self.agent_pos
            result.actions_executed = i

            if self.agent_pos == self.goal_pos:
                self.goal_reached = True
                result.is_goal_reached = True
                break

            if not self.hazards_enabled:
                continue

            if self.agent_pos in self.death_pits:
                self.death_counter += 1
                result.is_dead = True
                result.current_position = self.agent_pos
                break

            visited_teleports = set()
            while self.agent_pos in self.teleport_map:
                if self.agent_pos in visited_teleports:
                    break
                visited_teleports.add(self.agent_pos)

                self.agent_pos = self.teleport_map[self.agent_pos]
                self.cells_explored.add(self.agent_pos)
                result.teleported = True
                result.current_position = self.agent_pos

                if self.agent_pos == self.goal_pos:
                    self.goal_reached = True
                    result.is_goal_reached = True
                    break

                if self.agent_pos in self.death_pits:
                    self.death_counter += 1
                    result.is_dead = True
                    break

            if result.is_goal_reached or result.is_dead:
                break

            if self.agent_pos in self.confusion_pads:
                self.confused_counter += 1
                result.is_confused = True
                self.confusion_turns_remaining = max(self.confusion_turns_remaining, 2)

        self.turn_counter += 1

        if self.confusion_turns_remaining > 0:
            self.confusion_turns_remaining -= 1

        if result.is_dead:
            self.agent_pos = self.start_pos
            self.cells_explored.add(self.agent_pos)

        return result

    def get_episode_stats(self) -> dict:
        return {
            "turns_taken": self.turn_counter,
            "deaths": self.death_counter,
            "confused": self.confused_counter,
            "cells_explored": len(self.cells_explored),
            "goal_reached": self.goal_reached,
        }