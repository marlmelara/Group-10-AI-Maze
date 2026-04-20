"""
Hybrid RL + classical-search agent for the maze-with-hazards task.

Design
------
The agent has no sensor input - it must discover the maze through attempted
moves and interpret TurnResult feedback.  We combine two ideas:

  A) Online mapping (classical)
     - Walls: per-edge trit (-1 unknown / 0 open / 1 wall).
     - Teleport pairs: recorded when a teleport fires.
     - Confusion pads: recorded when is_confused first appears on a cell.
     - Death-pit cells recorded per rotation phase when the agent dies.

  B) Reinforcement-learning risk shaping
     - Tabular Q[r,c,phase] grows when the agent dies near (r,c); TD(0)
       spreads the signal to Manhattan-neighbours with geometric decay.
     - Q is decayed between episodes so stale fear doesn't accumulate.
     - The RiskConfig hyper-parameters (learning rate, propagation decay,
       death penalty, Q weight, Q decay, etc.) are what we carry between
       mazes when we evaluate transfer from alpha to beta / gamma.

Planner (classical search on the learned map)
---------------------------------------------
Each turn we run a weighted A* with Manhattan heuristic.  Edge cost:
    c(n, phi) = 1
              + fire_hard  if n is a known fire at the arrival phase
              + fire_soft  if n is a known fire at ANY phase
              + conf_cost  if n is a confusion pad
              + w_Q * Q[n, phi]
              + visit_pen  (anti-oscillation)
              + unknown    (slight cost to traverse unknown edges)

We also keep a stored successful path from the first goal-reaching episode
and replay matching prefixes on later episodes, because retrying a known
safe plan is cheaper than rerunning A*.  If the stored path is blocked
(wall discovered, fire at the arrival phase) we fall back to A*.

When the agent gets stuck we progressively relax the fire penalties,
then switch to frontier-directed search to escape local minima.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from environment import Action, TurnResult

Position = Tuple[int, int]
GRID = 64

ACTIONS_ORDER = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]
ACTION_DELTAS: Dict[Action, Tuple[int, int]] = {
    Action.MOVE_UP: (-1, 0),
    Action.MOVE_DOWN: (1, 0),
    Action.MOVE_LEFT: (0, -1),
    Action.MOVE_RIGHT: (0, 1),
}
DELTA_TO_ACTION: Dict[Tuple[int, int], Action] = {v: k for k, v in ACTION_DELTAS.items()}
_INVERT: Dict[Action, Action] = {
    Action.MOVE_UP: Action.MOVE_DOWN,
    Action.MOVE_DOWN: Action.MOVE_UP,
    Action.MOVE_LEFT: Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
    Action.WAIT: Action.WAIT,
}


@dataclass
class RiskConfig:
    fire_current_phase_hard: float = 1e6  # block fires at arrival phase
    fire_any_phase_cost: float = 20.0
    confusion_cost: float = 6.0
    teleport_unknown_cost: float = 8.0
    teleport_progress_bonus: float = -3.0
    unknown_edge_cost: float = 0.3
    visit_penalty_coef: float = 0.08
    death_penalty: float = 60.0
    td_alpha: float = 0.4
    td_gamma: float = 0.8
    q_episode_decay: float = 0.25
    q_weight: float = 0.02


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _dedupe_cycles(trajectory: List[Position]) -> List[Position]:
    """Remove revisit cycles: for each cell, keep only the last occurrence so
    the resulting list is a simple (non-repeating) walk from start to end.

    We scan forward: if we see a cell we've seen before, we cut the list back
    to the earlier occurrence and continue from there.  Equivalent to popping
    the stack whenever we revisit.
    """
    simple: List[Position] = []
    seen: Dict[Position, int] = {}
    for pos in trajectory:
        if pos in seen:
            # Cut back to and including that point.
            idx = seen[pos]
            for removed in simple[idx + 1:]:
                seen.pop(removed, None)
            simple = simple[:idx + 1]
        else:
            seen[pos] = len(simple)
            simple.append(pos)
    return simple


class HybridRLAgent:
    def __init__(self, risk_config: Optional[RiskConfig] = None):
        self.cfg = risk_config or RiskConfig()
        self.reset_for_new_maze()

    # ----- lifecycle --------------------------------------------------
    def reset_for_new_maze(self):
        """Forget maze-specific memory. Hyperparameters persist."""
        self.right_blocked = np.full((GRID, GRID), -1, dtype=np.int8)
        self.down_blocked = np.full((GRID, GRID), -1, dtype=np.int8)
        self.known_empty: Set[Position] = set()
        self.fires_by_phase: List[Set[Position]] = [set() for _ in range(4)]
        self.fires_any_phase: Set[Position] = set()
        self.confusion_known: Set[Position] = set()
        self.teleport_map: Dict[Position, Position] = {}
        self.teleport_destinations: Set[Position] = set()
        self.start_pos: Optional[Position] = None
        self.goal_pos: Optional[Position] = None
        self.Q = np.zeros((GRID, GRID, 4), dtype=np.float32)
        self.successful_paths: List[List[Position]] = []
        self.total_episodes = 0
        self.reset_episode()

    def reset_episode(self):
        self.pos: Optional[Position] = None
        self.prev_pos: Optional[Position] = None
        self.action_counter = 0
        self.turn_counter = 0
        self.confusion_turns_remaining = 0
        self.last_submitted: List[Action] = []
        self.pending_actions: List[Action] = []
        self.pending_target: Optional[Position] = None
        self.visit_counts: Dict[Position, int] = {}
        self.recent_positions: deque = deque(maxlen=24)
        self.stuck_counter = 0
        self.consecutive_waits = 0
        self.episode_trajectory: List[Position] = []
        if hasattr(self, 'Q') and getattr(self, 'total_episodes', 0) > 0:
            self.Q *= self.cfg.q_episode_decay
        self.total_episodes = getattr(self, 'total_episodes', 0) + 1

    # ----- public API -------------------------------------------------
    def set_start_goal(self, start: Position, goal: Position):
        self.start_pos = start
        self.goal_pos = goal
        self.pos = start
        self.known_empty.add(start)

    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        if last_result is not None:
            self._integrate(last_result)
        elif self.pos is None and self.start_pos is not None:
            self.pos = self.start_pos
            self.known_empty.add(self.pos)
        if self.pos is None:
            return [Action.WAIT]

        plan = self._plan()
        if not plan:
            plan = [Action.WAIT]

        # If this turn will be under confusion, submit the INVERSE so
        # the net motion matches the plan.  Submit one action at a time
        # under confusion for reliable interpretation.
        if self.confusion_turns_remaining > 0:
            plan = [_INVERT[a] for a in plan[:1]]

        self.last_submitted = plan[:]
        return plan

    # ----- integration ------------------------------------------------
    def _integrate(self, result: TurnResult):
        if self.pos is None:
            self.pos = result.current_position
            self.known_empty.add(self.pos)

        self.action_counter += result.actions_executed
        self._replay(self.last_submitted, result)

        if result.is_dead:
            death_cell = result.current_position
            phase = self._phase_at(max(0, self.action_counter - 1))
            # Record the fire at the exact phase we died at.  This preserves
            # the possibility that the same cell is safe at other phases -
            # essential for mazes where the only corridor to the goal is
            # blocked by a V-cluster whose rotations cover the corridor at
            # some phases but not others.
            self.fires_by_phase[phase].add(death_cell)
            self.fires_any_phase.add(death_cell)
            self._td_death_update(death_cell, phase)

        # Confusion tracking: set to 2 only when we stepped on a known pad
        # THIS turn - otherwise an ongoing confusion period would reset.
        stepped_on_new_pad = (result.current_position in self.confusion_known
                              and result.is_confused
                              and self.confusion_turns_remaining <= 1)
        if stepped_on_new_pad:
            self.confusion_turns_remaining = 2
        if self.confusion_turns_remaining > 0:
            self.confusion_turns_remaining -= 1

        # Stuck detection via bounded recent-positions window.
        self.recent_positions.append(result.current_position)
        distinct_recent = len(set(self.recent_positions))
        if (len(self.recent_positions) == self.recent_positions.maxlen
                and distinct_recent <= 5):
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 2)

        self.prev_pos = self.pos
        self.pos = result.current_position if not result.is_dead else self.start_pos
        self.known_empty.add(self.pos)
        self.visit_counts[self.pos] = self.visit_counts.get(self.pos, 0) + 1
        self.episode_trajectory.append(self.pos)

        if result.is_goal_reached:
            if self.goal_pos is None:
                self.goal_pos = result.current_position
            if self.episode_trajectory:
                cleaned = _dedupe_cycles(self.episode_trajectory)
                self.successful_paths.append(cleaned)
                self.successful_paths.sort(key=len)
                self.successful_paths = self.successful_paths[:2]

        # Invalidate cached plan on any unexpected event.
        if (result.wall_hits > 0 or result.is_dead or result.teleported
                or result.is_confused or result.hit_wind):
            self.pending_actions = []
            self.pending_target = None

        self.turn_counter += 1

    def _replay(self, submitted: List[Action], result: TurnResult):
        """Re-simulate a batch to update our map."""
        cursor = self.pos
        end_pos = result.current_position
        executed = min(result.actions_executed, len(submitted))
        wall_hits_left = result.wall_hits
        confused_at_start = self.confusion_turns_remaining > 0

        for i in range(executed):
            action = submitted[i]
            if confused_at_start:
                action = _INVERT[action]
            is_last = (i == executed - 1)
            if action == Action.WAIT:
                continue

            dr, dc = ACTION_DELTAS[action]
            target = (cursor[0] + dr, cursor[1] + dc)
            if not (0 <= target[0] < GRID and 0 <= target[1] < GRID):
                self._mark_wall(cursor, action)
                continue

            if not is_last:
                # Interior actions are pre-validated to be known-open.
                self._mark_open(cursor, action)
                cursor = target
                self.known_empty.add(cursor)
                continue

            # Last action: decide outcome.
            if result.is_dead:
                self._mark_open(cursor, action)
                cursor = target
                self.known_empty.add(cursor)
            elif result.teleported:
                self._mark_open(cursor, action)
                cursor = target
                self.known_empty.add(cursor)
                if cursor != end_pos:
                    self.teleport_map[cursor] = end_pos
                    self.teleport_destinations.add(end_pos)
                    self.known_empty.add(end_pos)
                    cursor = end_pos
            elif result.hit_wind:
                self._mark_wall(cursor, action)
            elif wall_hits_left > 0 and target != end_pos:
                self._mark_wall(cursor, action)
                wall_hits_left -= 1
            else:
                self._mark_open(cursor, action)
                cursor = target
                self.known_empty.add(cursor)

        if result.is_confused and not confused_at_start:
            self.confusion_known.add(cursor)

    def _mark_wall(self, pos: Position, action: Action):
        r, c = pos
        if action == Action.MOVE_UP and r > 0:
            self.down_blocked[r - 1, c] = 1
        elif action == Action.MOVE_DOWN and r < GRID - 1:
            self.down_blocked[r, c] = 1
        elif action == Action.MOVE_LEFT and c > 0:
            self.right_blocked[r, c - 1] = 1
        elif action == Action.MOVE_RIGHT and c < GRID - 1:
            self.right_blocked[r, c] = 1

    def _mark_open(self, pos: Position, action: Action):
        r, c = pos
        if action == Action.MOVE_UP and r > 0:
            self.down_blocked[r - 1, c] = 0
        elif action == Action.MOVE_DOWN and r < GRID - 1:
            self.down_blocked[r, c] = 0
        elif action == Action.MOVE_LEFT and c > 0:
            self.right_blocked[r, c - 1] = 0
        elif action == Action.MOVE_RIGHT and c < GRID - 1:
            self.right_blocked[r, c] = 0

    def _phase_at(self, action_index: int) -> int:
        return (action_index // 5) % 4

    def _td_death_update(self, center: Position, phase: int):
        self.Q[center[0], center[1], phase] += self.cfg.td_alpha * (
            self.cfg.death_penalty - self.Q[center[0], center[1], phase])
        r0, c0 = center
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r0 + dr, c0 + dc
                if not (0 <= nr < GRID and 0 <= nc < GRID):
                    continue
                decay = self.cfg.td_gamma ** (abs(dr) + abs(dc))
                self.Q[nr, nc, phase] += self.cfg.td_alpha * decay * (
                    self.cfg.death_penalty - self.Q[nr, nc, phase])

    # ----- edge helpers ----------------------------------------------
    def _edge_wall(self, cell: Position, action: Action) -> bool:
        r, c = cell
        if action == Action.MOVE_UP:
            return r == 0 or self.down_blocked[r - 1, c] == 1
        if action == Action.MOVE_DOWN:
            return r == GRID - 1 or self.down_blocked[r, c] == 1
        if action == Action.MOVE_LEFT:
            return c == 0 or self.right_blocked[r, c - 1] == 1
        if action == Action.MOVE_RIGHT:
            return c == GRID - 1 or self.right_blocked[r, c] == 1
        return False

    def _edge_unknown(self, cell: Position, action: Action) -> bool:
        r, c = cell
        if action == Action.MOVE_UP:
            return r > 0 and self.down_blocked[r - 1, c] == -1
        if action == Action.MOVE_DOWN:
            return r < GRID - 1 and self.down_blocked[r, c] == -1
        if action == Action.MOVE_LEFT:
            return c > 0 and self.right_blocked[r, c - 1] == -1
        if action == Action.MOVE_RIGHT:
            return c < GRID - 1 and self.right_blocked[r, c] == -1
        return False

    def _cell_has_unknown_edge(self, pos: Position) -> bool:
        for a in ACTIONS_ORDER:
            if self._edge_unknown(pos, a):
                return True
        return False

    # ----- search -----------------------------------------------------
    MAX_PLAN_NODES = 3000
    MAX_PLAN_WALL_SECONDS = 0.05  # 50 ms per search call

    def _search(self, target: Optional[Position], frontier_mode: bool,
                fire_block_any_phase: bool,
                fire_scale: float = 1.0,
                visit_scale: float = 1.0) -> List[Action]:
        """A* with Manhattan heuristic. Returns list of actions."""
        if self.pos is None:
            return []
        start = self.pos
        cur_phase = self._phase_at(self.action_counter)

        def h(pos: Position) -> float:
            return 0.0 if target is None else float(_manhattan(pos, target))

        pq: List[Tuple[float, int, Position]] = []
        counter = 0
        heapq.heappush(pq, (h(start), counter, start))
        best: Dict[Position, float] = {start: 0.0}
        came_from: Dict[Position, Tuple[Position, Action]] = {}
        closed: Set[Position] = set()
        reached: Optional[Position] = None
        max_nodes = self.MAX_PLAN_NODES
        nodes = 0
        import time as _time
        t_start = _time.time()

        while pq and nodes < max_nodes:
            if nodes % 256 == 0 and _time.time() - t_start > self.MAX_PLAN_WALL_SECONDS:
                break
            f_cost, _, pos = heapq.heappop(pq)
            if pos in closed:
                continue
            closed.add(pos)
            g = best[pos]
            nodes += 1

            if frontier_mode:
                if pos != start and self._cell_has_unknown_edge(pos):
                    reached = pos
                    break
            elif pos == target:
                reached = pos
                break

            for action in ACTIONS_ORDER:
                if self._edge_wall(pos, action):
                    continue
                unknown = self._edge_unknown(pos, action)
                dr, dc = ACTION_DELTAS[action]
                nxt = (pos[0] + dr, pos[1] + dc)

                depth = max(0, int(g))
                arrival_phase = ((self.action_counter + depth + 1) // 5) % 4
                # Hard block fires at the exact arrival phase for near steps.
                if nxt in self.fires_by_phase[arrival_phase]:
                    if depth == 0:
                        continue
                    cell_cost = 1.0 + self.cfg.fire_any_phase_cost * fire_scale
                else:
                    cell_cost = 1.0

                if fire_block_any_phase and nxt in self.fires_any_phase:
                    continue
                elif nxt in self.fires_any_phase:
                    cell_cost += self.cfg.fire_any_phase_cost * fire_scale

                if nxt in self.confusion_known:
                    cell_cost += self.cfg.confusion_cost
                if nxt in self.teleport_map:
                    dest = self.teleport_map[nxt]
                    if (self.goal_pos and
                            _manhattan(dest, self.goal_pos) < _manhattan(nxt, self.goal_pos) - 3):
                        cell_cost += self.cfg.teleport_progress_bonus
                    else:
                        cell_cost += self.cfg.teleport_unknown_cost
                cell_cost += (self.cfg.q_weight *
                              float(self.Q[nxt[0], nxt[1], cur_phase]) * fire_scale)
                if unknown:
                    cell_cost += self.cfg.unknown_edge_cost
                visits = self.visit_counts.get(nxt, 0)
                cell_cost += min(3.5, self.cfg.visit_penalty_coef * visits) * visit_scale

                # A* soundness: edge costs must be strictly positive to avoid
                # negative-weight cycles in `came_from` (which would cause the
                # reconstruction loop below to spin forever).  The teleport
                # progress bonus can drive cell_cost negative, so floor here.
                if cell_cost < 0.05:
                    cell_cost = 0.05

                new_g = g + cell_cost
                if new_g < best.get(nxt, float('inf')):
                    best[nxt] = new_g
                    came_from[nxt] = (pos, action)
                    counter += 1
                    heapq.heappush(pq, (new_g + h(nxt), counter, nxt))

        if reached is None:
            return []
        # Reconstruct with an explicit cycle guard.  A pathological update
        # pattern could still produce a came_from cycle; never spin forever.
        actions: List[Action] = []
        cur = reached
        seen_recon: Set[Position] = set()
        while cur in came_from:
            if cur in seen_recon:
                return []
            seen_recon.add(cur)
            prev_pos, act = came_from[cur]
            actions.append(act)
            cur = prev_pos
        actions.reverse()
        return actions

    # ----- known-map BFS short-circuit --------------------------------
    def _known_map_bfs(self) -> Optional[List[Action]]:
        """Layered BFS with progressively weaker safety:

          1. Fast: position-only BFS, block all fires_any_phase.
          2. Fast: position-only BFS, only block fires at the current phase.
          3. Slow fallback: time-indexed BFS with WAIT action, block fires
             at each arrival phase (this lets the agent WAIT out rotations).
        """
        if self.pos is None or self.goal_pos is None:
            return None

        cur_phase = self._phase_at(self.action_counter)

        def simple_bfs(block_any_phase: bool) -> Optional[List[Action]]:
            start = self.pos
            q: deque = deque([start])
            parent: Dict[Position, Tuple[Position, Action]] = {start: (start, Action.WAIT)}
            goal = self.goal_pos
            while q:
                pos = q.popleft()
                if pos == goal:
                    break
                for action in ACTIONS_ORDER:
                    if self._edge_wall(pos, action):
                        continue
                    if self._edge_unknown(pos, action):
                        continue
                    dr, dc = ACTION_DELTAS[action]
                    nxt = (pos[0] + dr, pos[1] + dc)
                    if nxt in self.teleport_map:
                        nxt = self.teleport_map[nxt]
                    if nxt in parent:
                        continue
                    if nxt in self.fires_by_phase[cur_phase]:
                        continue
                    if block_any_phase and nxt in self.fires_any_phase:
                        continue
                    parent[nxt] = (pos, action)
                    q.append(nxt)
            if goal not in parent:
                return None
            actions: List[Action] = []
            cur = goal
            while parent[cur][0] != cur:
                prev, act = parent[cur]
                actions.append(act)
                cur = prev
            actions.reverse()
            return actions

        # 1) Safe: no ever-fire cells.
        plan = simple_bfs(block_any_phase=True)
        if plan:
            return plan
        # 2) Relaxed: only avoid fires at current phase.
        plan = simple_bfs(block_any_phase=False)
        if plan:
            return plan
        return None

    def _time_indexed_bfs(self) -> Optional[List[Action]]:
        """Time-indexed BFS with WAIT: state = (pos, action_counter mod 20).
        Slow but handles cases where we must WAIT for fire rotation."""
        if self.pos is None or self.goal_pos is None:
            return None
        start = (self.pos, self.action_counter % 20)
        q: deque = deque([start])
        parent: Dict[Tuple[Position, int], Tuple[Tuple[Position, int], Action]] = {
            start: (start, Action.WAIT)
        }
        goal = self.goal_pos
        max_states = 40000
        states_seen = 0
        while q and states_seen < max_states:
            state = q.popleft()
            pos, t = state
            states_seen += 1
            if pos == goal:
                actions: List[Action] = []
                cur = state
                while parent[cur][0] != cur:
                    prev, act = parent[cur]
                    actions.append(act)
                    cur = prev
                actions.reverse()
                return actions
            for action in ACTIONS_ORDER + [Action.WAIT]:
                if action == Action.WAIT:
                    nxt = pos
                else:
                    if self._edge_wall(pos, action):
                        continue
                    if self._edge_unknown(pos, action):
                        continue
                    dr, dc = ACTION_DELTAS[action]
                    nxt = (pos[0] + dr, pos[1] + dc)
                    if nxt in self.teleport_map:
                        nxt = self.teleport_map[nxt]
                new_t = (t + 1) % 20
                arrival_phase = (new_t // 5) % 4
                if nxt in self.fires_by_phase[arrival_phase]:
                    continue
                ns = (nxt, new_t)
                if ns in parent:
                    continue
                parent[ns] = (state, action)
                q.append(ns)
        return None

    # ----- top-level planner -----------------------------------------
    # ----- Frontier-first planner (v2) --------------------------------
    # Explicit strategy:
    #   (1) Phase-aware BFS to goal if we already know a safe path.
    #   (2) Phase-aware BFS to the nearest frontier cell (known-open cell
    #       that still has an unknown edge) so the map grows toward goal.
    #   (3) Same as (2) but allow crossing known fires at non-fire phases.
    #   (4) As a last resort, take the least-visited adjacent open edge.
    # This guarantees the agent keeps exploring until the full reachable
    # region is mapped, even when fires block the direct line to the goal.

    def _phase_aware_bfs(self, target: Optional[Position],
                         frontier_target: bool = False,
                         block_any_phase_fires: bool = True,
                         max_states: int = 100_000) -> Optional[List[Action]]:
        """BFS over (pos, action_counter_mod_20) that respects fire rotations.
        If `frontier_target` is True, returns the path to the nearest known-open
        cell that still has at least one unknown edge.
        Only traverses known-open edges (never unknown, never walls)."""
        if self.pos is None:
            return None
        start_state = (self.pos, self.action_counter % 20)
        q: deque = deque([start_state])
        parent: Dict[Tuple[Position, int], Tuple[Tuple[Position, int], Action]] = {
            start_state: (start_state, Action.WAIT)
        }
        reached: Optional[Tuple[Position, int]] = None
        states_seen = 0
        while q and states_seen < max_states:
            state = q.popleft()
            pos, t = state
            states_seen += 1

            if frontier_target:
                if pos != self.pos and self._cell_has_unknown_edge(pos):
                    reached = state
                    break
            elif target is not None and pos == target:
                reached = state
                break

            for action in ACTIONS_ORDER + [Action.WAIT]:
                if action == Action.WAIT:
                    nxt = pos
                else:
                    if self._edge_wall(pos, action):
                        continue
                    if self._edge_unknown(pos, action):
                        continue
                    dr, dc = ACTION_DELTAS[action]
                    nxt = (pos[0] + dr, pos[1] + dc)
                    if nxt in self.teleport_map:
                        nxt = self.teleport_map[nxt]
                new_t = (t + 1) % 20
                arrival_phase = (new_t // 5) % 4
                # Hard block: arrival lands on a fire for this phase.
                if nxt in self.fires_by_phase[arrival_phase]:
                    continue
                # Optional: block cells that are fires at ANY learned phase.
                if block_any_phase_fires and nxt in self.fires_any_phase:
                    continue
                ns = (nxt, new_t)
                if ns in parent:
                    continue
                parent[ns] = (state, action)
                q.append(ns)

        if reached is None:
            return None
        actions: List[Action] = []
        cur = reached
        guard: Set[Tuple[Position, int]] = set()
        while parent[cur][0] != cur:
            if cur in guard:
                return None
            guard.add(cur)
            prev, act = parent[cur]
            actions.append(act)
            cur = prev
        actions.reverse()
        return actions

    def _fallback_move(self) -> List[Action]:
        """When no BFS plan is available, pick the least-visited adjacent
        open (or unknown) edge so we at least keep moving.  Prefer known-open
        edges over unknown ones to avoid unnecessary wall-bumps."""
        if self.pos is None:
            return [Action.WAIT]
        choices: List[Tuple[int, int, Action]] = []  # (is_unknown, visits, action)
        for action in ACTIONS_ORDER:
            if self._edge_wall(self.pos, action):
                continue
            dr, dc = ACTION_DELTAS[action]
            nxt = (self.pos[0] + dr, self.pos[1] + dc)
            if not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID):
                continue
            if nxt in self.fires_any_phase:
                continue   # skip known-dangerous cells
            is_unknown = 1 if self._edge_unknown(self.pos, action) else 0
            visits = self.visit_counts.get(nxt, 0)
            choices.append((is_unknown, visits, action))
        if choices:
            choices.sort()
            return [choices[0][2]]
        # Everything around is blocked or dangerous — try ANY open edge
        # including known fires, as a last resort.
        for action in ACTIONS_ORDER:
            if not self._edge_wall(self.pos, action):
                return [action]
        return [Action.WAIT]

    def _plan(self) -> List[Action]:
        if self.pos is None:
            return [Action.WAIT]

        # 1) Phase-aware BFS to the goal over known-open cells, avoiding
        #    every known fire at every phase.
        if self.goal_pos is not None and self.goal_pos in self.known_empty:
            plan = self._phase_aware_bfs(self.goal_pos,
                                         block_any_phase_fires=True)
            if plan:
                self.pending_target = self.goal_pos
                return self._trim_batch(plan)
            # Relax: accept crossing cells that are fires at OTHER phases.
            plan = self._phase_aware_bfs(self.goal_pos,
                                         block_any_phase_fires=False)
            if plan:
                self.pending_target = self.goal_pos
                return self._trim_batch(plan)

        # 2) No safe path to goal yet — go to the nearest frontier cell to
        #    expand the map.  Try strict fire avoidance first.
        plan = self._phase_aware_bfs(None, frontier_target=True,
                                     block_any_phase_fires=True)
        if plan:
            self.pending_target = None
            return self._trim_batch(plan)
        plan = self._phase_aware_bfs(None, frontier_target=True,
                                     block_any_phase_fires=False)
        if plan:
            self.pending_target = None
            return self._trim_batch(plan)

        # 3) Legacy fall-back (A* / visit-count chooser).  Reached only
        #    when neither a known-path-to-goal nor any reachable frontier
        #    exists - i.e. we're in an isolated closed region.
        bfs_plan = self._known_map_bfs()
        if bfs_plan:
            # NB: do NOT reset consecutive_waits here.  _trim_batch uses it
            # to bound how long we'll WAIT for a fire rotation before forcing
            # through.  Resetting it every call caused a deadlock: BFS keeps
            # returning a plan whose first step is a fire at the arrival phase,
            # _trim_batch returns [WAIT], we re-enter, BFS succeeds again,
            # counter resets, agent waits forever.
            return self._trim_batch(bfs_plan)
        # Only reset when BFS failed and we're about to try fresh planning.
        self.consecutive_waits = 0

        # Reuse previous plan when the next step is still known-safe.
        if (self.pending_actions and self.pending_target == self.goal_pos):
            next_act = self.pending_actions[0]
            if (not self._edge_wall(self.pos, next_act)
                    and not self._edge_unknown(self.pos, next_act)):
                dr, dc = ACTION_DELTAS[next_act]
                nxt = (self.pos[0] + dr, self.pos[1] + dc)
                if nxt in self.teleport_map:
                    nxt = self.teleport_map[nxt]
                phase = ((self.action_counter + 1) // 5) % 4
                if (nxt not in self.fires_by_phase[phase]
                        and nxt in self.known_empty):
                    return self._trim_batch(self.pending_actions)

        # Stuck-level progressive relaxation.
        stuck_level = 0
        if self.stuck_counter > 6: stuck_level = 1
        if self.stuck_counter > 20: stuck_level = 2
        if self.stuck_counter > 60: stuck_level = 3

        plan: List[Action] = []
        if self.goal_pos is not None and stuck_level < 3:
            plan = self._search(self.goal_pos, frontier_mode=False,
                                fire_block_any_phase=True, fire_scale=1.0,
                                visit_scale=1.0)
            self.pending_target = self.goal_pos
        if not plan and self.goal_pos is not None:
            plan = self._search(self.goal_pos, frontier_mode=False,
                                fire_block_any_phase=False,
                                fire_scale=max(0.0, 1.0 - 0.4 * stuck_level),
                                visit_scale=max(0.0, 1.0 - 0.3 * stuck_level))
            self.pending_target = self.goal_pos
        if not plan:
            plan = self._search(None, frontier_mode=True,
                                fire_block_any_phase=(stuck_level < 2),
                                fire_scale=max(0.0, 1.0 - 0.4 * stuck_level),
                                visit_scale=max(0.0, 1.0 - 0.3 * stuck_level))
            self.pending_target = None

        if not plan:
            choices: List[Tuple[int, Action]] = []
            for action in ACTIONS_ORDER:
                if self._edge_wall(self.pos, action):
                    continue
                dr, dc = ACTION_DELTAS[action]
                nxt = (self.pos[0] + dr, self.pos[1] + dc)
                if 0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID:
                    choices.append((self.visit_counts.get(nxt, 0), action))
            if choices:
                return [min(choices, key=lambda x: x[0])[1]]
            return [Action.WAIT]

        return self._trim_batch(plan)

    def _trim_batch(self, plan: List[Action]) -> List[Action]:
        cursor = self.pos
        trimmed: List[Action] = []
        for act in plan[:5]:
            if act == Action.WAIT:
                trimmed.append(act)
                continue
            if self._edge_wall(cursor, act):
                break
            dr, dc = ACTION_DELTAS[act]
            next_cell = (cursor[0] + dr, cursor[1] + dc)
            land = self.teleport_map.get(next_cell, next_cell)
            arrival_phase = ((self.action_counter + len(trimmed) + 1) // 5) % 4
            if land in self.fires_by_phase[arrival_phase]:
                if not trimmed:
                    # Only WAIT for a bounded number of consecutive turns.
                    # After that, accept the risk and step anyway.  If we
                    # die the cell is marked for this phase.
                    if self.consecutive_waits < 8:
                        self.consecutive_waits += 1
                        return [Action.WAIT]
                    # Force through; the death updates knowledge.
                    self.consecutive_waits = 0
                else:
                    break
            else:
                self.consecutive_waits = 0
            unknown = self._edge_unknown(cursor, act)
            trimmed.append(act)
            if unknown:
                break
            cursor = land
            if next_cell in self.teleport_map:
                break
            if cursor not in self.known_empty:
                break
            if cursor in self.confusion_known:
                break
        if not trimmed:
            trimmed = [plan[0]]
        self.pending_actions = plan[len(trimmed):]
        return trimmed
