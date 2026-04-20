from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from environment import Action

Position = Tuple[int, int]


def neighbors(
    cells: np.ndarray,
    right_blocked: np.ndarray,
    down_blocked: np.ndarray,
    pos: Position,
) -> List[Position]:
    rows, cols = cells.shape
    r, c = pos
    out: List[Position] = []

    if r > 0 and down_blocked[r - 1, c] == 0 and cells[r - 1, c] == 0:
        out.append((r - 1, c))
    if r < rows - 1 and down_blocked[r, c] == 0 and cells[r + 1, c] == 0:
        out.append((r + 1, c))
    if c > 0 and right_blocked[r, c - 1] == 0 and cells[r, c - 1] == 0:
        out.append((r, c - 1))
    if c < cols - 1 and right_blocked[r, c] == 0 and cells[r, c + 1] == 0:
        out.append((r, c + 1))

    return out


def bfs_path(
    cells: np.ndarray,
    right_blocked: np.ndarray,
    down_blocked: np.ndarray,
    start: Position,
    goal: Position,
) -> Optional[List[Position]]:
    q = deque([start])
    parent: Dict[Position, Optional[Position]] = {start: None}

    while q:
        cur = q.popleft()

        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]

        for nxt in neighbors(cells, right_blocked, down_blocked, cur):
            if nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)

    return None


def positions_to_actions(path: List[Position]) -> List[Action]:
    actions: List[Action] = []

    for i in range(1, len(path)):
        r1, c1 = path[i - 1]
        r2, c2 = path[i]

        dr = r2 - r1
        dc = c2 - c1

        if dr == -1 and dc == 0:
            actions.append(Action.MOVE_UP)
        elif dr == 1 and dc == 0:
            actions.append(Action.MOVE_DOWN)
        elif dr == 0 and dc == -1:
            actions.append(Action.MOVE_LEFT)
        elif dr == 0 and dc == 1:
            actions.append(Action.MOVE_RIGHT)
        else:
            raise ValueError(f"Invalid adjacent step from {path[i - 1]} to {path[i]}.")

    return actions


def chunk_actions(actions: List[Action], chunk_size: int = 5) -> List[List[Action]]:
    return [actions[i:i + chunk_size] for i in range(0, len(actions), chunk_size)]