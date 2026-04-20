from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from maze_config import (
    START_POS,
    GOAL_POS,
    HAZARD_SEED,
    HAZARD_DENSITY,
    PIT_RATIO,
    CONFUSION_RATIO,
    TELEPORT_RATIO,
)

Position = Tuple[int, int]
_CACHE: Optional[dict] = None


def validate_open_cell(cells: np.ndarray, pos: Position, label: str) -> None:
    rows, cols = cells.shape
    r, c = pos
    if not (0 <= r < rows and 0 <= c < cols):
        raise ValueError(f"{label} {pos} is out of bounds.")
    if cells[r, c] == 1:
        raise ValueError(f"{label} {pos} is a wall cell.")


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
    parent = {start: None}

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


def bfs_path_avoiding(
    cells: np.ndarray,
    right_blocked: np.ndarray,
    down_blocked: np.ndarray,
    start: Position,
    goal: Position,
    blocked_cells: Set[Position],
) -> Optional[List[Position]]:
    q = deque([start])
    parent = {start: None}

    while q:
        cur = q.popleft()
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]

        for nxt in neighbors(cells, right_blocked, down_blocked, cur):
            if nxt in blocked_cells and nxt != goal:
                continue
            if nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)

    return None


def distances_from(
    cells: np.ndarray,
    right_blocked: np.ndarray,
    down_blocked: np.ndarray,
    start: Position,
) -> Dict[Position, int]:
    q = deque([start])
    dist = {start: 0}

    while q:
        cur = q.popleft()
        for nxt in neighbors(cells, right_blocked, down_blocked, cur):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)

    return dist


def pick_feature_targets(
    cells: np.ndarray,
    right_blocked: np.ndarray,
    down_blocked: np.ndarray,
    safe_path: List[Position],
) -> dict:
    safe_path_set = set(safe_path)
    dist = distances_from(cells, right_blocked, down_blocked, START_POS)

    candidates = sorted(
        [p for p in dist if p not in safe_path_set and p != START_POS and p != GOAL_POS],
        key=lambda p: (dist[p], p[0], p[1]),
    )

    if len(candidates) < 6:
        raise ValueError("Not enough candidate cells to place featured hazards.")

    # 1) Featured pit
    featured_pit = candidates[0]
    pit_path = bfs_path(cells, right_blocked, down_blocked, START_POS, featured_pit)
    if pit_path is None:
        raise ValueError("Could not find path to featured pit.")

    blocked_for_others = {featured_pit}

    # 2) Featured confusion, avoiding pit
    featured_confusion = None
    confusion_path = None
    for p in candidates[1:]:
        if p in blocked_for_others:
            continue
        test_path = bfs_path_avoiding(
            cells, right_blocked, down_blocked, START_POS, p, blocked_for_others
        )
        if test_path is not None:
            featured_confusion = p
            confusion_path = test_path
            break

    if featured_confusion is None or confusion_path is None:
        raise ValueError("Could not find featured confusion pad avoiding the pit.")

    blocked_for_others.add(featured_confusion)

    # 3) Featured teleport source, avoiding pit + confusion
    featured_teleport_source = None
    teleport_path = None
    for p in candidates[1:]:
        if p in blocked_for_others:
            continue
        test_path = bfs_path_avoiding(
            cells, right_blocked, down_blocked, START_POS, p, blocked_for_others
        )
        if test_path is not None and len(test_path) >= 3:
            featured_teleport_source = p
            teleport_path = test_path
            break

    if featured_teleport_source is None or teleport_path is None:
        raise ValueError("Could not find featured teleport source avoiding other featured hazards.")

    blocked_for_dest = blocked_for_others | {featured_teleport_source}

    # 4) Far teleport destination
    far_candidates = sorted(
        [
            p for p in dist
            if p not in safe_path_set
            and p not in blocked_for_dest
            and p != START_POS
            and p != GOAL_POS
        ],
        key=lambda p: (-dist[p], p[0], p[1]),
    )

    if not far_candidates:
        raise ValueError("No teleport destination candidate found.")

    featured_teleport_dest = far_candidates[0]

    return {
        "pit": featured_pit,
        "confusion": featured_confusion,
        "teleport_source": featured_teleport_source,
        "teleport_dest": featured_teleport_dest,
        "pit_path": pit_path,
        "confusion_path": confusion_path,
        "teleport_path": teleport_path,
    }


def generate_hazard_layout(
    cells: np.ndarray,
    right_blocked: np.ndarray,
    down_blocked: np.ndarray,
) -> dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    validate_open_cell(cells, START_POS, "START_POS")
    validate_open_cell(cells, GOAL_POS, "GOAL_POS")

    safe_path = bfs_path(cells, right_blocked, down_blocked, START_POS, GOAL_POS)
    if safe_path is None:
        raise ValueError("No path exists from start to goal.")

    featured = pick_feature_targets(cells, right_blocked, down_blocked, safe_path)

    open_cells = {tuple(p) for p in np.argwhere(cells == 0)}
    navigable_count = len(open_cells)

    total_hazards = round(navigable_count * HAZARD_DENSITY)
    pit_total = round(total_hazards * PIT_RATIO)
    confusion_total = round(total_hazards * CONFUSION_RATIO)
    teleport_total = max(1, total_hazards - pit_total - confusion_total)

    death_pits: Set[Position] = {featured["pit"]}
    confusion_pads: Set[Position] = {featured["confusion"]}
    teleport_map: Dict[Position, Position] = {featured["teleport_source"]: featured["teleport_dest"]}

    reserved = set(safe_path)
    for path_key, target_key in (
        ("pit_path", "pit"),
        ("confusion_path", "confusion"),
        ("teleport_path", "teleport_source"),
    ):
        path = featured[path_key]
        target = featured[target_key]
        if path:
            reserved.update(path[:-1])
            reserved.add(target)

    reserved.update({START_POS, GOAL_POS, featured["teleport_dest"]})

    candidates = sorted(
        [
            p
            for p in open_cells
            if p not in reserved
            and p not in death_pits
            and p not in confusion_pads
            and p not in teleport_map
            and p not in set(teleport_map.values())
        ]
    )

    rng = np.random.default_rng(HAZARD_SEED)
    rng.shuffle(candidates)

    pit_needed = max(0, pit_total - len(death_pits))
    conf_needed = max(0, confusion_total - len(confusion_pads))
    tele_needed = max(0, teleport_total - len(teleport_map))

    idx = 0
    for _ in range(pit_needed):
        if idx >= len(candidates):
            break
        death_pits.add(candidates[idx])
        idx += 1

    for _ in range(conf_needed):
        while idx < len(candidates) and candidates[idx] in death_pits:
            idx += 1
        if idx >= len(candidates):
            break
        confusion_pads.add(candidates[idx])
        idx += 1

    remaining_sources = []
    while idx < len(candidates):
        if candidates[idx] not in death_pits and candidates[idx] not in confusion_pads:
            remaining_sources.append(candidates[idx])
        idx += 1

    rng.shuffle(remaining_sources)

    remaining_dests = [
        p
        for p in sorted(open_cells)
        if p not in death_pits
        and p not in confusion_pads
        and p not in teleport_map
        and p not in set(teleport_map.values())
        and p not in reserved
    ]
    rng.shuffle(remaining_dests)

    for src, dst in zip(remaining_sources[:tele_needed], remaining_dests[:tele_needed]):
        if src != dst:
            teleport_map[src] = dst

    actual_total = len(death_pits) + len(confusion_pads) + len(teleport_map)
    actual_density = actual_total / navigable_count

    _CACHE = {
        "safe_path": safe_path,
        "death_pits": death_pits,
        "confusion_pads": confusion_pads,
        "teleport_map": teleport_map,
        "summary": {
            "navigable_cells": navigable_count,
            "death_pits": len(death_pits),
            "confusion_pads": len(confusion_pads),
            "teleport_sources": len(teleport_map),
            "total_hazards": actual_total,
            "hazard_density": actual_density,
            "safe_path_length": len(safe_path),
        },
        "featured": featured,
    }
    return _CACHE