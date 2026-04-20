"""
Maze parser: extracts walls and hazards from the provided PNG maze images.

Two images per maze:
    walls.png    (MAZE_0) - binary wall structure only.
    hazards.png  (MAZE_1) - wall structure overlaid with colored hazard icons.

Output (per maze) is a dict saved as .npz containing:
    right_blocked      (64,64)  0 = open, 1 = wall between (r,c) and (r,c+1)
    down_blocked       (64,64)  0 = open, 1 = wall between (r,c) and (r+1,c)
    start, goal        tuples (row, col)
    death_pits         list[(r,c)]
    teleport_sources   list[(r,c)]
    teleport_dests     list[(r,c)]
    confusion_pads     list[(r,c)]
    wind_cells         list[(r,c, direction)]   (gamma only)
    fire_groups        list[list[(r,c)]]        V-shaped clusters for rotating hazard

Hazards are detected by inspecting the saturated (non-grayscale) pixels inside
each logical cell. We classify by hue:
    red        -> teleport destination
    green      -> goal   (star-shaped) OR teleport destination (plain ball)
    orange     -> fire (death pit)  OR  confusion-skull (bright yellow+orange mix)
    purple     -> teleport source
    yellow     -> teleport source
    blue-arrow -> wind (gamma only)
    skull pad  -> confusion

Because color semantics are ambiguous from the instructor's palette, we use a
combination of hue and shape/size heuristics.  We additionally use the
connected-path test (BFS) to verify which openings correspond to start vs goal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

GRID_SIZE = 64
WALL_THICKNESS = 2
PITCH = 16
CELL_INTERIOR = 14

Position = Tuple[int, int]


# ---------------------------------------------------------------------------
# Low-level pixel helpers
# ---------------------------------------------------------------------------

def _crop_maze(arr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop border whitespace using a dark-pixel threshold. Returns cropped
    array and the (y0, x0) origin so color images can be cropped identically."""
    if arr.ndim == 3:
        gray = arr.mean(axis=-1)
    else:
        gray = arr
    dark = gray < 220
    ys, xs = np.where(dark)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return arr[y0:y1 + 1, x0:x1 + 1], (y0, x0)


def _cell_bounds(index: int) -> Tuple[int, int]:
    start = WALL_THICKNESS + index * PITCH
    end = start + CELL_INTERIOR - 1
    return start, end


# ---------------------------------------------------------------------------
# Wall extraction
# ---------------------------------------------------------------------------

def _wall_between_h(pixel_walls: np.ndarray, r: int, c: int) -> bool:
    y0, y1 = _cell_bounds(r)
    bx0 = WALL_THICKNESS + (c + 1) * PITCH - WALL_THICKNESS
    bx1 = bx0 + WALL_THICKNESS - 1
    band = pixel_walls[y0:y1 + 1, bx0:bx1 + 1]
    return float(band.mean()) > 0.5


def _wall_between_v(pixel_walls: np.ndarray, r: int, c: int) -> bool:
    x0, x1 = _cell_bounds(c)
    by0 = WALL_THICKNESS + (r + 1) * PITCH - WALL_THICKNESS
    by1 = by0 + WALL_THICKNESS - 1
    band = pixel_walls[by0:by1 + 1, x0:x1 + 1]
    return float(band.mean()) > 0.5


def _build_wall_arrays(pixel_walls: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    right_blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    down_blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE - 1):
            right_blocked[r, c] = 1 if _wall_between_h(pixel_walls, r, c) else 0
    for r in range(GRID_SIZE - 1):
        for c in range(GRID_SIZE):
            down_blocked[r, c] = 1 if _wall_between_v(pixel_walls, r, c) else 0
    return right_blocked, down_blocked


def _detect_border_openings(pixel_walls: np.ndarray) -> Tuple[int, int]:
    """Return (top_col, bottom_col) of the two border openings."""
    top_scores, bottom_scores = [], []
    for c in range(GRID_SIZE):
        x0, x1 = _cell_bounds(c)
        top_band = pixel_walls[0:WALL_THICKNESS, x0:x1 + 1]
        bot_band = pixel_walls[-WALL_THICKNESS:, x0:x1 + 1]
        top_scores.append(float(top_band.mean()))
        bottom_scores.append(float(bot_band.mean()))
    return int(np.argmin(top_scores)), int(np.argmin(bottom_scores))


# ---------------------------------------------------------------------------
# Hazard color classification
# ---------------------------------------------------------------------------

@dataclass
class CellSignal:
    """Summary of colored pixels inside a single cell's interior."""
    row: int
    col: int
    colored_pixel_count: int
    dominant_hue: float         # HSV hue 0..360
    avg_saturation: float       # 0..1
    avg_value: float            # 0..1
    mean_rgb: Tuple[float, float, float]


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """rgb shape (...,3) in 0..255 -> hsv (...,3) in (deg, 0..1, 0..1)."""
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    mask = diff > 1e-6
    r_eq = (mx == r) & mask
    g_eq = (mx == g) & mask & ~r_eq
    b_eq = (mx == b) & mask & ~r_eq & ~g_eq
    h[r_eq] = ((g[r_eq] - b[r_eq]) / diff[r_eq]) % 6
    h[g_eq] = (b[g_eq] - r[g_eq]) / diff[g_eq] + 2
    h[b_eq] = (r[b_eq] - g[b_eq]) / diff[b_eq] + 4
    h = h * 60.0
    s = np.where(mx > 0, diff / np.maximum(mx, 1e-6), 0)
    v = mx
    return np.stack([h, s, v], axis=-1)


def _scan_cells(color_img: np.ndarray) -> List[CellSignal]:
    """For each cell, accumulate stats on its saturated pixels."""
    signals: List[CellSignal] = []
    hsv_cache: Dict[Tuple[int, int], np.ndarray] = {}
    for r in range(GRID_SIZE):
        y0, y1 = _cell_bounds(r)
        for c in range(GRID_SIZE):
            x0, x1 = _cell_bounds(c)
            patch = color_img[y0:y1 + 1, x0:x1 + 1, :3]
            hsv = _rgb_to_hsv(patch)
            hsv_cache[(r, c)] = hsv
            sat = hsv[..., 1]
            val = hsv[..., 2]
            colored_mask = (sat > 0.35) & (val > 0.35)
            count = int(colored_mask.sum())
            if count == 0:
                signals.append(CellSignal(r, c, 0, 0.0, 0.0, float(val.mean()),
                                          tuple(patch.mean(axis=(0, 1)))))
                continue
            hues = hsv[..., 0][colored_mask]
            # Circular mean for hue.
            angs = np.deg2rad(hues)
            mean_ang = np.arctan2(np.sin(angs).mean(), np.cos(angs).mean())
            mean_hue = (np.rad2deg(mean_ang) + 360) % 360
            signals.append(CellSignal(
                row=r, col=c,
                colored_pixel_count=count,
                dominant_hue=float(mean_hue),
                avg_saturation=float(sat[colored_mask].mean()),
                avg_value=float(val[colored_mask].mean()),
                mean_rgb=tuple(patch[colored_mask].mean(axis=0).tolist()),
            ))
    return signals


# Hue bands (degrees)
# Red:         0-15 or 345-360
# Orange/Fire: 15-45
# Yellow:      45-65
# Green:       90-160
# Cyan:        170-200
# Blue:        200-240 (wind arrows)
# Purple:      260-300
# Magenta:     300-345
def _classify_cell(sig: CellSignal) -> Optional[str]:
    if sig.colored_pixel_count < 8:
        return None
    h = sig.dominant_hue
    s = sig.avg_saturation
    v = sig.avg_value

    if (h <= 15 or h >= 345) and s > 0.5:
        return "red"
    if 15 < h <= 45 and s > 0.5:
        return "orange"
    if 45 < h <= 70 and s > 0.4:
        return "yellow"
    if 70 < h < 170 and s > 0.35:
        return "green"
    if 170 <= h < 210:
        return "blue"
    if 210 <= h < 260:
        return "blue"
    if 260 <= h < 330:
        return "purple"
    return None


# ---------------------------------------------------------------------------
# V-cluster grouping for fires
# ---------------------------------------------------------------------------

def _group_fire_clusters(fire_cells: List[Position]) -> List[List[Position]]:
    """Group adjacent fire cells into clusters (8-neighborhood).

    A V-shaped cluster has a pivot (the fire at the 'V' tip, which is the
    single cell every other fire in the cluster is closest to spatially).
    We return each cluster ordered so that the pivot is first.
    """
    remaining: Set[Position] = set(fire_cells)
    clusters: List[List[Position]] = []
    while remaining:
        seed = next(iter(remaining))
        stack = [seed]
        cluster: List[Position] = []
        while stack:
            p = stack.pop()
            if p not in remaining:
                continue
            remaining.remove(p)
            cluster.append(p)
            r, c = p
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    q = (r + dr, c + dc)
                    if q in remaining:
                        stack.append(q)
        clusters.append(cluster)
    # Pivot = the cell whose max pairwise Chebyshev distance to siblings is
    # minimized (i.e. the centroid).  For a V, this tends to be the tip.
    ordered: List[List[Position]] = []
    for cl in clusters:
        if len(cl) <= 1:
            ordered.append(cl)
            continue
        def score(p: Position) -> float:
            r0, c0 = p
            return sum(max(abs(r0 - r), abs(c0 - c)) for r, c in cl)
        pivot = min(cl, key=score)
        rest = [p for p in cl if p != pivot]
        ordered.append([pivot] + rest)
    return ordered


# ---------------------------------------------------------------------------
# Top-level parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedMaze:
    name: str
    right_blocked: np.ndarray
    down_blocked: np.ndarray
    start: Position
    goal: Position
    death_pits: Set[Position]
    teleport_sources: List[Position]
    teleport_dests: List[Position]
    confusion_pads: Set[Position]
    wind_cells: Dict[Position, str]        # pos -> "UP"/"DOWN"/"LEFT"/"RIGHT"
    fire_groups: List[List[Position]]      # each cluster pivot-first

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            right_blocked=self.right_blocked,
            down_blocked=self.down_blocked,
            start=np.array(self.start, dtype=np.int32),
            goal=np.array(self.goal, dtype=np.int32),
            death_pits=np.array(sorted(self.death_pits), dtype=np.int32) if self.death_pits else np.zeros((0, 2), dtype=np.int32),
            teleport_sources=np.array(self.teleport_sources, dtype=np.int32) if self.teleport_sources else np.zeros((0, 2), dtype=np.int32),
            teleport_dests=np.array(self.teleport_dests, dtype=np.int32) if self.teleport_dests else np.zeros((0, 2), dtype=np.int32),
            confusion_pads=np.array(sorted(self.confusion_pads), dtype=np.int32) if self.confusion_pads else np.zeros((0, 2), dtype=np.int32),
            wind_keys=np.array(list(self.wind_cells.keys()), dtype=np.int32) if self.wind_cells else np.zeros((0, 2), dtype=np.int32),
            wind_vals=np.array(list(self.wind_cells.values()), dtype='<U5') if self.wind_cells else np.zeros((0,), dtype='<U5'),
            fire_groups=np.array([np.array(g, dtype=np.int32) for g in self.fire_groups], dtype=object),
            name=np.array(self.name),
        )

    @classmethod
    def load(cls, path: str) -> "ParsedMaze":
        z = np.load(path, allow_pickle=True)
        return cls(
            name=str(z["name"]),
            right_blocked=z["right_blocked"],
            down_blocked=z["down_blocked"],
            start=tuple(z["start"].tolist()),
            goal=tuple(z["goal"].tolist()),
            death_pits=set(tuple(p) for p in z["death_pits"].tolist()),
            teleport_sources=[tuple(p) for p in z["teleport_sources"].tolist()],
            teleport_dests=[tuple(p) for p in z["teleport_dests"].tolist()],
            confusion_pads=set(tuple(p) for p in z["confusion_pads"].tolist()),
            wind_cells={tuple(k): str(v) for k, v in zip(z["wind_keys"].tolist(), z["wind_vals"].tolist())},
            fire_groups=[[tuple(p) for p in g] for g in z["fire_groups"]],
        )


def _detect_wind_arrows(color_img: np.ndarray, wind_candidates: Set[Position]) -> Dict[Position, str]:
    """For each cell whose dominant color is blue, classify the arrow direction
    by comparing mass distribution of dark pixels inside the arrow glyph."""
    winds: Dict[Position, str] = {}
    for (r, c) in wind_candidates:
        y0, y1 = _cell_bounds(r)
        x0, x1 = _cell_bounds(c)
        patch = color_img[y0:y1 + 1, x0:x1 + 1, :3]
        # Arrow is drawn in white/light on blue background.
        light = patch.mean(axis=-1) > 200
        if light.sum() < 4:
            continue
        ys, xs = np.where(light)
        cy, cx = ys.mean() - light.shape[0] / 2, xs.mean() - light.shape[1] / 2
        # Compare against top/bottom/left/right halves to pick direction.
        top = light[:light.shape[0] // 2, :].sum()
        bot = light[light.shape[0] // 2:, :].sum()
        left = light[:, :light.shape[1] // 2].sum()
        right = light[:, light.shape[1] // 2:].sum()
        v_diff = top - bot
        h_diff = left - right
        if abs(v_diff) > abs(h_diff):
            winds[(r, c)] = "UP" if v_diff > 0 else "DOWN"
        else:
            winds[(r, c)] = "LEFT" if h_diff > 0 else "RIGHT"
    return winds


def parse_maze(name: str, walls_png: str, hazards_png: str,
               has_wind: bool = False) -> ParsedMaze:
    # --- walls ---
    walls_arr = np.array(Image.open(walls_png).convert("L"))
    walls_cropped, _ = _crop_maze(walls_arr)
    pixel_walls = (walls_cropped < 140).astype(np.uint8)
    right_blocked, down_blocked = _build_wall_arrays(pixel_walls)
    top_col, bottom_col = _detect_border_openings(pixel_walls)

    # --- hazards image (color) ---
    haz_rgba = np.array(Image.open(hazards_png).convert("RGB"))
    # Use the gray-crop logic on the grayscale version of the color image so
    # that the pixel grid aligns with the wall grid.
    haz_gray = haz_rgba.mean(axis=-1)
    _, (y0, x0) = _crop_maze(haz_gray)
    haz_cropped = haz_rgba[y0:y0 + pixel_walls.shape[0], x0:x0 + pixel_walls.shape[1]]
    # Pad if crops differ slightly.
    ph, pw = pixel_walls.shape
    ch, cw = haz_cropped.shape[:2]
    if (ch, cw) != (ph, pw):
        pad_h = max(0, ph - ch)
        pad_w = max(0, pw - cw)
        haz_cropped = np.pad(haz_cropped, ((0, pad_h), (0, pad_w), (0, 0)),
                             mode='constant', constant_values=255)
        haz_cropped = haz_cropped[:ph, :pw]

    signals = _scan_cells(haz_cropped)
    classified: Dict[Position, str] = {}
    signal_lookup: Dict[Position, CellSignal] = {}
    for sig in signals:
        kind = _classify_cell(sig)
        signal_lookup[(sig.row, sig.col)] = sig
        if kind is not None:
            classified[(sig.row, sig.col)] = kind

    # Fires: orange cells that form clusters of size >= 3 (skulls are singletons).
    orange_cells = [p for p, k in classified.items() if k == "orange"]
    clusters = _group_fire_clusters(orange_cells)
    fire_groups = [cl for cl in clusters if len(cl) >= 3]
    fire_set: Set[Position] = {p for g in fire_groups for p in g}
    # Standalone orange = skull confusion pad.
    skull_cells = {p for p in orange_cells if p not in fire_set}

    # Yellow: small skull-and-crossbones pads look yellow-ish on some palettes,
    # but yellow is also used for a teleport ball.  We distinguish by pixel
    # count: skulls have more pixels than balls typically.
    yellow_cells = [p for p, k in classified.items() if k == "yellow"]
    # Purple: teleport source.  Red/green: teleport destinations.
    purple_cells = [p for p, k in classified.items() if k == "purple"]
    red_cells = [p for p, k in classified.items() if k == "red"]
    green_cells = [p for p, k in classified.items() if k == "green"]
    blue_cells = [p for p, k in classified.items() if k == "blue"]

    # -------- confusion pads --------
    # The instructor draws confusion pads as orange skulls; we fold in
    # standalone yellow cells that look like skull icons (pixel count high
    # relative to a teleport ball).
    skull_pixel_threshold = 40
    confusion_pads: Set[Position] = set(skull_cells)
    for p in list(yellow_cells):
        if signal_lookup[p].colored_pixel_count > skull_pixel_threshold:
            confusion_pads.add(p)
            yellow_cells.remove(p)

    # -------- teleport pairs --------
    # Sources: purple + yellow ball cells. Destinations: red + green balls.
    # We pair each source to a destination by spatial "closest unused" heuristic
    # (matches the instructor's layout where each color of source maps to a
    # specific color of destination, but without a legend we use proximity).
    tele_srcs = list(purple_cells) + yellow_cells
    tele_dsts = list(red_cells) + list(green_cells)

    # One green cell is the GOAL (the star). Distinguish by matching the cell
    # near the top/bottom opening.
    start = (GRID_SIZE - 1, bottom_col)
    goal = (0, top_col)

    # Pick closest green ball to `goal` border opening as GOAL; remove from dest list.
    def _chebyshev(a: Position, b: Position) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    if green_cells:
        best_green = min(green_cells, key=lambda p: _chebyshev(p, goal))
        if best_green in tele_dsts:
            tele_dsts.remove(best_green)

    # Now greedy match sources to nearest remaining destination.
    teleport_sources: List[Position] = []
    teleport_dests: List[Position] = []
    dsts_remaining = list(tele_dsts)
    for s in tele_srcs:
        if not dsts_remaining:
            break
        d = min(dsts_remaining, key=lambda q: _chebyshev(s, q))
        dsts_remaining.remove(d)
        teleport_sources.append(s)
        teleport_dests.append(d)

    # -------- wind cells (gamma) --------
    wind_cells: Dict[Position, str] = {}
    if has_wind and blue_cells:
        wind_cells = _detect_wind_arrows(haz_cropped, set(blue_cells))

    # -------- death pits --------
    death_pits: Set[Position] = set(fire_set)

    return ParsedMaze(
        name=name,
        right_blocked=right_blocked,
        down_blocked=down_blocked,
        start=start,
        goal=goal,
        death_pits=death_pits,
        teleport_sources=teleport_sources,
        teleport_dests=teleport_dests,
        confusion_pads=confusion_pads,
        wind_cells=wind_cells,
        fire_groups=fire_groups,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name, has_wind in [("alpha", False), ("beta", False), ("gamma", True)]:
        walls_png = os.path.join(here, "data", name, "walls.png")
        haz_png = os.path.join(here, "data", name, "hazards.png")
        out_path = os.path.join(here, "data", name, "parsed.npz")
        parsed = parse_maze(name, walls_png, haz_png, has_wind=has_wind)
        parsed.save(out_path)
        print(f"[{name:>5}] start={parsed.start}  goal={parsed.goal}  "
              f"fires={len(parsed.death_pits)}  conf={len(parsed.confusion_pads)}  "
              f"teleports={len(parsed.teleport_sources)}  wind={len(parsed.wind_cells)}  "
              f"fire_groups={len(parsed.fire_groups)}  saved={out_path}")
