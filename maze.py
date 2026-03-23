from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "maze.png"
GRID_SIZE = 64
BIN_THRESHOLD = 140
DARK_CROP_THRESHOLD = 220

# Expected raster structure after crop:
# total size = 2 + GRID_SIZE * PITCH
# where PITCH = WALL_THICKNESS + CELL_INTERIOR
WALL_THICKNESS = 2
PITCH = 16
CELL_INTERIOR = 14


def crop_maze(arr: np.ndarray) -> np.ndarray:
    dark = arr < DARK_CROP_THRESHOLD
    ys, xs = np.where(dark)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    margin = 0
    y0 = max(0, y0 - margin)
    y1 = min(arr.shape[0] - 1, y1 + margin)
    x0 = max(0, x0 - margin)
    x1 = min(arr.shape[1] - 1, x1 + margin)

    return arr[y0:y1 + 1, x0:x1 + 1]


def pixel_walls_from_image(cropped: np.ndarray) -> np.ndarray:
    return (cropped < BIN_THRESHOLD).astype(np.uint8)


def validate_raster_shape(pixel_walls: np.ndarray) -> None:
    h, w = pixel_walls.shape
    expected = WALL_THICKNESS + GRID_SIZE * PITCH
    if h != expected or w != expected:
        raise ValueError(
            f"Unexpected cropped maze size {pixel_walls.shape}. "
            f"Expected {(expected, expected)} for a 64x64 maze with 2-pixel walls and 16-pixel pitch."
        )


def cell_bounds(index: int) -> tuple[int, int]:
    """
    Inclusive interior bounds for logical cell index.
    Example for index 0: rows/cols 2..15
    """
    start = WALL_THICKNESS + index * PITCH
    end = start + CELL_INTERIOR - 1
    return start, end


def cell_center(index: int) -> int:
    start, end = cell_bounds(index)
    return (start + end) // 2


def detect_top_bottom_openings(pixel_walls: np.ndarray) -> tuple[int, int]:
    """
    Detect which logical column contains the top entrance and bottom exit.
    We inspect only the outer 2-pixel border band.
    """
    top_scores = []
    bottom_scores = []

    for c in range(GRID_SIZE):
        x0, x1 = cell_bounds(c)

        top_band = pixel_walls[0:WALL_THICKNESS, x0:x1 + 1]
        bottom_band = pixel_walls[-WALL_THICKNESS:, x0:x1 + 1]

        # Lower mean = more open
        top_scores.append(float(top_band.mean()))
        bottom_scores.append(float(bottom_band.mean()))

    start_col = int(np.argmin(top_scores))
    goal_col = int(np.argmin(bottom_scores))
    return start_col, goal_col


def build_cell_centers() -> tuple[np.ndarray, np.ndarray]:
    y_centers = np.array([cell_center(r) for r in range(GRID_SIZE)])
    x_centers = np.array([cell_center(c) for c in range(GRID_SIZE)])
    return y_centers, x_centers


def wall_between_horizontal(pixel_walls: np.ndarray, r: int, c: int) -> bool:
    """
    Check wall between cell (r,c) and (r,c+1).
    We inspect the exact 2-pixel vertical wall band between the cells.
    """
    y0, y1 = cell_bounds(r)
    boundary_x0 = WALL_THICKNESS + (c + 1) * PITCH - WALL_THICKNESS
    boundary_x1 = boundary_x0 + WALL_THICKNESS - 1

    band = pixel_walls[y0:y1 + 1, boundary_x0:boundary_x1 + 1]
    return float(band.mean()) > 0.50


def wall_between_vertical(pixel_walls: np.ndarray, r: int, c: int) -> bool:
    """
    Check wall between cell (r,c) and (r+1,c).
    We inspect the exact 2-pixel horizontal wall band between the cells.
    """
    x0, x1 = cell_bounds(c)
    boundary_y0 = WALL_THICKNESS + (r + 1) * PITCH - WALL_THICKNESS
    boundary_y1 = boundary_y0 + WALL_THICKNESS - 1

    band = pixel_walls[boundary_y0:boundary_y1 + 1, x0:x1 + 1]
    return float(band.mean()) > 0.50


def build_adjacency_walls(pixel_walls: np.ndarray):
    """
    right_blocked[r, c] == 1 means moving from (r,c) to (r,c+1) is blocked
    down_blocked[r, c] == 1 means moving from (r,c) to (r+1,c) is blocked
    """
    right_blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    down_blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE - 1):
            right_blocked[r, c] = 1 if wall_between_horizontal(pixel_walls, r, c) else 0

    for r in range(GRID_SIZE - 1):
        for c in range(GRID_SIZE):
            down_blocked[r, c] = 1 if wall_between_vertical(pixel_walls, r, c) else 0

    return right_blocked, down_blocked


def save_overlay_preview(cropped_gray: np.ndarray, y_centers: np.ndarray, x_centers: np.ndarray) -> None:
    """
    Draw the 64x64 logical cell centers and guides on top of the cropped maze.
    """
    rgb = np.stack([cropped_gray] * 3, axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    radius = 2

    for y in y_centers:
        draw.line([(0, int(y)), (img.width - 1, int(y))], fill=(180, 180, 255), width=1)

    for x in x_centers:
        draw.line([(int(x), 0), (int(x), img.height - 1)], fill=(180, 180, 255), width=1)

    for y in y_centers:
        for x in x_centers:
            x0 = int(x) - radius
            y0 = int(y) - radius
            x1 = int(x) + radius
            y1 = int(y) + radius
            draw.ellipse((x0, y0, x1, y1), fill=(255, 0, 0))

    img.save("maze_overlay.png")

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title("64x64 Cell-Center Overlay")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def save_logical_preview(right_blocked: np.ndarray, down_blocked: np.ndarray, start_col: int, goal_col: int) -> None:
    """
    Draw the logical 64x64 maze using blocked edges.
    This is the correct visual for the extracted representation.
    """
    n = GRID_SIZE
    scale = 12
    size = n * scale + 1
    canvas = np.ones((size, size), dtype=np.uint8) * 255

    # Outer border
    canvas[0, :] = 0
    canvas[-1, :] = 0
    canvas[:, 0] = 0
    canvas[:, -1] = 0

    # Top entrance
    top_x0 = start_col * scale + 1
    top_x1 = (start_col + 1) * scale
    canvas[0, top_x0:top_x1] = 255

    # Bottom entrance
    bot_x0 = goal_col * scale + 1
    bot_x1 = (goal_col + 1) * scale
    canvas[-1, bot_x0:bot_x1] = 255

    # Vertical walls between cells
    for r in range(n):
        y0 = r * scale
        y1 = (r + 1) * scale
        for c in range(n - 1):
            if right_blocked[r, c]:
                x = (c + 1) * scale
                canvas[y0:y1 + 1, x] = 0

    # Horizontal walls between cells
    for r in range(n - 1):
        y = (r + 1) * scale
        for c in range(n):
            if down_blocked[r, c]:
                x0 = c * scale
                x1 = (c + 1) * scale
                canvas[y, x0:x1 + 1] = 0

    Image.fromarray(canvas).save("maze_logical_preview.png")

    plt.figure(figsize=(8, 8))
    plt.imshow(canvas, cmap="gray")
    plt.title("64x64 Logical Wall Preview")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def main():
    img = Image.open(IMAGE_PATH).convert("L")
    arr = np.array(img)

    cropped = crop_maze(arr)
    pixel_walls = pixel_walls_from_image(cropped)

    validate_raster_shape(pixel_walls)

    y_centers, x_centers = build_cell_centers()
    start_col, goal_col = detect_top_bottom_openings(pixel_walls)
    right_blocked, down_blocked = build_adjacency_walls(pixel_walls)

    # In this representation, all logical cells are navigable positions.
    cells = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

    np.save("maze_cells.npy", cells)
    np.save("maze_walls_right.npy", right_blocked)
    np.save("maze_walls_down.npy", down_blocked)

    print("Original image size:", img.size)
    print("Cropped pixel maze size:", cropped.shape)
    print("Logical maze size:", cells.shape)
    print("Top opening logical column:", start_col)
    print("Bottom opening logical column:", goal_col)
    print("Saved: maze_cells.npy, maze_walls_right.npy, maze_walls_down.npy, maze_overlay.png, maze_logical_preview.png")

    save_overlay_preview(cropped, y_centers, x_centers)
    save_logical_preview(right_blocked, down_blocked, start_col, goal_col)


if __name__ == "__main__":
    main()