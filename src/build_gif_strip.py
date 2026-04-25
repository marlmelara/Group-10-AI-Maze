"""Extract a 4-panel still strip from each demo GIF for the LaTeX report.

PDFs cannot reliably play animated GIFs across viewers, so we paste four
evenly-spaced frames horizontally into a single PNG that gives the
reader a static visual narrative of the agent's progression.

Usage:  python3 src/build_gif_strip.py
Outputs (one strip per source GIF found in figures/):
  figures/alpha_demo_exploration_strip.png
  figures/alpha_demo_converged_strip.png
  figures/beta_demo_exploration_strip.png
  figures/beta_demo_converged_strip.png
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

SOURCES = [
    "alpha_demo_exploration.gif",
    "alpha_demo_converged.gif",
    "beta_demo_exploration.gif",
    "beta_demo_converged.gif",
]

PANELS = 4
SEP_PX = 6
CAPTION_H = 28


def pick_indices(n_frames: int, panels: int = PANELS) -> list[int]:
    if n_frames <= panels:
        return list(range(n_frames))
    return [round(i * (n_frames - 1) / (panels - 1)) for i in range(panels)]


def build_strip(gif_path: Path, out_path: Path) -> None:
    if not gif_path.exists():
        print(f"  skip (missing): {gif_path.name}")
        return

    with Image.open(gif_path) as im:
        n = getattr(im, "n_frames", 1)
        idxs = pick_indices(n, PANELS)
        panels: list[Image.Image] = []
        for idx in idxs:
            im.seek(idx)
            panels.append(im.convert("RGB").copy())

    panel_w, panel_h = panels[0].size
    strip_w = panel_w * len(panels) + SEP_PX * (len(panels) - 1)
    strip_h = panel_h + CAPTION_H
    strip = Image.new("RGB", (strip_w, strip_h), "white")

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except (OSError, IOError):
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(strip)
    cursor_x = 0
    for i, panel in enumerate(panels):
        strip.paste(panel, (cursor_x, CAPTION_H))
        frac = idxs[i] / max(1, n - 1)
        if i == 0:
            label = "Start"
        elif i == len(panels) - 1:
            label = "End / Goal"
        else:
            label = f"{int(frac * 100)}%"
        try:
            tw = draw.textlength(label, font=font)
        except AttributeError:
            tw = font.getsize(label)[0]
        draw.text((cursor_x + (panel_w - tw) // 2, 4), label,
                  fill="black", font=font)
        cursor_x += panel_w + SEP_PX

    strip.save(out_path)
    print(f"  wrote: {out_path.name}  ({strip_w}x{strip_h}, "
          f"{len(panels)} panels from {n} frames)")


def main() -> None:
    print(f"Building 4-panel strips from {len(SOURCES)} GIFs in {FIG}")
    for src in SOURCES:
        gif = FIG / src
        out = FIG / src.replace(".gif", "_strip.png")
        build_strip(gif, out)


if __name__ == "__main__":
    main()
