#!/usr/bin/env python3
"""
make_paint_key.py — Generate a painting reference card (paint_key.svg).

Shows the engraving pattern for each Wang colour alongside the colour name
and a paint-colour swatch.  Print this and keep it next to your paints when
hand-finishing laser-cut tiles.

Usage:
    python make_paint_key.py [output.svg] [--hatch-spacing MM]
"""

import sys
import math
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
from wang_to_svg_curved import (
    fill_elements, COLOR_NAMES, COLOR_PATTERNS, HATCH_SPACING,
)

# ── Paint colour per Wang colour index ───────────────────────────────────────
PAINT_COLORS = [
    "#1a1410",  # 0 Ink   — near-black
    "#c0392b",  # 1 Red
    "#1a9090",  # 2 Teal
    "#c89b2a",  # 3 Gold
    "#5b7fa6",  # 4 Slate
    "#7b3f6e",  # 5 Plum
]

PATTERN_DESCS = [
    "blank — no marking",
    "vertical lines",
    "diagonal lines  /",
    "crosshatch  + ×",
    "concentric circles",
    "concentric triangles",
]

# ── Layout (mm) ───────────────────────────────────────────────────────────────
S          = 36.0   # tile sample size
CARD_PAD   = 7.0    # padding inside card
SWATCH_R   = 4.5    # paint swatch circle radius
LABEL_H    = 16.0   # height reserved below the tile sample
CARD_W     = S + CARD_PAD * 2
CARD_H     = S + CARD_PAD * 2 + LABEL_H
COLS       = 3
N_COLORS   = len(COLOR_NAMES)
ROWS       = math.ceil(N_COLORS / COLS)
OUTER_PAD  = 10.0

# Header strip
HEADER_H   = 12.0

DOC_W = OUTER_PAD * 2 + COLS * CARD_W
DOC_H = OUTER_PAD * 2 + HEADER_H + ROWS * CARD_H


def make_key_svg(hatch_spacing: float = HATCH_SPACING) -> str:
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg"',
        f'     width="{DOC_W:.3f}mm" height="{DOC_H:.3f}mm"',
        f'     viewBox="0 0 {DOC_W:.4f} {DOC_H:.4f}">',
        f'  <title>Wang Tile — Paint Colour Key</title>',
        '  <style>',
        '    .bg    { fill: #f5f0e8; }',
        '    .card  { fill: #ede7d9; stroke: #c8bfb0; stroke-width: 0.4; }',
        '    .tbg   { fill: #ffffff; stroke: #1a1410; stroke-width: 0.3; }',
        '    .tdiv  { stroke: #d0c8bc; stroke-width: 0.25; }',
        '    .hdr   { font: bold 5.5px/1 sans-serif; fill: #1a1410; letter-spacing: 0.05em; }',
        '    .name  { font: bold 4.8px/1 sans-serif; fill: #1a1410; }',
        '    .desc  { font: 3.4px/1 sans-serif; fill: #6b5e4e; }',
        '  </style>',
        # Sheet background
        f'  <rect width="{DOC_W:.4f}" height="{DOC_H:.4f}" class="bg"/>',
        # Header
        f'  <text x="{DOC_W / 2:.4f}" y="{OUTER_PAD + 8:.4f}" class="hdr"'
        f' text-anchor="middle">WANG TILE — PAINT COLOUR KEY</text>',
    ]

    grid_top = OUTER_PAD + HEADER_H

    for ci in range(N_COLORS):
        col = ci % COLS
        row = ci // COLS
        cx = OUTER_PAD + col * CARD_W
        cy = grid_top + row * CARD_H

        # Card background
        lines.append(
            f'  <rect x="{cx:.4f}" y="{cy:.4f}"'
            f' width="{CARD_W:.4f}" height="{CARD_H:.4f}" class="card" rx="2"/>'
        )

        # Tile sample origin
        tx = cx + CARD_PAD
        ty = cy + CARD_PAD

        # White tile background
        lines.append(
            f'  <rect x="{tx:.4f}" y="{ty:.4f}"'
            f' width="{S:.4f}" height="{S:.4f}" class="tbg"/>'
        )

        # Faint triangle dividers (centre cross)
        half = S / 2
        for x1, y1, x2, y2 in [
            (0, 0, half, half), (S, 0, half, half),
            (S, S, half, half), (0, S, half, half),
        ]:
            lines.append(
                f'  <line x1="{tx+x1:.4f}" y1="{ty+y1:.4f}"'
                f' x2="{tx+x2:.4f}" y2="{ty+y2:.4f}" class="tdiv"/>'
            )

        # Engraving pattern for all 4 triangles
        all_defs, blue_elems = [], []
        for edge_idx in range(4):
            clip_id = f"kc{ci}e{edge_idx}"
            defs, elems = fill_elements(edge_idx, ci, S, hatch_spacing, clip_id)
            all_defs.extend(defs)
            blue_elems.extend(elems)

        if all_defs or blue_elems:
            lines.append(f'  <g transform="translate({tx:.4f},{ty:.4f})">')
            if all_defs:
                lines.append('    <defs>')
                for d in all_defs:
                    lines.append(f'      {d}')
                lines.append('    </defs>')
            for e in blue_elems:
                lines.append(f'    {e}')
            lines.append('  </g>')

        # ── Label strip ───────────────────────────────────────────────────────
        label_top = ty + S + 3.5
        swatch_cx = tx + SWATCH_R + 1.5
        swatch_cy = label_top + SWATCH_R

        # Paint swatch
        lines.append(
            f'  <circle cx="{swatch_cx:.4f}" cy="{swatch_cy:.4f}" r="{SWATCH_R:.4f}"'
            f' fill="{PAINT_COLORS[ci]}" stroke="#1a1410" stroke-width="0.3"/>'
        )

        # Colour name + pattern description
        text_x = swatch_cx + SWATCH_R + 3.0
        lines.append(
            f'  <text x="{text_x:.4f}" y="{label_top + 4.5:.4f}" class="name">'
            f'{COLOR_NAMES[ci]}</text>'
        )
        lines.append(
            f'  <text x="{text_x:.4f}" y="{label_top + 9.5:.4f}" class="desc">'
            f'{PATTERN_DESCS[ci]}</text>'
        )

    lines.append('</svg>')
    return '\n'.join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", nargs="?", default="paint_key.svg",
                    help="Output SVG path (default: paint_key.svg)")
    ap.add_argument("--hatch-spacing", type=float, default=HATCH_SPACING, metavar="MM",
                    help=f"Hatch line spacing in mm (default {HATCH_SPACING})")
    args = ap.parse_args()

    out = Path(args.output)
    svg = make_key_svg(hatch_spacing=args.hatch_spacing)
    out.write_text(svg)
    print(f"Paint key  →  {out}  ({DOC_W:.1f} × {DOC_H:.1f} mm)")
    print(f"  {N_COLORS} colours, {COLS}×{ROWS} grid, hatch spacing {args.hatch_spacing} mm")
