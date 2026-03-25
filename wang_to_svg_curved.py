#!/usr/bin/env python3
"""
wang_to_svg.py — Wang Tile Designer JSON → laser-cuttable SVG puzzle pieces.

Reads the JSON produced by wang-tile-designer.html, counts how many times each
unique tile type appears in the grid, and emits a sheet of puzzle pieces that
can be assembled back into the same pattern.

Visual encoding (no color printing needed)
──────────────────────────────────────────
Each tile's four triangular edge segments are filled with a distinct
pattern per Wang-color, using blue hairlines:

  Color 0 (Ink)  : blank (no marking)
  Color 1 (Red)  : vertical blue lines   (90°)
  Color 2 (Teal) : diagonal blue lines   (45°  /)
  Color 3 (Gold) : blue crosshatch       (0° + 90°)
  Color 4 (Slate): concentric blue circles
  Color 5 (Plum) : concentric blue triangles

  Cut outline    : RED (#ff0000) hairline → cut through

Typical laser-software colour mapping
  #000000 black fill  → raster engrave layer
  #0000ff blue stroke → vector score/engrave layer
  #ff0000 red stroke  → vector cut layer

Tab/slot encoding ensures orientation-correct assembly
──────────────────────────────────────────────────────
  • Bottom edges carry TABS, top edges carry matching SLOTS.
  • Right edges carry TABS, left edges carry matching SLOTS.
  • Tab WIDTH encodes edge color — each color has a distinct width.

Usage
─────
  python wang_to_svg.py design.json [output.svg]
               [--tile-size MM] [--tab-height MM]
               [--margin MM] [--tolerance MM] [--hatch-spacing MM]
               [--stock-width IN] [--stock-height IN]

Output is split across as many files as needed to stay within the stock
dimensions.  If only one sheet is required the output path is used as-is;
otherwise files are named <stem>_01.svg, <stem>_02.svg, …
"""

import json
import math
import sys
import argparse
from pathlib import Path
from collections import Counter

# ── Defaults ──────────────────────────────────────────────────────────────────
TILE_SIZE      = 40.0   # mm
TAB_HEIGHT     = 5.0    # mm
TAB_TOLERANCE  = 0.15   # mm added to slots (kerf / fit clearance)
MARGIN         = 6.0    # mm gap between pieces on sheet
HATCH_SPACING  = 3.0    # mm between parallel hatch lines (perpendicular distance)

MM_PER_IN      = 25.4
STOCK_W_IN     = 12.0   # inches — default stock width
STOCK_H_IN     = 12.0   # inches — default stock height

# Tab/slot width per color index (mm) — must be clearly distinct
TAB_WIDTHS = [4.0, 6.5, 9.0, 11.5, 14.0, 16.5]

COLOR_NAMES = ["Ink", "Red", "Teal", "Gold", "Slate", "Plum"]

# Edge indices
TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3

# Which edges carry tabs (True) vs slots (False)
IS_TAB = {TOP: False, RIGHT: True, BOTTOM: True, LEFT: False}

ETCH_WIDTH = 0.1
CUT_WIDTH = 0.0762

# Pattern per color:
#   ('none',)                — no marking
#   ('hatch', (angle, ...))  — parallel lines at given angle(s) in degrees
#   ('circles',)             — concentric circles, clipped to triangle
#   ('triangles',)           — concentric scaled triangles, clipped to triangle
COLOR_PATTERNS = [
    ('none',),            # 0 Ink   — blank
    ('hatch', (90,)),     # 1 Red   — vertical blue lines
    ('hatch', (45,)),     # 2 Teal  — diagonal / blue lines
    ('hatch', (0, 90)),   # 3 Gold  — blue crosshatch
    ('circles',),         # 4 Slate — concentric blue circles
    ('triangles',),       # 5 Plum  — concentric blue triangles
]


# ── Triangle geometry & clipping ──────────────────────────────────────────────

def triangle_vertices(edge_idx, S):
    """Return 3 CW-ordered (x,y) vertices for the edge triangle in [0,S]²."""
    cx, cy = S / 2, S / 2
    return [
        [(0, 0), (S, 0),  (cx, cy)],   # TOP
        [(S, 0), (S, S),  (cx, cy)],   # RIGHT
        [(S, S), (0, S),  (cx, cy)],   # BOTTOM
        [(0, S), (0, 0),  (cx, cy)],   # LEFT
    ][edge_idx]


def clip_line_to_poly(x1, y1, x2, y2, poly):
    """
    Cyrus-Beck parametric clip of segment (x1,y1)-(x2,y2) to a convex polygon
    given in CW order (SVG y-down coordinates).
    Returns (cx1, cy1, cx2, cy2) or None if fully clipped away.
    """
    t0, t1 = 0.0, 1.0
    dx, dy = x2 - x1, y2 - y1
    n = len(poly)

    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        # Inward normal for CW polygon (SVG y-down): rotate edge CCW → (−ey, ex)
        nx, ny = -ey, ex
        denom = nx * dx + ny * dy
        numer = nx * (x1 - ax) + ny * (y1 - ay)

        if abs(denom) < 1e-12:
            if numer < -1e-9:
                return None          # parallel and outside
        elif denom > 0:              # entering
            t0 = max(t0, -numer / denom)
        else:                        # exiting
            t1 = min(t1, -numer / denom)

        if t0 >= t1 - 1e-9:
            return None

    if t0 >= t1 - 1e-9:
        return None

    return (x1 + t0 * dx, y1 + t0 * dy,
            x1 + t1 * dx, y1 + t1 * dy)


def hatch_segments(angle_deg, S, spacing):
    """
    Yield (x1,y1,x2,y2) long line segments at `angle_deg` covering [0,S]².
    Callers clip each segment to their triangle with clip_line_to_poly().

    Perpendicular spacing between lines equals `spacing` mm.
    """
    EPS = 0.001

    if angle_deg == 0:               # horizontal: y = c
        y = 0.0
        while y <= S + EPS:
            yield (-EPS, y, S + EPS, y)
            y += spacing

    elif angle_deg == 90:            # vertical: x = c
        x = 0.0
        while x <= S + EPS:
            yield (x, -EPS, x, S + EPS)
            x += spacing

    elif angle_deg == 45:            # y = x + c,  c = y − x ∈ [−S, S]
        step_c = spacing * math.sqrt(2)
        c = -S
        while c <= S + EPS:
            yield (-EPS, c - EPS, S + EPS, S + c + EPS)
            c += step_c

    elif angle_deg == -45:           # y = −x + c,  c = y + x ∈ [0, 2S]
        step_c = spacing * math.sqrt(2)
        c = 0.0
        while c <= 2 * S + EPS:
            yield (-EPS, c + EPS, S + EPS, c - S - EPS)
            c += step_c


def fill_elements(edge_idx, color, S, hatch_spacing, clip_id):
    """
    Return (defs, elems) — lists of SVG strings for <defs> and drawing elements.

    defs:  zero or one <clipPath> definition (needed for circles / triangles).
    elems: blue <line>, <circle>, or <polygon> elements for vector engraving.

    Callers must place defs inside a <defs> block BEFORE referencing clip_id.
    clip_id must be unique within the SVG document.
    """
    tri = triangle_vertices(edge_idx, S)
    pat = COLOR_PATTERNS[color]

    if pat[0] == 'none':
        return [], []

    if pat[0] == 'solid':
        pts = ' '.join(f'{x:.4f},{y:.4f}' for x, y in tri)
        return [], [f'<polygon points="{pts}" fill="#000000" stroke="none"/>']

    if pat[0] == 'hatch':
        elems = []
        for angle in pat[1]:
            for seg in hatch_segments(angle, S, hatch_spacing):
                result = clip_line_to_poly(*seg, tri)
                if result is None:
                    continue
                ax, ay, bx, by = result
                if math.hypot(bx - ax, by - ay) < 0.01:
                    continue
                elems.append(
                    f'<line x1="{ax:.4f}" y1="{ay:.4f}"'
                    f' x2="{bx:.4f}" y2="{by:.4f}"'
                    f' stroke="#0000ff" stroke-width="{ETCH_WIDTH}" fill="none"/>'
                )
        return [], elems

    # Shapes that need a clipPath — build the clip polygon once.
    pts_str = ' '.join(f'{x:.4f},{y:.4f}' for x, y in tri)
    defs = [
        f'<clipPath id="{clip_id}">',
        f'  <polygon points="{pts_str}"/>',
        f'</clipPath>',
    ]
    clip_attr = f'clip-path="url(#{clip_id})"'
    stroke_attr = 'stroke="#0000ff" stroke-width="{ETCH_WIDTH}" fill="none"'

    if pat[0] == 'circles':
        cx, cy = S / 2, S / 2
        max_r = S * math.sqrt(2)
        elems = []
        r = hatch_spacing
        while r <= max_r + 1e-9:
            elems.append(
                f'<circle cx="{cx:.4f}" cy="{cy:.4f}" r="{r:.4f}"'
                f' {stroke_attr} {clip_attr}/>'
            )
            r += hatch_spacing
        return defs, elems

    if pat[0] == 'triangles':
        # Concentric scaled copies of the edge triangle, shrinking toward centroid.
        cx = sum(x for x, y in tri) / 3
        cy = sum(y for x, y in tri) / 3
        max_dist = max(math.hypot(x - cx, y - cy) for x, y in tri)
        delta_s = hatch_spacing / max_dist
        elems = []
        s = 1.0
        while s > 1e-6:
            scaled = [(cx + s * (x - cx), cy + s * (y - cy)) for x, y in tri]
            sp = ' '.join(f'{x:.4f},{y:.4f}' for x, y in scaled)
            elems.append(
                f'<polygon points="{sp}" {stroke_attr} {clip_attr}/>'
            )
            s -= delta_s
        return defs, elems

    return [], []


# ── Cut-path geometry ─────────────────────────────────────────────────────────
#
# Each edge is rendered as a jigsaw-style knob (tab) or notch (slot) using
# cubic Bézier curves.
#
# The shape is defined in a canonical local space:
#   u  — distance along the edge, from 0 to S
#   v  — distance perpendicular to the edge, positive = outward from tile
#
# The knob profile (for a tab) has three sections:
#   1. Straight run from edge corner to shoulder start
#   2. Shoulder: gently rises from the edge baseline up to the neck
#   3. Neck: slight narrowing as it widens into the knob head
#   4. Knob head: a rounded circular bump (two quarter-circle Béziers)
#   5. Mirror of 3, 2, 1 back to the far corner
#
# Slots are the same shape, mirrored inward (v negated), and slightly
# enlarged by `tolerance` in both width and depth for a snug laser-cut fit.
#
# The canonical shape is then mapped into SVG coordinates via a per-edge
# linear transform:
#   TOP    (u,v) → (u,       -v      )  outward = up   = -y
#   RIGHT  (u,v) → (S+v,     u       )  outward = right = +x
#   BOTTOM (u,v) → (S-u,     S+v     )  outward = down = +y  (edge traversed R→L)
#   LEFT   (u,v) → (-v,      S-u     )  outward = left = -x  (edge traversed B→T)
#
# Because all transforms are linear, they apply identically to Bézier
# control points and to anchor points.

def _tx(u, v, edge_idx, S):
    """Map canonical (u along edge, v outward) to SVG (x, y)."""
    if edge_idx == TOP:
        return (u,     -v)
    elif edge_idx == RIGHT:
        return (S + v,  u)
    elif edge_idx == BOTTOM:
        return (S - u,  S + v)
    else:  # LEFT
        return (-v,     S - u)


def _pt(u, v, edge_idx, S, sgn):
    """Format a canonical (u, sgn*v) point as 'x.xxxx,y.yyyy' for SVG."""
    x, y = _tx(u, v * sgn, edge_idx, S)
    return f"{x:.4f},{y:.4f}"


def edge_path_segment(edge_idx, color, S, tab_height, tab_widths, tolerance,
                      flat=False):
    """
    Return an SVG path fragment (no leading 'M') that traces one edge.

    Each edge starts at its own corner:
      TOP=(0,0)  RIGHT=(S,0)  BOTTOM=(S,S)  LEFT=(0,S)
    and ends at the next corner in CW order.

    flat=True produces a simple straight line (used for puzzle-border edges).

    The knob is a true jigsaw "mushroom" shape with a neck narrower than the
    head, so assembled pieces are locked laterally (cannot slide apart along
    the edge). The shoulder width still encodes the edge color via tab_widths.

    Shape profile (canonical u/v space, v=outward, tab case):

        edge baseline  ─────┬──────────────┬─────
                       sh_L │  shoulder     │ sh_R
                             │               │
                      nk_L ──┘   neck   └── nk_R    ← neck (narrower than head)
                                 │   │
                           ╭─────┴───┴─────╮
                           │   knob head   │        ← head radius > neck half-width
                           ╰───────────────╯
    """
    end_corners = [(S, 0), (S, S), (0, S), (0, 0)]
    ex, ey = end_corners[edge_idx]
    if flat:
        return f"L {ex:.4f},{ey:.4f}"

    th = tab_height
    tw = tab_widths[color]
    is_tab = IS_TAB[edge_idx]

    if not is_tab:
        tw += tolerance
        th += tolerance

    sgn = 1 if is_tab else -1

    mid = S / 2

    # ── Geometry ──────────────────────────────────────────────────────────
    #
    # shoulder_half  : half-width of the base footprint on the edge line.
    #                  Encodes edge color (wider = different color).
    # neck_v         : v-height where the neck is at its narrowest.
    # head_r         : radius of the circular knob head.
    # neck_half      : half-width of the neck — MUST be < head_r for undercut.
    # head_cy        : v-coord of the knob-circle centre.
    # head_top       : v-coord of the topmost point of the knob.
    #
    # We clamp head_r so the knob never exceeds tab_height, then derive
    # neck_half as a fixed fraction of head_r (always < head_r → undercut).

    shoulder_half = tw * 0.50               # encodes color
    neck_v        = th * 0.30               # neck base height
    head_r        = min(tw * 0.38, th * 0.34)   # clamped so head fits in th
    neck_half     = head_r * 0.68           # < head_r → lateral undercut
    head_cy       = neck_v + head_r         # circle centre
    # head_top = head_cy + head_r = neck_v + 2*head_r ≤ 0.30*th + 0.68*th < th ✓

    k = 0.5523 * head_r                     # Bézier magic for quarter-circle

    def P(u, v):
        return _pt(u, v, edge_idx, S, sgn)

    def C3(u1, v1, u2, v2, u3, v3):
        return f"C {P(u1,v1)} {P(u2,v2)} {P(u3,v3)}"

    parts = []

    # ── 1. Straight run to shoulder ───────────────────────────────────────
    parts.append(f"L {P(mid - shoulder_half, 0)}")

    # ── 2. Shoulder → neck (left side) ───────────────────────────────────
    # Rise from the baseline and pinch inward to the narrow neck.
    # cp1: stay near baseline, shift inward slightly
    # cp2: just below neck level, already at neck u-position
    parts.append(C3(mid - shoulder_half, th * 0.10,
                    mid - neck_half,     neck_v * 0.55,
                    mid - neck_half,     neck_v))

    # ── 3. Neck → head entry at 9-o'clock (left side) ────────────────────
    # Curve sweeps outward (u decreases) and upward to the 9-o'clock point.
    # Arrival must be tangent to the circle → direction is straight up at that pt.
    # cp1: go upward from neck
    # cp2: approach 9-o'clock horizontally from the right
    parts.append(C3(mid - neck_half, neck_v + (head_cy - neck_v) * 0.55,
                    mid - head_r,    head_cy * 0.85,
                    mid - head_r,    head_cy))

    # ── 4. Knob head: left quarter-circle (9 → 12 o'clock) ───────────────
    head_top = head_cy + head_r
    parts.append(C3(mid - head_r, head_cy + k,
                    mid - k,      head_top,
                    mid,          head_top))

    # ── 5. Knob head: right quarter-circle (12 → 3 o'clock) ──────────────
    parts.append(C3(mid + k,      head_top,
                    mid + head_r, head_cy + k,
                    mid + head_r, head_cy))

    # ── 6. Head exit → neck (right side, mirror of step 3) ───────────────
    parts.append(C3(mid + head_r, head_cy * 0.85,
                    mid + neck_half, neck_v + (head_cy - neck_v) * 0.55,
                    mid + neck_half, neck_v))

    # ── 7. Neck → shoulder (right side, mirror of step 2) ────────────────
    parts.append(C3(mid + neck_half,     neck_v * 0.55,
                    mid + shoulder_half, th * 0.10,
                    mid + shoulder_half, 0))

    # ── 8. Straight run to end corner ────────────────────────────────────
    parts.append(f"L {ex:.4f},{ey:.4f}")

    return " ".join(parts)


def make_path_d(edges, S, tab_height, tab_widths, tolerance, flat_edges=frozenset()):
    """SVG 'd' attribute for the closed cut outline of one tile."""
    # The TOP edge starts at (0,0).
    d = "M 0.0000,0.0000"
    for e in range(4):
        seg = edge_path_segment(e, edges[e], S, tab_height, tab_widths,
                                tolerance, flat=(e in flat_edges))
        d += " " + seg
    return d + " Z"


# ── SVG builder ───────────────────────────────────────────────────────────────

def render_svg(pieces, *, n_cols, tile_size, tab_height, tab_widths, tolerance,
               margin, hatch_spacing):
    if not pieces:
        return '<svg xmlns="http://www.w3.org/2000/svg"/>'

    S       = tile_size
    n_cols  = min(len(pieces), n_cols)
    n_rows  = math.ceil(len(pieces) / n_cols)
    cell_w  = S + tab_height + margin
    cell_h  = S + tab_height + margin
    W = margin + n_cols * cell_w
    H = margin + n_rows * cell_h

    def _pat_desc(p):
        if p[0] == 'none':    return 'none'
        if p[0] == 'solid':   return 'solid-black'
        if p[0] == 'hatch':   return 'hatch(' + ','.join(str(a) for a in p[1]) + ')deg'
        return p[0]
    pat_notes = "  ".join(
        f"{COLOR_NAMES[i]}: {_pat_desc(COLOR_PATTERNS[i])}"
        for i in range(len(COLOR_PATTERNS))
    )

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg"',
        f'     width="{W:.3f}mm" height="{H:.3f}mm"',
        f'     viewBox="0 0 {W:.4f} {H:.4f}">',
        f'  <desc>Wang Tile Laser-Cut Puzzle'
        f' | tile={S}mm tab={tab_height}mm tol={tolerance}mm hatch={hatch_spacing}mm'
        f' | {pat_notes}'
        f' | black=raster blue=engrave red=cut</desc>',
    ]

    for i, (edges, flat_edges) in enumerate(pieces):
        col, row = i % n_cols, i // n_cols
        ox = margin + col * cell_w
        oy = margin + row * cell_h
        label = "  ".join(
            f'{["T","R","B","L"][j]}:{COLOR_NAMES[e]}'
            + ('/flat' if j in flat_edges else '')
            for j, e in enumerate(edges)
        )

        lines.append(f'  <!-- piece {i+1:03d}  {label} -->')
        lines.append(f'  <g transform="translate({ox:.4f},{oy:.4f})">')

        # ── Engrave fills (process before cut) ────────────────────────────
        all_defs = []
        blue_elems = []
        for edge_idx in range(4):
            clip_id = f"cp{i+1:03d}e{edge_idx}"
            defs, elems = fill_elements(
                edge_idx, edges[edge_idx], S, hatch_spacing, clip_id
            )
            all_defs.extend(defs)
            blue_elems.extend(elems)

        if all_defs:
            lines.append('    <defs>')
            for d in all_defs:
                lines.append(f'      {d}')
            lines.append('    </defs>')

        if blue_elems:
            lines.append(f'    <g class="engrave">')
            for e in blue_elems:
                lines.append(f'      {e}')
            lines.append('    </g>')

        # ── Cut outline (red) ─────────────────────────────────────────────
        cut_d = make_path_d(edges, S, tab_height, tab_widths, tolerance, flat_edges)
        lines.append(
            f'    <path d="{cut_d}" fill="none" stroke="#ff0000"'
            f' stroke-width="{CUT_WIDTH}" stroke-linecap="square" stroke-linejoin="miter"/>'
        )
        lines.append('  </g>')

    lines.append('</svg>')
    return '\n'.join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input",  help="Wang Tile Designer JSON export")
    ap.add_argument("output", nargs="?",
                    help="Output SVG (default: input with .svg extension)")
    ap.add_argument("--tile-size",      type=float, default=TILE_SIZE,      metavar="MM",
                    help=f"Tile edge length in mm (default {TILE_SIZE})")
    ap.add_argument("--tab-height",     type=float, default=TAB_HEIGHT,     metavar="MM",
                    help=f"Tab/slot depth in mm (default {TAB_HEIGHT})")
    ap.add_argument("--margin",         type=float, default=MARGIN,         metavar="MM",
                    help=f"Gap between pieces in mm (default {MARGIN})")
    ap.add_argument("--tolerance",      type=float, default=TAB_TOLERANCE,  metavar="MM",
                    help=f"Slot enlargement for kerf compensation (default {TAB_TOLERANCE})")
    ap.add_argument("--hatch-spacing",  type=float, default=HATCH_SPACING,  metavar="MM",
                    help=f"Spacing between engraved hatch lines (default {HATCH_SPACING})")
    ap.add_argument("--stock-width",    type=float, default=STOCK_W_IN,     metavar="IN",
                    help=f"Stock sheet width in inches (default {STOCK_W_IN})")
    ap.add_argument("--stock-height",   type=float, default=STOCK_H_IN,     metavar="IN",
                    help=f"Stock sheet height in inches (default {STOCK_H_IN})")
    ap.add_argument("--border", action="store_true",
                    help="Give perimeter tiles flat outer edges (clean puzzle border)")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output) if args.output else src.with_suffix(".svg")

    try:
        data = json.loads(src.read_text())
    except FileNotFoundError:
        sys.exit(f"Error: '{src}' not found.")
    except json.JSONDecodeError as exc:
        sys.exit(f"Error parsing JSON: {exc}")

    tile_defs = {t["id"]: t["edges"] for t in data["tiles"]}
    usage     = Counter(tid for tid in data["grid"] if tid is not None)
    grid_n    = data["gridN"]

    max_color = max(e for t in data["tiles"] for e in t["edges"])
    if max_color >= len(TAB_WIDTHS):
        sys.exit(f"Error: color index {max_color} exceeds supported range "
                 f"0–{len(TAB_WIDTHS)-1}.")

    pieces = []
    if args.border:
        # Positional iteration — mark perimeter edges as flat.
        for idx, tid in enumerate(data["grid"]):
            if tid is None:
                continue
            row, col = divmod(idx, grid_n)
            flat = set()
            if row == 0:            flat.add(TOP)
            if row == grid_n - 1:   flat.add(BOTTOM)
            if col == 0:            flat.add(LEFT)
            if col == grid_n - 1:   flat.add(RIGHT)
            pieces.append((tile_defs[tid], frozenset(flat)))
    else:
        for tid, cnt in sorted(usage.items()):
            pieces.extend([(tile_defs[tid], frozenset())] * cnt)

    n_used   = len(usage)
    n_total  = sum(usage.values())
    n_unused = len(tile_defs) - n_used

    # ── Sheet layout from stock dimensions ───────────────────────────────────
    stock_w = args.stock_width  * MM_PER_IN
    stock_h = args.stock_height * MM_PER_IN
    cell    = args.tile_size + args.tab_height + args.margin
    n_cols  = max(1, int((stock_w - args.margin) / cell))
    n_rows  = max(1, int((stock_h - args.margin) / cell))
    per_sheet = n_cols * n_rows

    if per_sheet < 1:
        sys.exit("Error: stock dimensions are too small to fit even one piece. "
                 "Try a larger stock or a smaller tile size.")

    batches   = [pieces[i:i + per_sheet] for i in range(0, max(1, n_total), per_sheet)]
    n_sheets  = len(batches)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"Grid         : {grid_n}×{grid_n}")
    if args.border:
        n_border = 4 * grid_n - 4 if grid_n > 1 else 1
        print(f"Border       : ON — {n_border} perimeter pieces with flat outer edges")
    print(f"Tile types   : {len(tile_defs)} defined, {n_used} in grid"
          + (f", {n_unused} unused" if n_unused else ""))
    print(f"Pieces to cut: {n_total}")
    print()
    print(f"{'ID':>3}  {'Qty':>4}   {'Top':<7}{'Right':<7}{'Bottom':<7}{'Left'}")
    print("─" * 48)
    for tid, cnt in sorted(usage.items()):
        e = tile_defs[tid]
        print(f"{tid:>3}  {cnt:>4}×   "
              f"{COLOR_NAMES[e[0]]:<7}{COLOR_NAMES[e[1]]:<7}"
              f"{COLOR_NAMES[e[2]]:<7}{COLOR_NAMES[e[3]]}")

    # ── Render sheets ─────────────────────────────────────────────────────────
    render_kw = dict(
        n_cols=n_cols,
        tile_size=args.tile_size,
        tab_height=args.tab_height,
        tab_widths=TAB_WIDTHS,
        tolerance=args.tolerance,
        margin=args.margin,
        hatch_spacing=args.hatch_spacing,
    )

    written = []
    for idx, batch in enumerate(batches):
        if n_sheets == 1:
            out_path = dst
        else:
            out_path = dst.parent / f"{dst.stem}_{idx+1:02d}{dst.suffix}"
        out_path.write_text(render_svg(batch, **render_kw))
        written.append(out_path)

    # ── Report ────────────────────────────────────────────────────────────────
    sheet_w = args.margin + n_cols * cell
    sheet_h = args.margin + min(n_rows, math.ceil(n_total / n_cols)) * cell
    print()
    print(f"Stock        : {args.stock_width}\" × {args.stock_height}\"  "
          f"({stock_w:.1f} × {stock_h:.1f} mm)")
    print(f"Layout       : {n_cols} col × {n_rows} row = {per_sheet} pieces/sheet")
    print(f"Sheets       : {n_sheets}")
    for p in written:
        print(f"  {p}")
    print(f"SVG size     : {sheet_w:.1f} × {sheet_h:.1f} mm  (last sheet may be shorter)")
    print()
    print("Laser layers (assign in your laser software by color):")
    print("  #0000ff blue stroke  → vector engrave / score")
    print("  #ff0000 red stroke   → vector cut (through)")
    print()
    print("Visual patterns per edge color:")
    descs = [
        "blank (no marking)",
        "vertical blue lines   (90°)",
        "diagonal blue lines   (45°  /)",
        "blue crosshatch       (0° + 90°)",
        "concentric blue circles",
        "concentric blue triangles",
    ]
    for i, name in enumerate(COLOR_NAMES):
        print(f"  {i}  {name:<6}  {descs[i]}")


if __name__ == "__main__":
    main()
