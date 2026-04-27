#!/usr/bin/env python3
"""
server.py — Wang Tile fabrication pipeline server.

Serves index.html and API endpoints:
  POST /generate  — { design, params } → { svgs: [...], stats: {...} }
  POST /export    — { design, params } → laser-cut-tiles.zip
  POST /discover  — { gridN } → { candidates: [Candidate x 9] }
  POST /breed     — { gridN, parents: [tileset] } → { candidates: [Candidate x 9] }

Run from the project root or design-pipeline-tool/ directory:
  python design-pipeline-tool/server.py
  # then open http://localhost:8765
"""

import io
import json
import random
import sys
import zipfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Import from parent directory (project root).
# resolve() is required so relative __file__ paths work when running from inside
# the design-pipeline-tool/ directory (e.g. `python3 server.py`).
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from wang_to_svg_curved import render_svg, TAB_WIDTHS, MM_PER_IN
from make_paint_key import make_key_svg

import torch
from counting import collect_puzzles
import knuth
from knuth import grade_difficulty

# grade_difficulty only uses unbounded_entropy_strip's output for inv_pressure,
# which is an informational metric and does NOT contribute to the score. The
# DFS inside it can OOM for large grid_h + dense compat graphs, so bypass it.
knuth.unbounded_entropy_strip = lambda *a, **kw: None

HERE = Path(__file__).resolve().parent
TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3
MAX_COLORS = 6  # matches index.html COLORS palette length
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_N, _E, _S, _W = 0, 1, 2, 3


def make_pieces(data, border=False):
    """Build the pieces list from a Wang Tile Designer JSON object."""
    tile_defs = {t["id"]: t["edges"] for t in data["tiles"]}
    usage = Counter(tid for tid in data["grid"] if tid is not None)
    grid_n = data["gridN"]
    pieces = []
    if border:
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
    return pieces, usage, tile_defs, grid_n


def generate(design, params):
    """Run the full SVG generation pipeline. Returns (svgs, stats)."""
    S   = float(params.get("tileSize",     40.0))
    th  = float(params.get("tabHeight",     5.0))
    mg  = float(params.get("margin",        6.0))
    tol = float(params.get("tolerance",    0.17))
    hs  = float(params.get("hatchSpacing",  3.0))
    sw  = float(params.get("stockWidth",   11.0)) * MM_PER_IN
    sh  = float(params.get("stockHeight",  11.0)) * MM_PER_IN
    brd = bool(params.get("border",       False))

    pieces, usage, tile_defs, grid_n = make_pieces(design, brd)

    cell = S + th + mg
    n_cols = max(1, int((sw - mg) / cell))
    n_rows = max(1, int((sh - mg) / cell))
    per_sheet = n_cols * n_rows
    n_total = sum(usage.values())

    batches = [pieces[i:i + per_sheet] for i in range(0, max(1, n_total), per_sheet)]
    kw = dict(n_cols=n_cols, tile_size=S, tab_height=th, tab_widths=TAB_WIDTHS,
              tolerance=tol, margin=mg, hatch_spacing=hs)
    svgs = [render_svg(batch, **kw) for batch in batches]

    n_used   = len(usage)
    n_unused = len(tile_defs) - n_used
    stats = {
        "grid":     f"{grid_n}×{grid_n}",
        "types":    f"{n_used} used" + (f", {n_unused} unused" if n_unused else ""),
        "pieces":   n_total,
        "perSheet": per_sheet,
        "sheets":   len(batches),
        "layout":   f"{n_cols} col × {n_rows} row",
    }
    return svgs, stats


# ═══════════════════════════════════════════════════════
#  PICBREEDER DISCOVERY / BREEDING
# ═══════════════════════════════════════════════════════
#
#  Tilesets are generated/mutated with 1-indexed edge colors in [1..MAX_COLORS],
#  which keeps "max_color+1" arithmetic simple. They get converted to the
#  0-indexed form the frontend expects (COLORS[] palette indices) on response.

_PRESETS = [
    # (band, num_colors, num_tiles) — initial batch is intentionally very simple;
    # the user evolves toward complexity via breeding.
    ("easy",   2, 3),
    ("easy",   2, 3),
    ("easy",   2, 4),
    ("medium", 2, 4),
    ("medium", 2, 5),
    ("medium", 3, 5),
    ("hard",   3, 6),
    ("hard",   3, 7),
    ("hard",   3, 8),
]


def gen_tileset(num_colors, num_tiles, rng, allowed=None):
    """Grow a tileset by attaching neighbors that share an edge — guarantees
    at least one valid 2-tile tiling exists. Mirrors knuth.generate_tileset,
    but biased toward reusing existing edge colors so small tilesets actually
    feel easy (matching edges are common rather than bottlenecked).

    The `num_colors` value is a count, not a range: we pick that many distinct
    colors uniformly from `allowed` (default: all colors) so each initial tileset
    visually looks different rather than always using colors 1-2."""
    if not allowed:
        allowed = list(range(1, MAX_COLORS + 1))
    num_colors = max(1, min(num_colors, len(allowed)))
    palette = rng.sample(allowed, num_colors)

    def rand_color(used):
        # 80% reuse an existing color (keeps edge-matching friendly),
        # 20% pick anything from this tileset's palette.
        if used and rng.random() < 0.8:
            return rng.choice(list(used))
        return rng.choice(palette)

    tiles = {tuple(rng.choice(palette) for _ in range(4))}
    attempts = 0
    while len(tiles) < num_tiles and attempts < 10000:
        attempts += 1
        used = {c for t in tiles for c in t}
        src = rng.choice(list(tiles))
        if rng.random() < 0.5:
            t = (rand_color(used), rand_color(used), rand_color(used), src[_E])
        else:
            t = (src[_S], rand_color(used), rand_color(used), rand_color(used))
        tiles.add(t)
    return [list(t) for t in tiles]


def mutate_tileset(parent_tiles, rng, n_muts=None, allowed=None):
    """Apply mutations to a tileset. n_muts=0 returns the tileset unchanged —
    useful when a single parent is selected and the user just wants puzzle-layout
    variation. The delete op may remove any tile, including parent tiles.
    `allowed` restricts which colors new_color/recolor may introduce."""
    if not allowed:
        allowed = list(range(1, MAX_COLORS + 1))
    tiles = [tuple(t) for t in parent_tiles]
    if n_muts == 0:
        return [list(t) for t in tiles]
    ops = ["new_color", "recolor", "rotate", "flip", "hybrid", "delete"]
    if n_muts is None:
        n_muts = rng.randint(2, 4)
    for _ in range(n_muts):
        op = rng.choice(ops)
        a = rng.choice(tiles)
        if op == "new_color":
            used = {c for t in tiles for c in t}
            choices = [c for c in allowed if c not in used] or list(allowed)
            t = list(a)
            t[rng.randint(0, 3)] = rng.choice(choices)
            tiles.append(tuple(t))
        elif op == "recolor":
            t = list(a)
            t[rng.randint(0, 3)] = rng.choice(allowed)
            tiles.append(tuple(t))
        elif op == "rotate":
            tiles.append((a[_W], a[_N], a[_E], a[_S]))
        elif op == "flip":
            tiles.append((a[_N], a[_W], a[_S], a[_E]))
        elif op == "hybrid":
            b = rng.choice(tiles)
            tiles.append((a[_N], a[_E], b[_S], b[_W]))
        elif op == "delete":
            if len(tiles) > 1:
                tiles.pop(rng.randint(0, len(tiles) - 1))
    # Dedup while preserving order
    seen = set()
    out = []
    for t in tiles:
        if t not in seen:
            seen.add(t)
            out.append(list(t))
    return out


def crossover_tileset(parent_a, parent_b, rng, n_muts=None, allowed=None):
    """Sample a random subset of tiles from each of two parents, combine them,
    then apply fewer mutations than single-parent breeding (0–2 vs 2–4)."""
    a_tiles = [tuple(t) for t in parent_a]
    b_tiles = [tuple(t) for t in parent_b]

    n_from_a = rng.randint(1, len(a_tiles))
    n_from_b = rng.randint(1, len(b_tiles))
    combined_raw = rng.sample(a_tiles, n_from_a) + rng.sample(b_tiles, n_from_b)

    seen = set()
    combined = []
    for t in combined_raw:
        if t not in seen:
            seen.add(t)
            combined.append(list(t))

    if n_muts is None:
        n_muts = rng.randint(0, 2)
    return mutate_tileset(combined, rng, n_muts=n_muts, allowed=allowed)


def _assign_bands(candidates):
    """Relative ranking: sort by score and split roughly into thirds. Gives a
    consistent green/yellow/red spread even when the raw scoring is skewed."""
    if not candidates:
        return
    order = sorted(range(len(candidates)), key=lambda i: candidates[i]["difficulty"]["score"])
    n = len(order)
    for rank, idx in enumerate(order):
        if rank < n / 3:            band = "easy"
        elif rank < 2 * n / 3:      band = "medium"
        else:                       band = "hard"
        candidates[idx]["difficulty"]["band"] = band


def build_candidate(tiles_1idx, gridN, rng):
    """Given a 1-indexed tileset, sample one puzzle and grade it. Returns the
    candidate dict (0-indexed for frontend) or None if no puzzle exists.

    Only the tiles actually present in the sampled puzzle are included in the
    returned tileset — the "empirical" tileset. The underlying generator may
    produce extra tiles that the solver never needed; shipping those would lie
    about what's in the puzzle and inflate the mutation space on breed."""
    tiles_t = torch.tensor(tiles_1idx, device=_DEVICE, dtype=torch.long)
    puzzles = collect_puzzles(
        tiles_t, gridN, gridN,
        target=1, batch_cols=2_000, batch_puzzles=32, max_stale=2,
    )
    if puzzles is None or puzzles.size(0) == 0:
        return None

    puzzle = puzzles[0]  # shape (W, L, 4), indexed [col][row]
    # Walk the puzzle cells and build the empirical tileset in order of first
    # appearance, so the shipped tile ids are 0..m-1 with no gaps.
    empirical = []                 # list of (n,e,s,w) tuples, 1-indexed colors
    empirical_lookup = {}          # tuple -> empirical id
    grid = [None] * (gridN * gridN)
    usage = Counter()
    for col in range(gridN):
        for row in range(gridN):
            key = tuple(puzzle[col][row].tolist())
            if key not in empirical_lookup:
                empirical_lookup[key] = len(empirical)
                empirical.append(key)
            tid = empirical_lookup[key]
            grid[row * gridN + col] = tid
            usage[tid] += 1

    # Re-grade against the empirical tileset so the difficulty score reflects
    # only what's actually in play.
    emp_t = torch.tensor([list(t) for t in empirical], device=_DEVICE, dtype=torch.long)
    grade = grade_difficulty(emp_t, dict(usage), gridN, gridN, num_samples=80)
    score = float(grade["difficulty_score"])

    return {
        "version": 1,
        "gridN":   gridN,
        # 0-indexed for COLORS[] palette
        "tiles":   [{"id": i, "edges": [c - 1 for c in empirical[i]]}
                    for i in range(len(empirical))],
        "grid":    grid,
        "difficulty": {
            "score": score,
            "label": grade["difficulty_label"],
            # "band" is assigned later by _assign_bands() once the whole batch is built
            "band":  "medium",
        },
    }


def build_candidate_with_retry(tiles_factory, gridN, rng, max_attempts=4):
    """tiles_factory(rng) -> tileset. Retries if no puzzle exists."""
    for _ in range(max_attempts):
        tiles = tiles_factory(rng)
        cand = build_candidate(tiles, gridN, rng)
        if cand is not None:
            return cand
    return None


def discover(gridN, allowed=None):
    rng = random.Random()
    out = []
    for _band, num_colors, num_tiles in _PRESETS:
        cand = build_candidate_with_retry(
            lambda r, nc=num_colors, nt=num_tiles: gen_tileset(nc, nt, r, allowed),
            gridN, rng,
        )
        if cand is not None:
            out.append(cand)
    _assign_bands(out)
    return out


def breed(gridN, parents, allowed=None):
    """Parents: list of tilesets in the frontend 0-indexed shape
    [{id, edges:[n,e,s,w]}, ...]. Produces ~9 children via mutation and crossover.

    With one parent: re-rolls the puzzle layout without mutating the tileset.
    With multiple parents: roughly half the children come from single-parent
    mutation (2–4 ops), half from two-parent crossover (0–2 ops)."""
    rng = random.Random()
    parents_1idx = [[[e + 1 for e in t["edges"]] for t in p] for p in parents]

    single_parent = len(parents_1idx) == 1
    out = []
    attempts = 0
    while len(out) < 9 and attempts < 27:
        attempts += 1
        if single_parent:
            pa = parents_1idx[0]
            factory = lambda r, p=pa: mutate_tileset(p, r, n_muts=0, allowed=allowed)
        elif len(parents_1idx) >= 2 and rng.random() < 0.5:
            pa, pb = rng.sample(parents_1idx, 2)
            factory = lambda r, a=pa, b=pb: crossover_tileset(a, b, r, allowed=allowed)
        else:
            pa = rng.choice(parents_1idx)
            factory = lambda r, p=pa: mutate_tileset(p, r, allowed=allowed)

        cand = build_candidate_with_retry(factory, gridN, rng)
        if cand is not None:
            out.append(cand)

    out = out[:9]
    _assign_bands(out)
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {fmt % args}")

    def _send(self, code, content_type, body):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", (HERE / "index.html").read_bytes())
        else:
            self._send(404, "text/plain", "Not found")

    def do_POST(self):
        # Discovery/breeding endpoints have a different payload shape; route first.
        if self.path in ("/discover", "/breed"):
            try:
                payload = self._read_body()
                gridN = int(payload.get("gridN", 8))
                gridN = max(2, min(25, gridN))
                # allowedColors arrives as 0-indexed; convert to internal 1-indexed.
                ac = payload.get("allowedColors")
                allowed = [c + 1 for c in ac] if ac else None
                if self.path == "/discover":
                    candidates = discover(gridN, allowed=allowed)
                else:
                    candidates = breed(gridN, payload.get("parents", []), allowed=allowed)
                self._send(200, "application/json",
                           json.dumps({"candidates": candidates}))
            except Exception as exc:
                import traceback; traceback.print_exc()
                self._send(400, "application/json", json.dumps({"error": str(exc)}))
            return

        try:
            payload = self._read_body()
            svgs, stats = generate(payload["design"], payload.get("params", {}))
        except Exception as exc:
            self._send(400, "application/json", json.dumps({"error": str(exc)}))
            return

        if self.path == "/generate":
            self._send(200, "application/json", json.dumps({"svgs": svgs, "stats": stats}))

        elif self.path == "/export":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                svg_names = []
                for i, svg in enumerate(svgs):
                    name = f"sheet_{i+1:02d}.svg" if len(svgs) > 1 else "tiles.svg"
                    zf.writestr(name, svg)
                    svg_names.append(name)
                hs = float(payload.get("params", {}).get("hatchSpacing", 3.0))
                zf.writestr("paint_key.svg", make_key_svg(hatch_spacing=hs))
                file_list = ", ".join(svg_names)
                readme = (
                    "Wang Tile Laser-Cut Puzzle\n"
                    "==========================\n\n"
                    "Files in this archive:\n"
                    f"  {file_list}    laser cut these (red = cut through, blue = engrave)\n"
                    "  paint_key.svg   reference for painting each edge color\n\n"
                    "Instructions:\n"
                    f"  1. Laser cut: {file_list}\n"
                      "  2. Assemble the puzzle: Work through the sheet files in\n"
                    "     order (sheet_01, sheet_02, …). Within each sheet, pieces\n"
                    "     fill left-to-right, top-to-bottom (row-wise). Lay them\n"
                    "     out in that sequence to reconstruct the original pattern.\n"
                    "     Tabs and slots are color-coded so each piece only fits\n"
                    "     in the correct orientation.\n"
                    "  3. Paint each edge triangle according to paint_key.svg.\n"
                    "     The engraved patterns identify which color goes where.\n"
                )
                zf.writestr("README.txt", readme)
            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", 'attachment; filename="laser-cut-tiles.zip"')
            self.send_header("Content-Length", len(data))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        else:
            self._send(404, "text/plain", "Not found")


if __name__ == "__main__":
    port = 8765
    print(f"Wang Tile Design Pipeline  →  http://localhost:{port}")
    print("Ctrl+C to stop\n")
    HTTPServer(("localhost", port), Handler).serve_forever()
