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

def _h_compat(a, b):
    """True if tile a can sit immediately left of tile b"""
    return a[_E] == b[_W]
 
 
def _v_compat(a, b):
    """True if tile a can sit immediately above tile b"""
    return a[_S] == b[_N]
 
 
def _any_h_neighbor(tile, tileset):
    """True if tile has at least one right-neighbor in tileset."""
    return any(_h_compat(tile, t) for t in tileset)
 
 
def _any_v_neighbor(tile, tileset):
    """True if tile has at least one tile that can go below it in tileset."""
    return any(_v_compat(tile, t) for t in tileset)
 
 
def _is_reachable(tile, tileset):
    """Heuristic: a tile is useful only if at least one other tile in the set
    can precede it horizontally or vertically (i.e. it isn't an island)."""
    return (
        any(_h_compat(t, tile) for t in tileset if t != tile) or
        any(_v_compat(t, tile) for t in tileset if t != tile)
    )


def gen_tileset(num_colors, num_tiles, rng, allowed=None):
    """Grow a tileset that spans a range of complexities.
    
    The strategy is as follows:
    1. Pick a random palette of num_colors from allowed
    2. Seed the tileset with one tile whose edges all come from the palette
    3. Grow the tileset by one of the 3 expansion moves, whose probabilities 
       shift as the tileset size grows

       attach - pick an existing tile and create a new one whose touching edge
       matches. The probability starts high and falls as the set grows.

       bridge - pick 2 existing tiles and create a new tile whose N matches
       tile-A's S and whose W matches tile-B's E. This connects sub-graphs,
       raising the branch factor. The probability grows with the set size.

       free - pick a tile with completely random edges from the palette. This
       adds variety but is only accepted if the candidate has at least one
       compatible neighbor in the set.
    """
    if not allowed:
        allowed = list(range(1, MAX_COLORS + 1))
    num_colors = max(1, min(num_colors, len(allowed)))
    palette = rng.sample(allowed, num_colors)
 
    def rand_pal():
        return rng.choice(palette)
 
    seed = tuple(rand_pal() for _ in range(4))
    tiles = [seed]
 
    attempts = 0
    while len(tiles) < num_tiles and attempts < 20_000:
        attempts += 1
        progress = len(tiles) / num_tiles
 
        p_attach = max(0.40, 0.80 - 0.40 * progress)
        p_bridge = min(0.35, 0.10 + 0.40 * progress)
        roll = rng.random()
 
        if roll < p_attach or len(tiles) < 2:
            # Attach: new tile shares one edge with a random existing tile
            src = rng.choice(tiles)
            if rng.random() < 0.5:
                candidate = (rand_pal(), rand_pal(), rand_pal(), src[_E])  # to the right
            else:
                candidate = (src[_S], rand_pal(), rand_pal(), rand_pal())  # below
 
        elif roll < p_attach + p_bridge:
            # Bridge: fix both N and W from two (possibly the same) existing tiles
            ta = rng.choice(tiles)
            tb = rng.choice(tiles)
            candidate = (ta[_S], rand_pal(), rand_pal(), tb[_E])
 
        else:
            # Free: random edges, accepted only if at least one neighbor exists
            candidate = tuple(rand_pal() for _ in range(4))
            if not (
                _any_h_neighbor(candidate, tiles) or
                _any_v_neighbor(candidate, tiles) or
                any(_h_compat(candidate, t) for t in tiles) or
                any(_v_compat(candidate, t) for t in tiles)
            ):
                continue
 
        if candidate not in tiles:
            tiles.append(candidate)
 
    # Connectivity pass: drop pure-island tiles
    cleaned = [t for t in tiles if _is_reachable(t, tiles)]
    if len(cleaned) < 2:
        cleaned = tiles[:2]  # fallback: always return at least 2
 
    return [list(t) for t in cleaned]


def mutate_tileset(parent_tiles, rng, n_muts=None, allowed=None):
    """Apply mutations to a tileset. n_muts=0 returns the tileset unchanged —
    useful when a single parent is selected and the user just wants puzzle-layout
    variation. `allowed` restricts which colors new_color/recolor may introduce.
    
    The operations on the tileset are as follows:
    new_color - clone a tile with one dege changed to a color that isn't in the set
    recolor - clone a tile with one edge changed to any palette color
    rotate - add the 90 degree clockwise rotation of a random tile
    flip - add the horizontal mirror of a random tile
    hybrid - pick two tiles, try N/E from A and S/W from B and its complement,
             accept the first that has at least one compatible neighbor in the set
    smart_delete - remove a tile with few compatible neighbors, randomly picking
                   amongst the bottom third to avoid repicking the same tile
    shift_color - find the most frequent edge color and replace all its occurences
                  in one tile with another
    """
    if not allowed:
        allowed = list(range(1, MAX_COLORS + 1))
    tiles = [tuple(t) for t in parent_tiles]
    if n_muts == 0:
        return [list(t) for t in tiles]
    if n_muts is None:
        n_muts = rng.randint(2, 5)
 
    ops = ["new_color", "recolor", "rotate", "flip",
           "hybrid", "smart_delete", "shift_color"]
 
    for _ in range(n_muts):
        op = rng.choice(ops)
 
        if op == "new_color":
            used = {c for t in tiles for c in t}
            choices = [c for c in allowed if c not in used] or list(allowed)
            a = rng.choice(tiles)
            t = list(a)
            t[rng.randint(0, 3)] = rng.choice(choices)
            candidate = tuple(t)
            if candidate not in tiles:
                tiles.append(candidate)
 
        elif op == "recolor":
            a = rng.choice(tiles)
            t = list(a)
            t[rng.randint(0, 3)] = rng.choice(allowed)
            candidate = tuple(t)
            if candidate not in tiles:
                tiles.append(candidate)
 
        elif op == "rotate":
            a = rng.choice(tiles)
            # 90 degrees clockwise
            candidate = (a[_W], a[_N], a[_E], a[_S])
            if candidate not in tiles:
                tiles.append(candidate)
 
        elif op == "flip":
            a = rng.choice(tiles)
            candidate = (a[_N], a[_W], a[_S], a[_E])  # swap E/W
            if candidate not in tiles:
                tiles.append(candidate)
 
        elif op == "hybrid":
            a = rng.choice(tiles)
            b = rng.choice(tiles)
            for cand in [
                (a[_N], a[_E], b[_S], b[_W]),  # N/E from a, S/W from b
                (b[_N], b[_E], a[_S], a[_W]),  # complementary split
            ]:
                if cand in tiles:
                    continue
                if (
                    _any_h_neighbor(cand, tiles) or
                    _any_v_neighbor(cand, tiles) or
                    any(_h_compat(cand, t) for t in tiles) or
                    any(_v_compat(cand, t) for t in tiles)
                ):
                    tiles.append(cand)
                    break
 
        elif op == "smart_delete":
            if len(tiles) <= 2:
                continue
 
            def neighbor_count(tile):
                return sum(
                    1 for t in tiles
                    if t is not tile and (
                        _h_compat(tile, t) or _h_compat(t, tile) or
                        _v_compat(tile, t) or _v_compat(t, tile)
                    )
                )
 
            scored = sorted(tiles, key=neighbor_count)
            # Pick randomly from the least-connected third to avoid always
            # deleting the same tile.
            bottom = scored[:max(1, len(scored) // 3)]
            tiles.remove(rng.choice(bottom))
 
        elif op == "shift_color":
            from collections import Counter
            freq = Counter(c for t in tiles for c in t)
            if not freq:
                continue
            dominant = freq.most_common(1)[0][0]
            victims = [t for t in tiles if dominant in t]
            if not victims:
                continue
            a = rng.choice(victims)
            replacement = rng.choice(
                [c for c in allowed if c != dominant] or list(allowed)
            )
            new_tile = tuple(replacement if c == dominant else c for c in a)
            if new_tile not in tiles:
                tiles.remove(a)
                tiles.append(new_tile)
 
    # Dedup, preserve order
    seen: set = set()
    out = []
    for t in tiles:
        if t not in seen:
            seen.add(t)
            out.append(list(t))
    return out



def crossover_tileset(parent_a, parent_b, rng, n_muts=None, allowed=None):
    """Sample a random subset of tiles from each of two parents, combine them,
    then apply fewer mutations than single-parent breeding (0–2 vs 2–4).
    
    Strategies for combining:
    union_sample - take random subsets from each parent and merge them
    compatible_first - seed with all of parent A, then absorb tiles from parent B
                       only if they share at least one compatible edge with something
                       already in the combined set
    color_matched - find edge colors that appear in both parents, pick one as a pivot
                    and include every tile from either parent that features it.
    """
    if not allowed:
        allowed = list(range(1, MAX_COLORS + 1))
 
    a_tiles = [tuple(t) for t in parent_a]
    b_tiles = [tuple(t) for t in parent_b]
 
    strategy = rng.choice(["union_sample", "compatible_first", "color_matched"])
 
    if strategy == "union_sample":
        n_from_a = rng.randint(max(1, len(a_tiles) // 2), len(a_tiles))
        n_from_b = rng.randint(max(1, len(b_tiles) // 2), len(b_tiles))
        raw = rng.sample(a_tiles, n_from_a) + rng.sample(b_tiles, n_from_b)
 
    elif strategy == "compatible_first":
        raw = list(a_tiles)
        for t in rng.sample(b_tiles, len(b_tiles)):
            if (
                _any_h_neighbor(t, raw) or _any_v_neighbor(t, raw) or
                any(_h_compat(t, r) for r in raw) or
                any(_v_compat(t, r) for r in raw)
            ):
                raw.append(t)
 
    else:  # color_matched
        colors_a = {c for t in a_tiles for c in t}
        colors_b = {c for t in b_tiles for c in t}
        shared = list(colors_a & colors_b)
        if shared:
            pivot = rng.choice(shared)
            raw = (
                [t for t in a_tiles if pivot in t] +
                [t for t in b_tiles if pivot in t]
            )
        else:
            # Fallback: union sample
            raw = (
                rng.sample(a_tiles, max(1, len(a_tiles) // 2)) +
                rng.sample(b_tiles, max(1, len(b_tiles) // 2))
            )
 
    # Dedup
    seen: set = set()
    combined = []
    for t in raw:
        if t not in seen:
            seen.add(t)
            combined.append(list(t))
 
    if not combined:
        combined = [list(rng.choice(a_tiles + b_tiles))]
 
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
