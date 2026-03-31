#!/usr/bin/env python3
"""
server.py — Wang Tile fabrication pipeline server.

Serves index.html and two API endpoints:
  POST /generate  — { design, params } → { svgs: [...], stats: {...} }
  POST /export    — { design, params } → laser-cut-tiles.zip

Run from the project root or design-pipeline-tool/ directory:
  python design-pipeline-tool/server.py
  # then open http://localhost:8765
"""

import io
import json
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

HERE = Path(__file__).resolve().parent
TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3


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
