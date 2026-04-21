#!/usr/bin/env python3
"""
difficulty.py — Knuth estimator + difficulty grading for Wang tilings.

Builds on counting.py. Works with bounded tile inventories.
"""

import torch
import math
import random
from collections import Counter

# Edge indices — prefixed to avoid shadowing grid dimension params
_N, _E, _S, _W = 0, 1, 2, 3

def estimate_human_difficulty(tiles, inventory, grid_h, grid_w, num_samples=3000):
    """
    Estimates puzzle difficulty from a human solver's perspective.

    Humans pattern-match locally — they fill forced cells, guess when stuck,
    and struggle when wrong guesses propagate far before revealing themselves.

    Returns a dict with interpretable metrics and a 0-100 difficulty score.
    """
    tiles_cpu = tiles.cpu()
    num_types = len(tiles_cpu)
    area = grid_h * grid_w

    # Preallocate grid once, reset per sample
    grid = [[None] * grid_h for _ in range(grid_w)]

    forced_move_counts = []       # fraction of placements that were forced
    error_propagation_gaps = []   # steps between a guess and its contradiction
    guess_depths = []             # how many guesses deep before a dead end
    first_guess_steps = []        # where in the grid the first real choice appears
    completion_rates = []         # 1.0 if solved, else fraction completed

    def get_valid(col, row, remaining):
        valid = []
        for t in range(num_types):
            if remaining.get(t, 0) <= 0:
                continue
            tile = tiles_cpu[t]
            if row > 0 and tiles_cpu[grid[col][row - 1]][_S] != tile[_N]:
                continue
            if col > 0 and tiles_cpu[grid[col - 1][row]][_E] != tile[_W]:
                continue
            valid.append(t)
        return valid

    for _ in range(num_samples):
        # Reset grid
        for c in range(grid_w):
            for r in range(grid_h):
                grid[c][r] = None

        remaining = dict(inventory)
        forced = 0
        guesses = []          # list of (step, num_choices) for each non-forced placement
        dead = False
        steps_completed = 0

        for step in range(area):
            col, row = divmod(step, grid_h)
            valid = get_valid(col, row, remaining)

            if not valid:
                dead = True
                # How far did we get from the last guess before hitting a wall?
                if guesses:
                    error_propagation_gaps.append(step - guesses[-1][0])
                break

            if len(valid) == 1:
                forced += 1
            else:
                guesses.append((step, len(valid)))
                if not first_guess_steps or len(first_guess_steps) < len(forced_move_counts) + 1:
                    # first guess in this sample
                    pass

            chosen = random.choice(valid)
            grid[col][row] = chosen
            remaining[chosen] -= 1
            steps_completed += 1

        forced_move_counts.append(forced / max(steps_completed, 1))
        guess_depths.append(len(guesses))
        completion_rates.append(steps_completed / area)

        if not dead and guesses:
            first_guess_steps.append(guesses[0][0])

    # --- Aggregate metrics ---

    avg_forced_ratio = sum(forced_move_counts) / num_samples
    # How often a random walk completes (proxy for solution density)
    completion_rate = sum(completion_rates) / num_samples
    dead_end_rate = 1.0 - (sum(1 for c in completion_rates if c == 1.0) / num_samples)

    avg_error_gap = (
        sum(error_propagation_gaps) / len(error_propagation_gaps)
        if error_propagation_gaps else 0.0
    )
    avg_guess_depth = sum(guess_depths) / num_samples

    avg_first_guess = (
        sum(first_guess_steps) / len(first_guess_steps)
        if first_guess_steps else float(area)
    )

    # --- Scoring (0-100) ---
    #
    # forced_ratio:   high = easy (lots of forced moves, little real choice)
    # error_gap:      high = hard (mistakes propagate far before detection)
    # guess_depth:    high = hard (many real decisions required)
    # dead_end_rate:  high = hard (most paths fail)
    # first_guess:    low  = hard (forced to guess early with little context)

    # Each component scaled to 0-25 then summed
    forced_score    = (1.0 - avg_forced_ratio) * 25          # low forced moves = hard
    error_score     = min(25, (avg_error_gap / area) * 50)   # long gaps = hard
    guess_score     = min(25, (avg_guess_depth / area) * 50) # many guesses = hard
    dead_score      = dead_end_rate * 25                     # high dead ends = hard

    difficulty = min(100.0, forced_score + error_score + guess_score + dead_score)

    labels = [
        (0,  20,  "Trivial"),
        (20, 40,  "Easy"),
        (40, 60,  "Moderate"),
        (60, 80,  "Hard"),
        (80, 101, "Brutal"),
    ]
    label = next(lab for lo, hi, lab in labels if lo <= difficulty < hi)

    return {
        "difficulty_score":        difficulty,
        "difficulty_label":        label,
        # Human-interpretable breakdown
        "forced_move_ratio":       avg_forced_ratio,
        "avg_error_propagation":   avg_error_gap,
        "avg_guess_depth":         avg_guess_depth,
        "avg_first_guess_step":    avg_first_guess,
        "dead_end_rate":           dead_end_rate,
        "completion_rate":         completion_rate,
        # Scoring components (each 0-25)
        "score_forced":            forced_score,
        "score_error_propagation": error_score,
        "score_guess_depth":       guess_score,
        "score_dead_ends":         dead_score,
        # Meta
        "grid":                    (grid_w, grid_h),
        "area":                    area,
        "num_samples":             num_samples,
    }

def knuth_estimate(tiles, inventory, grid_h, grid_w, num_samples=5000):
    tiles_cpu = tiles.cpu()
    num_types = len(tiles_cpu)
    area = grid_h * grid_w

    estimates = []
    branching_at_step = [[] for _ in range(area)]
    dead_end_steps = []
    lookahead_depths = []
    solutions_found = 0

    for _ in range(num_samples):
        grid = [[None] * grid_h for _ in range(grid_w)]
        remaining = dict(inventory)
        weight = 1.0
        choice_points = []  # track in same pass

        for step in range(area):
            col, row = divmod(step, grid_h)

            valid = [
                t for t in range(num_types)
                if remaining.get(t, 0) > 0
                and (row == 0 or tiles_cpu[grid[col][row-1]][_S] == tiles_cpu[t][_N])
                and (col == 0 or tiles_cpu[grid[col-1][row]][_E] == tiles_cpu[t][_W])
            ]

            branching_at_step[step].append(len(valid))

            if not valid:
                dead_end_steps.append(step)
                if choice_points:
                    lookahead_depths.append(step - choice_points[-1])
                estimates.append(0.0)
                break

            if len(valid) > 1:
                choice_points.append(step)

            weight *= len(valid)
            chosen = random.choice(valid)
            grid[col][row] = chosen
            remaining[chosen] -= 1
        else:
            estimates.append(weight)
            solutions_found += 1

    est_count = sum(estimates) / num_samples
    est_var = sum((e - est_count) ** 2 for e in estimates) / num_samples
    est_std = math.sqrt(est_var / num_samples)

    avg_branching = [
        sum(bp) / len(bp) if bp else 0.0
        for bp in branching_at_step
    ]

    zero_freq = [
        sum(1 for b in bp if b == 0) / len(bp) if bp else 0.0
        for bp in branching_at_step
    ]

    dead_end_rate = len(dead_end_steps) / num_samples
    avg_dead_end_pos = (
        sum(dead_end_steps) / len(dead_end_steps)
        if dead_end_steps else area
    )

    # Lookahead depth: when a walk dies at step k, find the last step
    # before k where branching was > 1. The gap is the lookahead needed.
    lookahead_depths = []
    for _ in range(min(num_samples, 2000)):
        grid = [[None] * grid_h for _ in range(grid_w)]
        remaining = dict(inventory)
        choice_points = []

        for step in range(area):
            col = step // grid_h
            row = step % grid_h

            valid_list = []
            for t in range(num_types):
                if remaining.get(t, 0) <= 0:
                    continue
                tile = tiles_cpu[t]
                if row > 0 and tiles_cpu[grid[col][row - 1]][_S] != tile[_N]:
                    continue
                if col > 0 and tiles_cpu[grid[col - 1][row]][_E] != tile[_W]:
                    continue
                valid_list.append(t)

            if len(valid_list) == 0:
                if choice_points:
                    lookahead_depths.append(step - choice_points[-1])
                break

            if len(valid_list) > 1:
                choice_points.append(step)

            chosen = random.choice(valid_list)
            grid[col][row] = chosen
            remaining[chosen] -= 1

    avg_lookahead = (
        sum(lookahead_depths) / len(lookahead_depths)
        if lookahead_depths else 0.0
    )

    return {
        "estimated_count": est_count,
        "std_error": est_std,
        "confidence_95": (max(0, est_count - 1.96 * est_std),
                          est_count + 1.96 * est_std),
        "solutions_found": solutions_found,
        "dead_end_rate": dead_end_rate,
        "avg_dead_end_position": avg_dead_end_pos,
        "avg_branching_profile": avg_branching,
        "zero_choice_frequency": zero_freq,
        "avg_lookahead_depth": avg_lookahead,
        "num_samples": num_samples,
    }


def unbounded_entropy_strip(tiles, grid_h, strip_width=5):
    """
    Estimate unbounded entropy per site using a narrow strip.

    Returns entropy per site (float), or None if intractable.
    """
    import numpy as np
    tiles_cpu = tiles.cpu().numpy()
    num_types = len(tiles_cpu)

    if num_types ** min(grid_h, 8) > 5_000_000:
        return None

    valid_cols = []

    def dfs(partial):
        if len(partial) == grid_h:
            valid_cols.append(tuple(partial))
            return
        bottom_idx = partial[-1]
        bottom = tiles_cpu[bottom_idx]
        for t in range(num_types):
            if bottom[_S] == tiles_cpu[t][_N]:
                partial.append(t)
                dfs(partial)
                partial.pop()

    for t in range(num_types):
        dfs([t])

    if not valid_cols:
        return float("-inf")

    n = len(valid_cols)
    if n > 10000:
        return None

    T = np.zeros((n, n), dtype=np.float64)
    for i, ca in enumerate(valid_cols):
        for j, cb in enumerate(valid_cols):
            ok = True
            for r in range(grid_h):
                if tiles_cpu[ca[r]][_E] != tiles_cpu[cb[r]][_W]:
                    ok = False
                    break
            if ok:
                T[i, j] = 1

    if strip_width == 1:
        count = n
    else:
        Tk = np.linalg.matrix_power(T, strip_width - 1)
        count = Tk.sum()

    if count <= 0:
        return float("-inf")

    return math.log(count) / (grid_h * strip_width)


def color_frequency_entropy(tiles, inventory):
    """
    Entropy of the edge-color frequency distribution.

    Low entropy = skewed colors = bottleneck difficulty.
    High entropy = uniform colors = uniform difficulty.
    """
    tiles_cpu = tiles.cpu()
    color_counts = Counter()
    for t_idx, count in inventory.items():
        tile = tiles_cpu[t_idx]
        for edge in range(4):
            color_counts[tile[edge].item()] += count

    total = sum(color_counts.values())
    if total == 0:
        return 0.0

    ent = 0.0
    for c in color_counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p)
    return ent


def constraint_tightness(tiles):
    """Fraction of tile pairs that are horizontally incompatible."""
    tiles_cpu = tiles.cpu()
    k = len(tiles_cpu)
    incompat = 0
    for a in range(k):
        for b in range(k):
            if tiles_cpu[a][_E] != tiles_cpu[b][_W]:
                incompat += 1
    return incompat / (k * k)


def grade_difficulty(tiles, inventory, grid_h, grid_w, num_samples=10000):
    """
    Full difficulty analysis. Returns a dict with all metrics + score.
    """
    area = grid_h * grid_w
    total_tiles = sum(inventory.values())

    ke = knuth_estimate(tiles, inventory, grid_h, grid_w, num_samples)

    ub_entropy = unbounded_entropy_strip(tiles, grid_h, strip_width=min(grid_w, 6))

    tightness = constraint_tightness(tiles)
    color_ent = color_frequency_entropy(tiles, inventory)

    if ub_entropy is not None and ub_entropy > 0:
        ub_approx_count = math.exp(ub_entropy * area)
        inv_pressure = 1.0 - min(1.0, ke["estimated_count"] / max(ub_approx_count, 1))
    else:
        inv_pressure = None

    bounded_entropy = (
        math.log(ke["estimated_count"]) / area
        if ke["estimated_count"] > 1 else 0.0
    )

    profile = ke["avg_branching_profile"]
    mean_branching = sum(profile) / len(profile) if profile else 0

    ent_score = max(0, 1 - bounded_entropy / 0.5) * 25
    dead_score = ke["dead_end_rate"] * 25
    branch_score = max(0, 1 - mean_branching / 5) * 15
    sol_score = (20 if ke["estimated_count"] <= 1
                 else 10 if ke["estimated_count"] <= 10
                 else 5 if ke["estimated_count"] <= 100
                 else 0)
    lookahead_score = min(15, ke["avg_lookahead_depth"] * 1.5)

    difficulty = min(100, ent_score + dead_score + branch_score + sol_score + lookahead_score)

    labels = [
        (0, 20, "Trivial"),
        (20, 40, "Easy"),
        (40, 60, "Moderate"),
        (60, 80, "Hard"),
        (80, 101, "Brutal"),
    ]
    label = next(lab for lo, hi, lab in labels if lo <= difficulty < hi)

    return {
        "estimated_solutions": ke["estimated_count"],
        "confidence_95": ke["confidence_95"],
        "dead_end_rate": ke["dead_end_rate"],
        "avg_dead_end_position": ke["avg_dead_end_position"],
        "avg_branching_profile": ke["avg_branching_profile"],
        "avg_lookahead_depth": ke["avg_lookahead_depth"],
        "solutions_found": ke["solutions_found"],
        "unbounded_entropy": ub_entropy,
        "bounded_entropy": bounded_entropy,
        "constraint_tightness": tightness,
        "color_frequency_entropy": color_ent,
        "inventory_pressure": inv_pressure,
        "difficulty_score": difficulty,
        "difficulty_label": label,
        "grid": (grid_w, grid_h),
        "area": area,
        "tile_types": len(tiles),
        "total_tiles": total_tiles,
        "num_samples": num_samples,
    }


def print_report(result):
    """Pretty-print a difficulty report."""
    grid_w, grid_h = result["grid"]
    area = result["area"]

    print()
    print(f"{'=' * 60}")
    print(f"  Wang Tiling Difficulty Report — {grid_w}×{grid_h} grid")
    print(f"{'=' * 60}")
    print(f"  Tile types:          {result['tile_types']}")
    print(f"  Total tiles:         {result['total_tiles']} (need {area})")
    print()

    print(f"  --- Solution Count ---")
    lo, hi = result["confidence_95"]
    print(f"  Estimated solutions: {result['estimated_solutions']:.1f}")
    print(f"  95% CI:              [{lo:.1f}, {hi:.1f}]")
    print(f"  Walks finding soln:  {result['solutions_found']}/{result['num_samples']}")
    print()

    print(f"  --- Dead Ends ---")
    print(f"  Dead end rate:       {result['dead_end_rate']:.1%}")
    print(f"  Avg dead end at:     step {result['avg_dead_end_position']:.1f} of {area}")
    print(f"  Avg lookahead depth: {result['avg_lookahead_depth']:.1f} steps")
    print()

    print(f"  --- Entropy ---")
    if result["unbounded_entropy"] is not None:
        print(f"  Unbounded (strip):   {result['unbounded_entropy']:.4f}")
    else:
        print(f"  Unbounded (strip):   [grid too large to compute]")
    print(f"  Bounded:             {result['bounded_entropy']:.4f}")
    print()

    print(f"  --- Structural ---")
    print(f"  Constraint tight.:   {result['constraint_tightness']:.1%}")
    print(f"  Color freq entropy:  {result['color_frequency_entropy']:.3f}")
    if result["inventory_pressure"] is not None:
        print(f"  Inventory pressure:  {result['inventory_pressure']:.3f}")
    else:
        print(f"  Inventory pressure:  [unbounded count unavailable]")
    print()

    print(f"  --- Branching Profile ---")
    profile = result["avg_branching_profile"]
    cols_to_show = set([0, grid_w // 2, grid_w - 1])
    for step in range(area):
        col = step // grid_h
        row = step % grid_h
        if col in cols_to_show:
            bar = "█" * int(profile[step] * 2)
            print(f"    [{col:2d},{row:2d}] {profile[step]:5.1f} {bar}")
        elif col == 1 and row == 0:
            print(f"    ...")
    print()

    print(f"{'=' * 60}")
    print(f"  DIFFICULTY: {result['difficulty_score']:.0f}/100 — {result['difficulty_label']}")
    print(f"{'=' * 60}")
    print()


def generate_tileset(num_colors, num_tiles, seed=None):
    """
    Generate a tileset guaranteed to have at least one valid tiling.

    Strategy: start with a seed tile, then grow the tileset by ensuring
    every east edge has at least one tile with a matching west edge,
    and every south edge has at least one tile with a matching north edge.
    """
    rng = random.Random(seed)

    tiles = set()

    # Seed tile
    first = tuple(rng.randint(1, num_colors) for _ in range(4))
    tiles.add(first)

    # Grow until we have enough, ensuring connectivity
    attempts = 0
    while len(tiles) < num_tiles and attempts < 10000:
        attempts += 1

        # Pick a random existing tile and build a compatible neighbor
        src = rng.choice(list(tiles))
        direction = rng.choice(['east', 'south'])

        if direction == 'east':
            # New tile's west must match src's east
            w = src[_E]
            n = rng.randint(1, num_colors)
            e = rng.randint(1, num_colors)
            s = rng.randint(1, num_colors)
            tiles.add((n, e, s, w))
        else:
            # New tile's north must match src's south
            n = src[_S]
            e = rng.randint(1, num_colors)
            s = rng.randint(1, num_colors)
            w = rng.randint(1, num_colors)
            tiles.add((n, e, s, w))

    return torch.tensor(list(tiles), device=device)

if __name__ == "__main__":
    from counting import collect_puzzles

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tiles = generate_tileset(1, 8).to(device)

    grid_h = 25
    grid_w = 25

    print("Sampling puzzles via transfer matrix...")
    puzzles = collect_puzzles(tiles, grid_h, grid_w, target=100, batch_cols=100_000)
    if puzzles is not None:
        print(f"Got {puzzles.size(0)} puzzles of shape {puzzles.shape}")

        for i in range(puzzles.size(0)):
            print(f"Grading puzzle {i}")
            puzzle = puzzles[i]
            flat = puzzle.reshape(-1, 4)
            usage = Counter()
            for row in flat:
                for t_idx in range(len(tiles)):
                    if torch.equal(row, tiles[t_idx]):
                        usage[t_idx] = usage.get(t_idx, 0) + 1
                        break

            result = grade_difficulty(tiles, dict(usage), grid_h, grid_w, num_samples=100)
            print_report(result)
    else:
        print("No puzzles found.")
