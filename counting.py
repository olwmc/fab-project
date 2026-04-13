
#!/usr/bin/env python3
import torch

N, E, S, W = 0, 1, 2, 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tile_compat(tiles):
    """Vertical compatibility matrix."""
    return (tiles[:, S][:, None] == tiles[:, N][None, :]).double()


def sample_columns(tiles, compat, L, k):
    """Returns (k, L) tensor of tile indices."""
    n = len(tiles)

    counts = torch.zeros(L, n, device=device, dtype=torch.float64)
    counts[-1, :] = 1
    for l in range(L - 2, -1, -1):
        counts[l] = compat @ counts[l + 1]

    total = counts[0].sum()
    if total == 0:
        return None

    col_indices = torch.zeros(k, L, device=device, dtype=torch.long)
    col_indices[:, 0] = torch.multinomial((counts[0] / total).float(), k, replacement=True)

    for l in range(1, L):
        prev = col_indices[:, l - 1]
        probs = compat[prev] * counts[l].unsqueeze(0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        col_indices[:, l] = torch.multinomial(probs.float(), 1).squeeze(1)

    return col_indices


def build_transfer_sparse(tiles, col_indices):
    """Build sparse transfer matrix by hashing east/west edge vectors."""
    cols = tiles[col_indices]
    east = cols[:, :, E]
    west = cols[:, :, W]
    n = len(cols)

    west_cpu = west.cpu()
    east_cpu = east.cpu()

    west_buckets = {}
    for j in range(n):
        key = west_cpu[j].numpy().tobytes()
        if key in west_buckets:
            west_buckets[key].append(j)
        else:
            west_buckets[key] = [j]

    rows = []
    cols_idx = []
    for i in range(n):
        key = east_cpu[i].numpy().tobytes()
        bucket = west_buckets.get(key)
        if bucket is not None:
            for j in bucket:
                rows.append(i)
                cols_idx.append(j)

    if not rows:
        values = torch.zeros(0, device=device, dtype=torch.float64)
        indices = torch.zeros(2, 0, device=device, dtype=torch.long)
    else:
        rows = torch.tensor(rows, device=device, dtype=torch.long)
        cols_idx = torch.tensor(cols_idx, device=device, dtype=torch.long)
        values = torch.ones(len(rows), device=device, dtype=torch.float64)
        indices = torch.stack([rows, cols_idx])

    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def sample_puzzles(col_indices, T, W, k):
    """Returns (k, W) tensor of column indices."""
    n = T.size(0)

    counts = torch.zeros(W, n, device=device, dtype=torch.float64)
    counts[-1, :] = 1
    for w in range(W - 2, -1, -1):
        counts[w] = torch.sparse.mm(T, counts[w + 1].unsqueeze(1)).squeeze(1)

    total = counts[0].sum()
    if total == 0:
        return None

    puzzle_indices = torch.zeros(k, W, device=device, dtype=torch.long)
    puzzle_indices[:, 0] = torch.multinomial((counts[0] / total).float(), k, replacement=True)

    T_dense_cache = {}
    for w in range(1, W):
        prev = puzzle_indices[:, w - 1]
        probs = torch.zeros(k, n, device=device, dtype=torch.float64)

        for idx in prev.unique():
            i = idx.item()
            if i not in T_dense_cache:
                T_dense_cache[i] = T[i].to_dense()
            which = prev == idx
            probs[which] = T_dense_cache[i] * counts[w]

        probs = probs / probs.sum(dim=1, keepdim=True)
        puzzle_indices[:, w] = torch.multinomial(probs.float(), 1).squeeze(1)

    return puzzle_indices


def puzzles_to_tiles(tiles, col_indices, puzzle_indices):
    """Convert index tensors to (num_puzzles, W, L, 4) tile tensor."""
    col_tile_indices = col_indices[puzzle_indices]
    return tiles[col_tile_indices]


def is_valid_puzzle(puzzle):
    """puzzle is (W, L, 4) tensor."""
    for col in puzzle:
        for i in range(len(col) - 1):
            if col[i][S] != col[i + 1][N]:
                return False

    for c in range(len(puzzle) - 1):
        for i in range(len(puzzle[c])):
            if puzzle[c][i][E] != puzzle[c + 1][i][W]:
                return False

    return True


def collect_puzzles(tiles, L, W, target, batch_cols=50_000, batch_puzzles=500, max_stale=50):
    compat = tile_compat(tiles)
    seen = set()
    all_puzzles = []
    stale = 0

    while len(all_puzzles) < target:
        col_idx = sample_columns(tiles, compat, L, batch_cols)
        if col_idx is None:
            stale += 1
            if stale >= max_stale:
                break
            continue

        T = build_transfer_sparse(tiles, col_idx)

        want = min(batch_puzzles, target - len(all_puzzles))
        puz_idx = sample_puzzles(col_idx, T, W, want)
        if puz_idx is None:
            stale += 1
            if stale >= max_stale:
                break
            continue

        full = puzzles_to_tiles(tiles, col_idx, puz_idx)
        flat = full.reshape(full.size(0), -1)

        found_new = False
        for i in range(flat.size(0)):
            key = flat[i].cpu().numpy().tobytes()
            if key not in seen:
                seen.add(key)
                all_puzzles.append(full[i])
                found_new = True
                if len(all_puzzles) >= target:
                    break

        if found_new:
            stale = 0
        else:
            stale += 1
            if stale >= max_stale:
                print(f"Gave up after {max_stale} stale rounds, found {len(all_puzzles)}/{target}")
                break

        print(f"  {len(all_puzzles)}/{target}")

    return torch.stack(all_puzzles) if all_puzzles else None


if __name__ == "__main__":
    tiles = torch.tensor([
        (1, 1, 1, 1),
        (1, 2, 1, 2),
        (2, 1, 2, 1),
        (2, 2, 2, 2),
        (2, 3, 2, 2),
        (2, 2, 3, 2),
        (3, 2, 2, 2),
    ], device=device)

    L = 25
    result = collect_puzzles(tiles, L, L, 500)
    if result is not None:
        print(f"Collected {result.size(0)} puzzles, shape {result.shape}")
        for puzzle in result:
            if not is_valid_puzzle(puzzle):
                print("Invalid puzzle!")
                exit(1)
        print("And they're all valid!")
