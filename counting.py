#!/usr/bin/env python3
import numpy as np

# NESW
N, E, S, W = 0, 1, 2, 3

# Compute all valid columns. Complexity here is |tiles|^L
def valid_to_place_below(tile1, tile2):
    return tile1[S] == tile2[N]

# This is explicit enumeration
def valid_columns(tiles, L):
    stack = [[t] for t in tiles]
    cols = []

    while stack:
        candidate = stack.pop()
        # CASE 1: This is a valid column
        if len(candidate) == L:
            cols.append(candidate)

        # CASE 2: It's not, so we try to expand
        else:
            bottom = candidate[-1]
            for tile in tiles:
                if valid_to_place_below(bottom, tile):
                    stack.append(candidate + [tile])
    return cols

def can_place_col_right(col1, col2):
    for i in range(len(col1)):
        if col1[i][E] != col2[i][W]:
            return 0

    return 1

def transfer_matrix(columns):
    T = np.zeros((len(columns), len(columns)), dtype=int)

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            T[i,j] = can_place_col_right(col1, col2)

    return T

if __name__ == "__main__":
    # Just some random tile colors
    tiles = [
        (1,1,1,1),
        (1,2,1,2),
        (2,1,2,1),
        (2,2,3,2),
    ]

    # Length of the square
    L = 4

    columns = valid_columns(tiles, L)
    print(len(columns))

    T = transfer_matrix(columns)
    count = np.linalg.matrix_power(T, L-1).sum()
    print(count)
