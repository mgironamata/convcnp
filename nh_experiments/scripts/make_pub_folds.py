#!/usr/bin/env python
"""Random k-fold split of the 531 basins for prediction in ungauged basins (PUB).

Writes basins/pub_k{K}_fold{i}_{train,test}.txt (i = 1..K). Every basin is in exactly one test fold.
Deterministic (numpy seed), so the split is reproducible.

  python nh_experiments/scripts/make_pub_folds.py [--k 5] [--seed 0]
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    basins = sorted(b.strip() for b in (ROOT / "basins" / "531_basin_list.txt").read_text().split() if b.strip())
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(basins))
    folds = np.array_split(perm, args.k)
    for i, test_idx in enumerate(folds, start=1):
        test = sorted(basins[j] for j in test_idx)
        train = sorted(set(basins) - set(test))
        (ROOT / "basins" / f"pub_k{args.k}_fold{i}_test.txt").write_text("\n".join(test) + "\n")
        (ROOT / "basins" / f"pub_k{args.k}_fold{i}_train.txt").write_text("\n".join(train) + "\n")
        print(f"fold {i}: {len(train)} train / {len(test)} test basins")


if __name__ == "__main__":
    main()
