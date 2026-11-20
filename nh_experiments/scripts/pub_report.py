#!/usr/bin/env python
"""Pool PUB (k-fold over basins) test results across folds: every basin scored once, out-of-sample.

  python nh_experiments/scripts/pub_report.py [--k 5] [--models convcnp_gs_pub lstm_pub] [--ensemble]

Single-seed rows use each run's test_metrics.csv. With --ensemble, simulations of all seeds of a fold
are averaged (test_results.p) and metrics recomputed with neuralhydrology's functions (slower).
Only models with all k folds finished are pooled; partial models are reported as such.
"""
import argparse
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import ensemble_metrics, load_run  # noqa: E402

RUNS = "/scratch2/mg963/convcnp_runs"


def describe(df, label):
    return {"model": label, "n_basins": int(df["NSE"].notna().sum()),
            "NSE_median": float(np.nanmedian(df["NSE"])), "NSE_mean": float(np.nanmean(df["NSE"])),
            "NSE<0": int((df["NSE"] < 0).sum()), "KGE_median": float(np.nanmedian(df["KGE"])),
            "Alpha_median": float(np.nanmedian(df["Alpha-NSE"])), "FHV_median": float(np.nanmedian(df["FHV"])),
            "FLV_median": float(np.nanmedian(df["FLV"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=["convcnp_gs_pub", "lstm_pub"])
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = []
    for model in args.models:
        per_seed = defaultdict(dict)            # seed -> fold -> run_dir
        for d in glob.glob(f"{RUNS}/{model}_fold*_seed*"):
            m = re.search(r"_fold(\d+)_seed(\d+)_", Path(d).name)
            if m and list(Path(d).glob("test/model_epoch*/test_metrics.csv")):
                per_seed[int(m.group(2))][int(m.group(1))] = Path(d)
        for seed in sorted(per_seed):
            folds = per_seed[seed]
            dfs = [pd.read_csv(sorted(folds[f].glob("test/model_epoch*/test_metrics.csv"))[-1],
                               dtype={"basin": str}).set_index("basin") for f in sorted(folds)]
            tag = f"{model} seed{seed}" + ("" if len(folds) == args.k else f" (folds {sorted(folds)} only)")
            rows.append(describe(pd.concat(dfs), tag))
        if args.ensemble:
            seeds = sorted(per_seed)
            complete = [f for f in range(1, args.k + 1) if all(f in per_seed[s] for s in seeds)]
            if len(seeds) > 1 and complete:
                parts = []
                for f in complete:
                    members = [dict(results=load_run(per_seed[s][f], "test")[0]) for s in seeds]
                    basins = sorted(set.intersection(*(set(m["results"]) for m in members)))
                    parts.append(ensemble_metrics(members, basins))
                tag = f"{model} ENSEMBLE of {len(seeds)}" + ("" if len(complete) == args.k else f" (folds {complete} only)")
                rows.append(describe(pd.concat(parts), tag))
    if not rows:
        sys.exit("no finished PUB runs yet")
    table = pd.DataFrame(rows).set_index("model")
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        print(table)
    if args.out:
        table.to_csv(args.out)


if __name__ == "__main__":
    main()
