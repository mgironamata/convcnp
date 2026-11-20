#!/usr/bin/env python
"""Summarise neuralhydrology test results: per run, and as seed ensembles.

  python nh_experiments/scripts/summarize.py /scratch2/mg963/convcnp_runs/*mres_seed*  [--period test]
      [--benchmark sacsma_metrics.csv] [--out summary.csv]

* Per run: median / mean of every metric over basins (last evaluated epoch).
* Ensemble: runs whose experiment name only differs by `_seedN` are grouped; simulations are
  averaged over seeds and metrics recomputed with neuralhydrology's own metric functions
  (Kratzert et al. 2019 report ensemble means of 8 seeds).
* Only basins present in *all* compared runs are used, so numbers are comparable.
"""
from __future__ import annotations

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from neuralhydrology.evaluation.metrics import calculate_metrics

METRICS = ["NSE", "KGE", "Alpha-NSE", "Beta-NSE", "Pearson-r", "FHV", "FLV", "FMS"]


def _last_epoch_dir(run_dir: Path, period: str) -> Path:
    dirs = sorted((run_dir / period).glob("model_epoch*"))
    if not dirs:
        raise FileNotFoundError(f"No {period} results in {run_dir} (run `run.py eval`)")
    return dirs[-1]


def _experiment_name(run_dir: Path) -> str:
    # <experiment>_seed<k>_<ddmm>_<hhmmss>
    name = re.sub(r"_\d{4}_\d{6}$", "", run_dir.name)
    return re.sub(r"_seed\d+$", "", name)


def _point_sim(xr, var: str):
    sim = xr[f"{var}_sim"]
    if "samples" in sim.dims:          # probabilistic heads: use the predictive mean
        sim = sim.mean("samples")
    return sim.isel(time_step=-1)


def load_run(run_dir: Path, period: str):
    ep = _last_epoch_dir(run_dir, period)
    with open(ep / f"{period}_results.p", "rb") as fp:
        results = pickle.load(fp)
    metrics = pd.read_csv(ep / f"{period}_metrics.csv", dtype={"basin": str}).set_index("basin")
    return results, metrics, ep.name


def ensemble_metrics(runs: list[dict], basins: list[str], var: str = "QObs(mm/d)") -> pd.DataFrame:
    rows = {}
    for basin in basins:
        xrs = [r["results"][basin]["1D"]["xr"] for r in runs]
        obs = xrs[0][f"{var}_obs"].isel(time_step=-1)
        sim = sum(_point_sim(x, var) for x in xrs) / len(xrs)
        if np.isfinite(obs.values).sum() < 2:
            continue
        try:
            rows[basin] = calculate_metrics(obs, sim, metrics=METRICS, resolution="1D")
        except Exception as exc:  # e.g. all-zero flow for FLV
            rows[basin] = {m: np.nan for m in METRICS}
            print(f"  ! {basin}: {exc}")
    return pd.DataFrame.from_dict(rows, orient="index")


def describe(df: pd.DataFrame, label: str) -> dict:
    out = {"model": label, "n_basins": int(df["NSE"].notna().sum())}
    for m in [c for c in METRICS if c in df.columns]:
        out[f"{m}_median"] = float(np.nanmedian(df[m]))
        out[f"{m}_mean"] = float(np.nanmean(df[m]))
    out["NSE<0 (n)"] = int((df["NSE"] < 0).sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--period", default="test")
    ap.add_argument("--benchmark", type=Path, help="per-basin metrics CSV (e.g. from sacsma_benchmark.py)")
    ap.add_argument("--out", type=Path, help="write summary table (csv)")
    args = ap.parse_args()

    runs = []
    for rd in args.run_dirs:
        try:
            results, metrics, epoch = load_run(rd, args.period)
        except FileNotFoundError as exc:
            print(f"skip: {exc}")
            continue
        runs.append(dict(dir=rd, exp=_experiment_name(rd), results=results, metrics=metrics, epoch=epoch))
    if not runs:
        raise SystemExit("no runs with results")

    common = sorted(set.intersection(*(set(r["results"]) for r in runs)))
    print(f"{len(runs)} runs, {len(common)} basins common to all\n")

    rows = []
    for r in runs:
        rows.append(describe(r["metrics"].loc[[b for b in common if b in r["metrics"].index]],
                             f"{r['dir'].name} [{r['epoch']}]"))
    groups = defaultdict(list)
    for r in runs:
        groups[r["exp"]].append(r)
    for exp, members in groups.items():
        if len(members) > 1:
            ens = ensemble_metrics(members, common)
            rows.append(describe(ens, f"{exp} ENSEMBLE of {len(members)}"))
    if args.benchmark:
        bm = pd.read_csv(args.benchmark, dtype={"basin": str}).set_index("basin")
        rows.append(describe(bm.loc[[b for b in common if b in bm.index]], f"benchmark: {args.benchmark.name}"))

    table = pd.DataFrame(rows).set_index("model")
    cols = ["n_basins", "NSE_median", "NSE_mean", "NSE<0 (n)", "KGE_median", "Alpha-NSE_median",
            "Beta-NSE_median", "FHV_median", "FLV_median", "FMS_median"]
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 70, "display.float_format", "{:.3f}".format):
        print(table[[c for c in cols if c in table.columns]])
    if args.out:
        table.to_csv(args.out)
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
