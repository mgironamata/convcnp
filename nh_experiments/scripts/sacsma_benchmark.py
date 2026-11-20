#!/usr/bin/env python
"""Per-basin metrics of the CAMELS SAC-SMA benchmark (Newman et al. 2015), scored exactly like
the neuralhydrology runs: same observations (USGS QObs converted to mm/d with neuralhydrology's
loader), same period, same metric functions.

  python nh_experiments/scripts/sacsma_benchmark.py \
      --model-output /scratch2/mg963/data/camels_us/unpacked/modelOutput_daymet \
      --data-dir /scratch2/mg963/data/camels_us/CAMELS_US \
      --start 1995-10-01 --end 2010-09-30 --seeds mean \
      --out /scratch2/mg963/convcnp_runs/sacsma_daymet_test_mres.csv

CAMELS v1.2 model output: one file per basin and calibration seed, `<basin>_<seed>_model_output.txt`,
whitespace separated with columns YR MNTH DY HR SWE PRCP RAIM TAIR PET ET MOD_RUN OBS_RUN (mm/d).
To my knowledge the SAC-SMA runs were calibrated per basin on the first 15 water years
(WY1981-1995, i.e. the "mres" training period) -- VERIFY this in raw/readme.txt before quoting it.
If so, on the mres test period this is a fair *gauged* (not PUB) benchmark. `--seeds mean` averages MOD_RUN over the 10 calibration seeds;
`--seeds 05` uses one seed (what the MRes notebook used).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from neuralhydrology.datasetzoo.camelsus import load_camels_us_discharge, load_camels_us_forcings
from neuralhydrology.evaluation.metrics import calculate_metrics

METRICS = ["NSE", "KGE", "Alpha-NSE", "Beta-NSE", "Pearson-r", "FHV", "FLV", "FMS"]


def read_model_output(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep=r"\s+")
    dates = pd.to_datetime(dict(year=df["YR"], month=df["MNTH"], day=df["DY"]))
    return pd.Series(df["MOD_RUN"].values, index=pd.DatetimeIndex(dates, name="date"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-output", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--basins", type=Path, default=Path(__file__).resolve().parents[1] / "basins" / "531_basin_list.txt")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--seeds", default="mean", help="'mean' (all seeds averaged) or a seed id like 05")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # The CAMELS v1.2 zip contains copies of many basins' files under *other* HUC folders whose
    # MOD_RUN differs slightly; use the copy in the basin's own HUC folder (huc_02 in camels_name).
    names = pd.read_csv(args.data_dir / "camels_attributes_v2.0" / "camels_name.txt", sep=";",
                        dtype={"gauge_id": str})
    own_huc = {g: f"{int(h):02d}" for g, h in zip(names["gauge_id"], names["huc_02"])}
    files = [f for f in args.model_output.rglob("*_model_output.txt") if "__MACOSX" not in f.parts]
    by_basin: dict[str, dict[str, Path]] = {}
    for f in files:
        m = re.match(r"(\d{8})_(\d+)_model_output\.txt", f.name)
        if m and f.parent.name == own_huc.get(m.group(1)):
            by_basin.setdefault(m.group(1), {})[m.group(2)] = f
    basins = [b.strip() for b in args.basins.read_text().split() if b.strip()]
    print(f"{len(files)} model-output files, {len(by_basin)} basins; evaluating {len(basins)} basins")

    rows = {}
    period = slice(pd.Timestamp(args.start), pd.Timestamp(args.end))
    for basin in basins:
        seeds = by_basin.get(basin)
        if not seeds:
            print(f"  ! no SAC-SMA output for {basin}")
            continue
        if args.seeds == "mean":
            sims = [read_model_output(p) for p in seeds.values()]
            sim = pd.concat(sims, axis=1).mean(axis=1)
        else:
            if args.seeds not in seeds:
                print(f"  ! seed {args.seeds} missing for {basin}")
                continue
            sim = read_model_output(seeds[args.seeds])
        _, area = load_camels_us_forcings(args.data_dir, basin, "daymet")
        obs = load_camels_us_discharge(args.data_dir, basin, area)
        obs[obs < 0] = np.nan
        df = pd.DataFrame({"obs": obs, "sim": sim}).loc[period]
        o = xr.DataArray(df["obs"].values, coords={"date": df.index}, dims="date")
        s = xr.DataArray(df["sim"].values, coords={"date": df.index}, dims="date")
        try:
            rows[basin] = calculate_metrics(o, s, metrics=METRICS, resolution="1D")
        except Exception as exc:
            print(f"  ! {basin}: {exc}")
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "basin"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out)
    print(f"SAC-SMA ({args.seeds}) {args.start}..{args.end}: n={len(out)}, median NSE={out['NSE'].median():.3f}, "
          f"mean NSE={out['NSE'].mean():.3f}\nwritten {args.out}")


if __name__ == "__main__":
    main()
