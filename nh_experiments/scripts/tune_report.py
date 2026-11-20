#!/usr/bin/env python
"""Compact table of finished tuning runs (tuning split: evaluated on WY1991-95), from test_metrics.csv only.

  python nh_experiments/scripts/tune_report.py [run_glob ...]   (default: /scratch2/mg963/convcnp_runs/tune_*)
"""
import glob
import sys

import pandas as pd

patterns = sys.argv[1:] or ["/scratch2/mg963/convcnp_runs/tune_*"]
rows = {}
for pat in patterns:
    for run in sorted(glob.glob(pat)):
        files = sorted(glob.glob(f"{run}/test/model_epoch*/test_metrics.csv"))
        if not files:
            continue
        df = pd.read_csv(files[-1], dtype={"basin": str}).set_index("basin")
        name = run.rstrip("/").split("/")[-1]
        name = name.replace("tune_", "").rsplit("_seed", 1)[0]
        rows[name] = {"n": len(df), "NSE_med": df.NSE.median(), "NSE_mean": df.NSE.mean(),
                      "NSE<0": int((df.NSE < 0).sum()), "KGE_med": df.KGE.median(),
                      "Alpha": df["Alpha-NSE"].median(), "FHV": df.FHV.median(), "FLV": df.FLV.median(),
                      "FMS": df.FMS.median()}
if not rows:
    sys.exit("no finished runs")
t = pd.DataFrame(rows).T.sort_values("NSE_med", ascending=False)
print(t.to_string(float_format=lambda v: f"{v:.3f}"))
