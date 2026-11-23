#!/usr/bin/env python
"""Evaluate trained runs with a fraction of forcing days hidden (evaluation only, no retraining).

  python nh_experiments/scripts/eval_missing_forcings.py RUN_DIR [RUN_DIR ...] --fill nan|ffill \
      [--fracs 0.1 0.3 0.5] [--mask-seed 0] [--gpu 0]

For every basin, each day is hidden independently with probability ``frac``: all dynamic inputs
(forcings) of that day are set to NaN in the raw CAMELS-US data, before normalisation. Discharge
and static attributes are untouched. The mask depends only on (mask_seed, basin, frac), so every
model sees exactly the same hidden days.

``--fill nan``   keep the NaNs: the ConvCNP's density channel treats them as missing (native).
``--fill ffill`` fill each gap with the last observed value (back-fill at the series start). The
                 LSTM needs this (a NaN input gives a NaN prediction); it is also run for the
                 ConvCNP to separate the effect of the density channel from that of the fill.

The original run directory is never written to. Each condition is evaluated in
``<out>/<run name>/<fill>_p<frac>/``, which symlinks the run's config, scaler and final
checkpoint; results land in its ``test/model_epochNNN/``.
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from convcnp_nh import register  # noqa: E402

register()

import neuralhydrology.datasetzoo.camelsus as camelsus  # noqa: E402
from neuralhydrology.evaluation.evaluate import start_evaluation  # noqa: E402
from neuralhydrology.utils.config import Config  # noqa: E402

OUT = Path("/scratch2/mg963/convcnp_runs/missing_forcing")
STATE = {"frac": 0.0, "fill": "nan", "mask_seed": 0}


def hide_days(df, basin: str, frac: float, fill: str, mask_seed: int, target_cols=None):
    """Return a copy of ``df`` with ``frac`` of the days' forcings hidden (NaN or forward-filled)."""
    if frac <= 0:
        return df
    target_cols = target_cols or [c for c in df.columns if "qobs" in c.lower()]
    cols = [c for c in df.columns if c not in target_cols]
    rng = np.random.default_rng(zlib.crc32(f"{mask_seed}|{basin}|{frac:.4f}".encode()))
    hidden = rng.random(len(df)) < frac
    df = df.copy()
    df.loc[hidden, cols] = np.nan
    if fill == "ffill":
        df[cols] = df[cols].ffill().bfill()
    elif fill != "nan":
        raise ValueError(fill)
    return df


def _patch_loader():
    original = camelsus.CamelsUS._load_basin_data

    def _load_basin_data(self, basin):
        df = original(self, basin)
        return hide_days(df, basin, STATE["frac"], STATE["fill"], STATE["mask_seed"])

    camelsus.CamelsUS._load_basin_data = _load_basin_data


def condition_dir(run_dir: Path, fill: str, frac: float, out: Path = OUT) -> Path:
    d = out / run_dir.name / f"{fill}_p{int(round(frac * 100)):02d}"
    d.mkdir(parents=True, exist_ok=True)
    ckpt = sorted(run_dir.glob("model_epoch*.pt"))[-1]
    for src in (run_dir / "config.yml", run_dir / "train_data", ckpt):
        link = d / src.name
        if not link.exists():
            link.symlink_to(src)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="+")
    ap.add_argument("--fill", choices=["nan", "ffill"], required=True)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    ap.add_argument("--mask-seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=str(OUT), help="root for the condition directories")
    ap.add_argument("--basin-file", default=None, help="override test_basin_file (smoke tests)")
    args = ap.parse_args()
    _patch_loader()
    STATE["fill"], STATE["mask_seed"] = args.fill, args.mask_seed
    for rd in map(Path, args.run_dir):
        for frac in args.fracs:
            d = condition_dir(rd, args.fill, frac, Path(args.out))
            if list(d.glob("test/model_epoch*/test_metrics.csv")):
                print(f"[missing] skip {d} (done)", flush=True)
                continue
            STATE["frac"] = frac
            cfg = Config(d / "config.yml")
            cfg.update_config({"device": "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"})
            if args.basin_file:
                cfg.update_config({"test_basin_file": args.basin_file})
            start_evaluation(cfg=cfg, run_dir=d, epoch=None, period="test")
            print(f"[missing] evaluated {d}", flush=True)


if __name__ == "__main__":
    main()
