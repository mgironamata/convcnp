#!/usr/bin/env python
"""Evaluate trained multi-forcing runs under missing-input scenarios (Gauch et al. 2025 style).

  python nh_experiments/scripts/eval_missing_scenarios.py RUN_DIR [RUN_DIR ...] [--scenarios ...] [--gpu 0]

Feature groups are the run's ``dynamic_inputs`` groups (one per forcing product: Daymet, Maurer, NLDAS).
Masks are fixed per (basin, day, group), drawn from ``mask_seed``, and applied to the raw data before
normalisation, so every model sees exactly the same missing days. Missing values stay NaN: the LSTM's
``nan_handling_method`` (masked mean / input replacing) and the ConvCNP's density channel handle them.

Scenarios (``SCENARIOS``):
  clean              all inputs present
  indep_pXX          each product is missing on a day independently with probability XX%
                     (Gauch et al. 2025, experiment 1, but with a fixed mask instead of a fresh draw per window)
  allday_pXX         all products missing together on XX% of days (no forcing at all on those days)
  gap_pXX            all products missing together in contiguous gaps covering XX% of days on average; gap
                     lengths are geometric with mean GAP_MEAN_DAYS (two-state Markov chain, stationary start)
  keep_<products>    the other products are missing for the whole period (experiment 2)

ConvCNP runs are evaluated in one causal pass over each basin's full series (``--fast-convcnp``, default
on). Every layer is causal and per-time-step, so this gives the same last-step predictions as
neuralhydrology's one-window-per-day tester (checked by ``--check``), about 100x faster.

Results go to ``<out>/<run name>/<scenario>/test/model_epochNNN/`` (the run directory is never written to).
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

import torch  # noqa: E402
import neuralhydrology.datasetzoo.camelsus as camelsus  # noqa: E402
import neuralhydrology.evaluation.tester as nh_tester  # noqa: E402
from neuralhydrology.evaluation.evaluate import start_evaluation  # noqa: E402
from neuralhydrology.utils.config import Config  # noqa: E402

OUT = Path("/scratch2/mg963/convcnp_runs/missing_scenarios")
PRODUCTS = ("daymet", "maurer", "nldas")
SCENARIOS = (["clean"] + [f"indep_p{p}" for p in (10, 30, 50)] + [f"allday_p{p}" for p in (10, 30, 50)]
             + ["keep_daymet", "keep_maurer", "keep_nldas",
                "keep_daymet_maurer", "keep_daymet_nldas", "keep_maurer_nldas"])
# single-product runs (one feature group): indep == allday, so random days vs contiguous gaps
SCENARIOS_SINGLE = ["clean"] + [f"indep_p{p}" for p in (10, 30, 50)] + [f"gap_p{p}" for p in (10, 30, 50)]
GAP_MEAN_DAYS = 10
STATE = {"scenario": "clean", "groups": None, "mask_seed": 0}


def product_of(column: str) -> str | None:
    """'prcp(mm/day)_daymet' -> 'daymet', 'PRCP(mm/day)_nldas_extended' -> 'nldas'; None if no suffix."""
    for p in PRODUCTS:
        if f")_{p}" in column:
            return p
    return None


def group_masks(n_days: int, basin: str, scenario: str, n_groups: int, group_products: list[str],
                mask_seed: int = 0) -> np.ndarray:
    """Boolean [n_groups, n_days] array, True = missing."""
    hidden = np.zeros((n_groups, n_days), dtype=bool)
    if scenario == "clean":
        return hidden
    rng = np.random.default_rng(zlib.crc32(f"{mask_seed}|{basin}|{scenario}".encode()))
    kind, _, arg = scenario.partition("_")
    if kind == "indep":
        p = int(arg[1:]) / 100
        for g in range(n_groups):
            hidden[g] = rng.random(n_days) < p
    elif kind == "allday":
        p = int(arg[1:]) / 100
        hidden[:] = rng.random(n_days) < p
    elif kind == "gap":
        p, L = int(arg[1:]) / 100, GAP_MEAN_DAYS
        p_start, p_end = p / (L * (1 - p)), 1 / L
        u = rng.random(n_days)
        miss = np.empty(n_days, dtype=bool)
        state = u[0] < p
        for t in range(n_days):
            if t:
                state = (u[t] >= p_end) if state else (u[t] < p_start)
            miss[t] = state
        hidden[:] = miss
    elif kind == "keep":
        keep = set(arg.split("_"))
        if not keep <= set(PRODUCTS) or None in group_products:
            raise ValueError(f"{scenario} needs product-suffixed feature groups, got {group_products}")
        for g, prod in enumerate(group_products):
            hidden[g] = prod not in keep
    else:
        raise ValueError(scenario)
    return hidden


def apply_scenario(df, basin: str, scenario: str, groups: list[list[str]], mask_seed: int = 0):
    if scenario == "clean":
        return df
    prods = [product_of(g[0]) for g in groups]
    hidden = group_masks(len(df), basin, scenario, len(groups), prods, mask_seed)
    df = df.copy()
    for g, cols in enumerate(groups):
        df.loc[hidden[g], cols] = np.nan
    return df


def _patch_loader():
    original = camelsus.CamelsUS._load_basin_data

    def _load_basin_data(self, basin):
        df = original(self, basin)
        return apply_scenario(df, basin, STATE["scenario"], STATE["groups"], STATE["mask_seed"])

    camelsus.CamelsUS._load_basin_data = _load_basin_data


def _fast_convcnp_evaluate(self, model, loader, frequencies, save_all_output=False):
    """Drop-in for BaseTester._evaluate: one causal pass over the basin's whole series."""
    if len(frequencies) != 1 or self.cfg.predict_last_n != 1:
        raise RuntimeError("fast ConvCNP evaluation needs a single frequency and predict_last_n = 1")
    ds, freq = loader.dataset, frequencies[0]
    basin = ds.lookup_table[0][0]
    ends = torch.tensor([ds.lookup_table[i][1][0] for i in range(len(ds))])
    data = {"x_d": {k: v[None].to(self.device) for k, v in ds._x_d[basin][freq].items()}}
    if ds._attributes:
        data["x_s"] = ds._attributes[basin][None].to(self.device)
    if ds.id_to_int:
        data["x_one_hot"] = torch.nn.functional.one_hot(torch.tensor([ds.id_to_int[basin]]),
                                                        num_classes=len(ds.id_to_int)).float().to(self.device)
    with torch.no_grad():
        y_hat = model(model.pre_model_hook(data, is_train=False))["y_hat"][0].cpu()     # [T, n_targets]
    preds = {freq: y_hat[ends][:, None, :].numpy()}
    obs = {freq: ds._y[basin][freq][ends][:, None, :].numpy()}
    dates = {freq: np.asarray(ds._dates[basin][freq])[ends.numpy()][:, None]}
    return preds, obs, dates, {"loss": np.nan}, None


def condition_dir(run_dir: Path, scenario: str, out: Path) -> Path:
    d = out / run_dir.name / scenario
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
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--mask-seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=2048, help="evaluation batch size (LSTM)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--basin-file", default=None, help="override test_basin_file (smoke tests)")
    ap.add_argument("--no-fast-convcnp", action="store_true", help="use neuralhydrology's per-window tester")
    args = ap.parse_args()
    _patch_loader()
    original_evaluate = nh_tester.BaseTester._evaluate
    STATE["mask_seed"] = args.mask_seed
    for rd in map(Path, args.run_dir):
        for scenario in args.scenarios:
            d = condition_dir(rd, scenario, Path(args.out))
            if list(d.glob("test/model_epoch*/test_metrics.csv")):
                print(f"[scenarios] skip {d} (done)", flush=True)
                continue
            cfg = Config(d / "config.yml")
            groups = cfg.dynamic_inputs if isinstance(cfg.dynamic_inputs[0], list) else [cfg.dynamic_inputs]
            STATE["scenario"], STATE["groups"] = scenario, groups
            upd = {"device": "cpu" if args.gpu < 0 else f"cuda:{args.gpu}", "batch_size": args.batch_size}
            if args.basin_file:
                upd["test_basin_file"] = args.basin_file
            cfg.update_config(upd)
            fast = cfg.model.lower() == "convcnp" and not args.no_fast_convcnp
            nh_tester.BaseTester._evaluate = _fast_convcnp_evaluate if fast else original_evaluate
            start_evaluation(cfg=cfg, run_dir=d, epoch=None, period="test")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()     # release cached blocks so a shared GPU sees the real footprint
            print(f"[scenarios] evaluated {d}{' (fast)' if fast else ''}", flush=True)


if __name__ == "__main__":
    main()
