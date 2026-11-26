#!/usr/bin/env python
"""Study B: filling gaps in observed streamflow, Kratzert test period (WY1990-99), 531 CAMELS-US basins.

Gap classes (``--lengths``, default 7 / 30 / 90 days): per basin and class, non-overlapping gaps of exactly that
length are placed at random until they cover ~15 % of test days. Each gap has >= 7 observed days on both sides
(``MARGIN``, also needed by the bias-corrected baseline) and >= 50 % observed days inside. Masks are fixed per
(basin, length, mask_seed) via crc32, identical for every method. Scores use gap days with observations only.

  predict  python eval_gapfill.py predict RUN_DIR [RUN_DIR ...] [--gpu 0]
           ConvCNP runs with observed Q as an input (autoregressive_inputs: QObs(mm/d)_shiftS). One full-series pass
           per basin and gap class with Q hidden on the gap days (the shifted column hidden S days later), so a
           non-causal gap-filling model (S = 0) sees Q on both sides of the gap and a causal AR model (S = 1) only
           the past. Writes OUT/<run name>.npz (simulations on the test dates, physical units).
  report   python eval_gapfill.py report
           Methods: linear and log-linear interpolation across the gap; simulation without Q (LSTM-IR and gss-noQ,
           3-seed ensembles from their saved clean test results); LSTM-IR simulation bias-corrected with the
           observed/simulated volume ratio of the MARGIN days on each side, interpolated linearly across the gap;
           the predicted ConvCNP runs (3-seed ensembles and single seeds). Metrics on gap days per basin and class:
           NSE, KGE, RMSE, median absolute volume error per gap. Paired win rates / Wilcoxon vs the gap-filling ConvCNP.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

R = Path("/scratch2/mg963/convcnp_runs")
OUT = R / "gapfill"
BASIN_FILE = HERE.parent / "basins" / "531_basin_list.txt"
TEST_DATES = pd.date_range("1989-10-01", "1999-09-30", freq="D")
LENGTHS = [7, 30, 90]
COVER = 0.15
MARGIN = 7
TARGET = "QObs(mm/d)"


# ---------------------------------------------------------------------------------------------------------------
def gap_starts(obs: np.ndarray, basin: str, length: int, mask_seed: int = 0, cover: float = COVER,
               margin: int = MARGIN) -> np.ndarray:
    """Start indices (into ``obs``) of non-overlapping gaps of ``length`` days covering ~``cover`` of the series."""
    n = len(obs)
    ok = np.isfinite(obs)
    rng = np.random.default_rng(zlib.crc32(f"{mask_seed}|{basin}|gap{length}".encode()))
    n_target = int(round(cover * n / length))
    blocked = np.zeros(n, dtype=bool)      # gap days plus their margins
    starts = []
    for s in rng.permutation(np.arange(margin, n - length - margin + 1)):
        if len(starts) >= n_target:
            break
        lo, hi = s - margin, s + length + margin
        if blocked[lo:hi].any():
            continue
        if not (ok[lo:s].all() and ok[s + length:hi].all()) or ok[s:s + length].mean() < 0.5:
            continue
        starts.append(s)
        blocked[lo:hi] = True
    return np.sort(np.asarray(starts, dtype=int))


def gap_mask(n: int, starts: np.ndarray, length: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s in starts:
        m[s:s + length] = True
    return m


def hide(q: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = q.astype(float).copy()
    out[mask] = np.nan
    return out


def interp(q_hidden: np.ndarray, log: bool = False, eps: float = 0.01) -> np.ndarray:
    s = pd.Series(np.log(q_hidden + eps) if log else q_hidden)
    s = s.interpolate(method="linear", limit_area="inside").to_numpy()
    return np.exp(s) - eps if log else s


def bias_correct(sim: np.ndarray, obs: np.ndarray, starts: np.ndarray, length: int, margin: int = MARGIN,
                 eps: float = 1e-3) -> np.ndarray:
    """Rescale ``sim`` inside each gap by the obs/sim volume ratio of the margins, linearly interpolated."""
    out = sim.copy()
    for s in starts:
        e = s + length
        rb = (np.nansum(obs[s - margin:s]) + eps) / (np.sum(sim[s - margin:s]) + eps)
        ra = (np.nansum(obs[e:e + margin]) + eps) / (np.sum(sim[e:e + margin]) + eps)
        w = (np.arange(length) + 0.5) / length
        out[s:e] = sim[s:e] * ((1 - w) * rb + w * ra)
    return out


def gap_metrics(sim: np.ndarray, obs: np.ndarray, starts: np.ndarray, length: int) -> dict:
    m = gap_mask(len(obs), starts, length) & np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[m], sim[m]
    if len(o) < 10 or np.var(o) == 0:
        return dict(NSE=np.nan, KGE=np.nan, RMSE=np.nan, VolErr=np.nan, n_days=int(len(o)))
    nse = 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)
    r = np.corrcoef(o, s)[0, 1] if np.std(s) > 0 else 0.0
    kge = 1 - np.sqrt((r - 1) ** 2 + (np.std(s) / np.std(o) - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)
    vol = []
    for st in starts:
        oo, ss = obs[st:st + length], sim[st:st + length]
        if np.isfinite(oo).all() and oo.sum() > 0:
            vol.append(abs(ss.sum() - oo.sum()) / oo.sum())
    return dict(NSE=nse, KGE=kge, RMSE=float(np.sqrt(np.mean((s - o) ** 2))),
                VolErr=float(np.median(vol)) if vol else np.nan, n_days=int(len(o)))


# ---------------------------------------------------------------------------------------------------------------
def predict(run_dirs, gpu: int, lengths, mask_seed: int, basin_file: str | None, out: Path):
    from convcnp_nh import register
    register()
    import torch
    from neuralhydrology.evaluation import get_tester
    from neuralhydrology.utils.config import Config

    out.mkdir(parents=True, exist_ok=True)
    for rd in map(Path, run_dirs):
        dest = out / f"{rd.name}.npz"
        if dest.exists():
            print(f"[gapfill] skip {rd.name} (done)", flush=True)
            continue
        cfg = Config(rd / "config.yml")
        (qkey,) = cfg.autoregressive_inputs
        shift = int(re.search(r"_shift(\d+)$", qkey).group(1))
        upd = {"device": "cpu" if gpu < 0 else f"cuda:{gpu}", "random_holdout_from_dynamic_features": {}}
        if basin_file:
            upd["test_basin_file"] = basin_file
        cfg.update_config(upd)
        tester = get_tester(cfg=cfg, run_dir=rd, period="test", init_model=True)
        tester._load_weights(epoch=None)
        model = tester.model.eval()
        scale = float(tester.scaler["xarray_feature_scale"][TARGET].values)
        center = float(tester.scaler["xarray_feature_center"][TARGET].values)
        basins = [b.strip() for b in open(cfg.test_basin_file) if b.strip()]
        res = {}
        for basin in basins:
            ds = tester._get_dataset(basin)
            dates = pd.DatetimeIndex(np.asarray(ds._dates[basin]["1D"]))
            idx = dates.get_indexer(TEST_DATES)
            if (idx < 0).any():
                raise RuntimeError(f"{basin}: test dates missing from the dataset")
            x_d = {k: v[None].to(tester.device) for k, v in ds._x_d[basin]["1D"].items()}
            obs = ds._y[basin]["1D"][idx, 0].numpy() * scale + center
            data_base = {}
            if ds._attributes:
                data_base["x_s"] = ds._attributes[basin][None].to(tester.device)
            if ds.id_to_int:
                data_base["x_one_hot"] = torch.nn.functional.one_hot(
                    torch.tensor([ds.id_to_int[basin]]), num_classes=len(ds.id_to_int)).float().to(tester.device)
            res[f"{basin}|obs"] = obs.astype(np.float32)
            for L in lengths:
                starts = gap_starts(obs, basin, L, mask_seed)
                hid = idx[gap_mask(len(obs), starts, L)] + shift       # dataset positions of the hidden Q inputs
                hid = hid[hid < len(dates)]
                xq = x_d[qkey].clone()
                xq[0, hid] = float("nan")
                data = dict(data_base, x_d=dict(x_d, **{qkey: xq}))
                with torch.no_grad():
                    y_hat = model(model.pre_model_hook(data, is_train=False))["y_hat"][0, :, 0].cpu().numpy()
                res[f"{basin}|{L}"] = (y_hat[idx] * scale + center).astype(np.float32)
        np.savez_compressed(dest, **res)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[gapfill] predicted {rd.name}: {len(basins)} basins, lengths {list(lengths)}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
def _run_dirs(name):
    return [Path(sorted(glob.glob(f"{R}/{name}_seed{k}_*"))[-1]) for k in (1, 2, 3)]


def _saved_sims(name):
    """Clean-test simulations of a run without Q input (from eval_missing_scenarios.py), per seed."""
    from summarize import load_run, _point_sim
    sims = []
    for rd in _run_dirs(name):
        results = load_run(R / "missing_scenarios" / rd.name / "clean", "test")[0]
        per = {}
        for b, r in results.items():
            xr = r["1D"]["xr"]
            s = _point_sim(xr, TARGET).to_series()
            per[b] = s.reindex(TEST_DATES).to_numpy(dtype=float)
        sims.append(per)
    return sims


def report(lengths, mask_seed: int, out: Path):
    from scipy.stats import wilcoxon
    gap_runs = {"gss-gapfill": "convcnp_gss_gapfill", "gss-AR50": "convcnp_gss_ar50"}
    pred = {m: [np.load(out / f"{rd.name}.npz") for rd in _run_dirs(n)] for m, n in gap_runs.items()}
    sim = {"LSTM-IR sim": _saved_sims("lstm_ir_multi"), "gss-noQ sim": _saved_sims("convcnp_gss_multi")}
    basins = [b.strip() for b in open(BASIN_FILE) if b.strip()]
    rows = []
    for basin in basins:
        obs = pred["gss-gapfill"][0][f"{basin}|obs"].astype(float)
        for L in lengths:
            starts = gap_starts(obs, basin, L, mask_seed)
            if len(starts) == 0:
                continue
            qh = hide(obs, gap_mask(len(obs), starts, L))
            methods = {"linear interp": interp(qh), "log-linear interp": interp(qh, log=True)}
            for m, seeds in sim.items():
                methods[m] = np.mean([s[basin] for s in seeds], axis=0)
            methods["LSTM-IR sim + bias corr."] = bias_correct(methods["LSTM-IR sim"], obs, starts, L)
            for m, seeds in pred.items():
                arr = [s[f"{basin}|{L}"].astype(float) for s in seeds]
                methods[m] = np.mean(arr, axis=0)
                for k, a in enumerate(arr, 1):
                    methods[f"{m} s{k}"] = a
            # the LSTM baseline's rescaling, applied identically to the ConvCNPs (ensemble simulations)
            methods["gss-gapfill + bias corr."] = bias_correct(methods["gss-gapfill"], obs, starts, L)
            methods["gss-noQ sim + bias corr."] = bias_correct(methods["gss-noQ sim"], obs, starts, L)
            for m, s in methods.items():
                rows.append(dict(basin=basin, length=L, method=m, **gap_metrics(s, obs, starts, L)))
    df = pd.DataFrame(rows)
    df.to_csv(R / "gapfill_basin_metrics.csv", index=False)
    med = df.pivot_table(index="method", columns="length", values=["NSE", "KGE", "RMSE", "VolErr"], aggfunc="median")
    seeds = df[df.method.str.contains(r" s\d$")].copy()
    seeds["model"] = seeds.method.str.replace(r" s\d$", "", regex=True)
    seedmean = (seeds.groupby(["model", "method", "length"]).NSE.median().groupby(["model", "length"]).mean()
                .unstack("length"))
    ref = "gss-gapfill"
    bc = " + bias corr."
    pairs = [(ref, m) for m in df.method.unique() if m != ref and not re.search(r" s\d$", m)]
    pairs += [(f"gss-gapfill{bc}", f"LSTM-IR sim{bc}"), (f"gss-noQ sim{bc}", f"LSTM-IR sim{bc}"),
              (f"gss-gapfill{bc}", f"gss-noQ sim{bc}"), (f"gss-gapfill{bc}", "LSTM-IR sim"),
              # effect of the rescaling itself, per model
              (f"LSTM-IR sim{bc}", "LSTM-IR sim"), (f"gss-gapfill{bc}", "gss-gapfill"), (f"gss-noQ sim{bc}", "gss-noQ sim")]
    comp = []
    for L in lengths:
        sub = df[df.length == L]
        for x, y in pairs:
            for metric in ("NSE", "KGE"):
                a = sub[sub.method == x].set_index("basin")[metric]
                b = sub[sub.method == y].set_index("basin")[metric]
                d = (a - b).dropna()
                comp.append(dict(length=L, metric=metric, a=x, b=y, n=len(d), win_rate=(d > 0).mean(),
                                 median_diff=d.median(),
                                 wilcoxon_p=wilcoxon(d).pvalue if (d != 0).any() else np.nan))
    comp = pd.DataFrame(comp)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print("median over basins, metrics on gap days (ensembles = 3-seed mean simulation)")
        print(med.drop(index=[m for m in med.index if re.search(r" s\d$", m)]))
        print("\nmedian NSE, mean of single seeds\n", seedmean)
        print("\n", comp.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    med.to_csv(R / "gapfill_median.csv")
    seedmean.to_csv(R / "gapfill_seedmean_median_nse.csv")
    comp.to_csv(R / "gapfill_paired.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["predict", "report"])
    ap.add_argument("run_dir", nargs="*")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    ap.add_argument("--mask-seed", type=int, default=0)
    ap.add_argument("--basin-file", default=None, help="override test_basin_file (smoke tests)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.stage == "predict":
        predict(args.run_dir, args.gpu, args.lengths, args.mask_seed, args.basin_file, Path(args.out))
    else:
        report(args.lengths, args.mask_seed, Path(args.out))


if __name__ == "__main__":
    main()
