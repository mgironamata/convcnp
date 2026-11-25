#!/usr/bin/env python
"""Ensemble and mixed-ensemble summaries from finished runs (CPU only).

  python nh_experiments/scripts/ensemble_report.py mres   # gauged MRes: LSTM 3/5/6 seeds, gs3, LSTM3+gs3
  python nh_experiments/scripts/ensemble_report.py pub    # PUB: LSTM3, gss3, LSTM3+gss3 (pooled over 5 folds)
  python nh_experiments/scripts/ensemble_report.py pilots # tuning-split pilots (batches L, M, N) vs gss seed 1
  python nh_experiments/scripts/ensemble_report.py gss6   # gauged MRes: gss 6 seeds vs LSTM3/LSTM5, gs3
  python nh_experiments/scripts/ensemble_report.py missing # forcing days hidden (eval_missing_forcings.py)
  python nh_experiments/scripts/ensemble_report.py scenarios # missing-input retraining (eval_missing_scenarios.py):
                                                             # gss / LSTM masked mean / LSTM input replacing, 3 seeds
  python nh_experiments/scripts/ensemble_report.py daymet  # Daymet-only missing data: gss vs LSTM input replacing

Ensembles average the members' simulated series and recompute the metrics (summarize.ensemble_metrics).
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import ensemble_metrics, load_run  # noqa: E402

R = "/scratch2/mg963/convcnp_runs"


def run_dir(name, seed):
    return Path(sorted(glob.glob(f"{R}/{name}_seed{seed}_*"))[-1])


def single(d):
    return pd.read_csv(sorted(d.glob("test/model_epoch*/test_metrics.csv"))[-1], dtype={"basin": str}).set_index("basin")


def ens(dirs):
    mem = [dict(results=load_run(d, "test")[0]) for d in dirs]
    basins = sorted(set.intersection(*(set(m["results"]) for m in mem)))
    return ensemble_metrics(mem, basins)


def describe(df):
    return dict(n=int(df.NSE.notna().sum()), NSE_median=df.NSE.median(), NSE_mean=df.NSE.mean(),
                NSE_lt0=int((df.NSE < 0).sum()), KGE_median=df.KGE.median(), Alpha_median=df["Alpha-NSE"].median(),
                FHV_median=df.FHV.median(), FLV_median=df.FLV.median())


def paired(a, b):
    d = (a.NSE - b.NSE).dropna()
    return dict(win_rate=(d > 0).mean(), median_diff=d.median(), wilcoxon_p=wilcoxon(d).pvalue if (d != 0).any() else np.nan)


def table(rows, comparisons, out):
    t = pd.DataFrame([dict(model=k, **describe(v)) for k, v in rows.items()]).set_index("model")
    c = pd.DataFrame([dict(a=a, b=b, **paired(rows[a], rows[b])) for a, b in comparisons])
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(t); print(c.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    t.to_csv(f"{R}/{out}_summary.csv"); c.to_csv(f"{R}/{out}_paired.csv", index=False)


def main(which):
    if which == "mres":
        L = lambda s: [run_dir("lstm_mres", k) for k in s]
        C = [run_dir("convcnp_gs_mres", k) for k in (1, 2, 3)]
        rows = {"LSTM3 (s1-3)": ens(L((1, 2, 3))), "LSTM5 (s1-5)": ens(L((1, 2, 3, 4, 5))),
                "LSTM6 (s1-6)": ens(L((1, 2, 3, 4, 5, 6))), "gs3": ens(C), "LSTM3+gs3": ens(L((1, 2, 3)) + C)}
        rows.update({f"LSTM s{k}": single(run_dir("lstm_mres", k)) for k in (4, 5, 6)})
        table(rows, [("LSTM3+gs3", "LSTM3 (s1-3)"), ("LSTM3+gs3", "LSTM5 (s1-5)"), ("LSTM3+gs3", "LSTM6 (s1-6)"),
                     ("LSTM6 (s1-6)", "LSTM3 (s1-3)"), ("gs3", "LSTM3 (s1-3)"), ("LSTM3+gs3", "gs3")], "mres_mixed_control")
    elif which == "pub":
        def pooled(models):
            return pd.concat([ens([run_dir(f"{m}_fold{f}", s) for m in models for s in (1, 2, 3)]) for f in range(1, 6)])
        rows = {"LSTM3": pooled(["lstm_pub"]), "gss3": pooled(["convcnp_gss_pub"]),
                "LSTM3+gss3": pooled(["lstm_pub", "convcnp_gss_pub"])}
        table(rows, [("gss3", "LSTM3"), ("LSTM3+gss3", "LSTM3"), ("LSTM3+gss3", "gss3")], "pub_mixed")
    elif which == "pilots":
        names = ["gss", "sharp", "lowdrop", "alpha03", "alpha10", "alpha01", "alpha01w", "peak05", "peak20",
                 "local2", "local2sharp"]
        rows = {"lstm_ref": single(run_dir("tune_lstm_ref", 1))}
        rows.update({n: single(run_dir(f"tune_L_{n}", 1)) for n in names})
        table(rows, [(n, "gss") for n in names if n != "gss"] + [("lstm_ref", "gss")], "pilots_LMN_seed1")
    elif which == "gss6":
        L = lambda s: [run_dir("lstm_mres", k) for k in s]
        G = lambda s: [run_dir("convcnp_gss_mres", k) for k in s]
        rows = {"LSTM3 (s1-3)": ens(L((1, 2, 3))), "LSTM5 (s1-5)": ens(L((1, 2, 3, 4, 5))),
                "gs3": ens([run_dir("convcnp_gs_mres", k) for k in (1, 2, 3)]), "gss3 (s1-3)": ens(G((1, 2, 3))),
                "gss6 (s1-6)": ens(G(range(1, 7))), "LSTM5+gss6": ens(L((1, 2, 3, 4, 5)) + G(range(1, 7)))}
        rows.update({f"gss s{k}": single(run_dir("convcnp_gss_mres", k)) for k in range(1, 7)})
        table(rows, [("gss6 (s1-6)", "LSTM3 (s1-3)"), ("gss6 (s1-6)", "LSTM5 (s1-5)"), ("gss3 (s1-3)", "gs3"),
                     ("gss6 (s1-6)", "gss3 (s1-3)"), ("LSTM5+gss6", "LSTM5 (s1-5)")], "mres_gss6")
    elif which == "missing":
        M = Path(R) / "missing_forcing"
        lstm = [run_dir("lstm_mres", k) for k in range(1, 6)]
        gss = [run_dir("convcnp_gss_mres", k) for k in range(1, 7)]
        cond = lambda dirs, fill, p: [M / d.name / f"{fill}_p{p:02d}" for d in dirs]
        rows, comps = {"LSTM5 clean": ens(lstm), "gss6 clean": ens(gss)}, [("gss6 clean", "LSTM5 clean")]
        for p in (10, 30, 50):
            rows[f"LSTM5 ffill p{p}"] = ens(cond(lstm, "ffill", p))
            rows[f"gss6 nan p{p}"] = ens(cond(gss, "nan", p))
            rows[f"gss6 ffill p{p}"] = ens(cond(gss, "ffill", p))
            comps += [(f"gss6 nan p{p}", f"LSTM5 ffill p{p}"), (f"gss6 ffill p{p}", f"LSTM5 ffill p{p}"),
                      (f"gss6 nan p{p}", f"gss6 ffill p{p}")]
        table(rows, comps, "missing_forcing")
    elif which in ("scenarios", "daymet"):
        S = Path(R) / "missing_scenarios"
        scen = (["clean", "indep_p10", "indep_p30", "indep_p50", "gap_p10", "gap_p30", "gap_p50"] if which == "daymet" else
                ["clean"] + [f"indep_p{p}" for p in (10, 30, 50)] + [f"allday_p{p}" for p in (10, 30, 50)]
                + ["keep_daymet", "keep_maurer", "keep_nldas", "keep_daymet_maurer", "keep_daymet_nldas",
                   "keep_maurer_nldas"])
        models = ({"gss": "convcnp_gss_daymet", "LSTM-IR": "lstm_ir_daymet"} if which == "daymet" else
                  {"gss": "convcnp_gss_multi", "LSTM-MM": "lstm_mm_multi", "LSTM-IR": "lstm_ir_multi"})
        out = "missing_daymet" if which == "daymet" else "missing_scenarios"
        rows, comps, grid, seedmean = {}, [], {}, {}
        for sc in scen:
            for m, cfg in models.items():
                rows[f"{m} {sc}"] = ens([S / run_dir(cfg, k).name / sc for k in (1, 2, 3)])
                grid.setdefault(m, {})[sc] = rows[f"{m} {sc}"].NSE.median()
                seedmean.setdefault(m, {})[sc] = np.mean([single(S / run_dir(cfg, k).name / sc).NSE.median()
                                                          for k in (1, 2, 3)])
            comps += [(f"gss {sc}", f"LSTM-IR {sc}")] if which == "daymet" else \
                [(f"gss {sc}", f"LSTM-MM {sc}"), (f"gss {sc}", f"LSTM-IR {sc}"), (f"LSTM-IR {sc}", f"LSTM-MM {sc}")]
        g, sm = pd.DataFrame(grid), pd.DataFrame(seedmean)
        print("median NSE, 3-seed ensembles\n", g.round(3).to_string())
        print("median NSE, mean of single seeds\n", sm.round(3).to_string())
        g.to_csv(f"{R}/{out}_median_nse.csv"); sm.to_csv(f"{R}/{out}_seedmean_median_nse.csv")
        table(rows, comps, out)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
