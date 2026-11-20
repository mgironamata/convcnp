#!/usr/bin/env python
"""Write a tiny synthetic dataset in the exact CAMELS-US file layout (for smoke tests only).

Streamflow is a two-reservoir response to synthetic precipitation, so models can learn something.

  python nh_experiments/scripts/make_synthetic_camels.py /scratch2/mg963/data/camels_us_synthetic
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_configs import STATIC  # noqa: E402

BASINS = ["01000001", "02000002", "03000003"]


def main(out: Path, start="1978-01-01", end="2011-12-31"):
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    doy = dates.day_of_year.values
    for b_i, basin in enumerate(BASINS):
        huc = basin[:2]
        area_m2 = int(5e8 * (b_i + 1))
        season = np.sin(2 * np.pi * (doy - 80) / 365.25)
        tmax = 15 + 12 * season + rng.normal(0, 3, n)
        tmin = tmax - 10 + rng.normal(0, 2, n)
        prcp = rng.gamma(0.4, 8, n) * (rng.random(n) < 0.35 + 0.1 * season)
        srad = 250 + 120 * season + rng.normal(0, 30, n)
        vp = 800 + 500 * season + rng.normal(0, 80, n)
        dayl = 43200 + 15000 * season
        # two linear reservoirs (fast + slow)
        fast, slow, q = 0.0, 50.0, np.zeros(n)
        for t in range(n):
            fast += 0.6 * prcp[t]; slow += 0.4 * prcp[t] - 0.02 * max(tmax[t], 0) * 0.1
            slow = max(slow, 0)
            q_f, q_s = 0.3 * fast, 0.02 * slow
            fast -= q_f; slow -= q_s
            q[t] = q_f + q_s
        q_cfs = q * area_m2 * 1e-3 / 86400 / 0.0283168466
        q_cfs[rng.random(n) < 0.01] = -999.0  # some missing values, like CAMELS

        fdir = out / "basin_mean_forcing" / "daymet" / huc
        fdir.mkdir(parents=True, exist_ok=True)
        with open(fdir / f"{basin}_lump_cida_forcing_leap.txt", "w") as fp:
            fp.write(f"{40 + b_i:.5f}\n{500 + 100 * b_i:.1f}\n{area_m2}\n")
            fp.write("Year Mnth Day Hr\tdayl(s)\tprcp(mm/day)\tsrad(W/m2)\tswe(mm)\ttmax(C)\ttmin(C)\tvp(Pa)\n")
            for t, d in enumerate(dates):
                fp.write(f"{d.year} {d.month:02d} {d.day:02d} 12\t{dayl[t]:.2f}\t{prcp[t]:.2f}\t{srad[t]:.2f}"
                         f"\t0.00\t{tmax[t]:.2f}\t{tmin[t]:.2f}\t{vp[t]:.2f}\n")
        qdir = out / "usgs_streamflow" / huc
        qdir.mkdir(parents=True, exist_ok=True)
        with open(qdir / f"{basin}_streamflow_qc.txt", "w") as fp:
            for t, d in enumerate(dates):
                fp.write(f"{basin} {d.year} {d.month:02d} {d.day:02d} {q_cfs[t]:10.2f} A\n")

    adir = out / "camels_attributes_v2.0"
    adir.mkdir(parents=True, exist_ok=True)
    names = pd.DataFrame({"gauge_id": BASINS, "huc_02": [int(b[:2]) for b in BASINS],
                          "gauge_name": [f"synthetic {b}" for b in BASINS]})
    names.to_csv(adir / "camels_name.txt", sep=";", index=False)
    attrs = pd.DataFrame(rng.normal(size=(len(BASINS), len(STATIC))) + 2, columns=STATIC)
    attrs.insert(0, "gauge_id", BASINS)
    attrs.to_csv(adir / "camels_synth.txt", sep=";", index=False)
    (out / "basin_list.txt").write_text("\n".join(BASINS) + "\n")
    print(f"synthetic CAMELS-US written to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
