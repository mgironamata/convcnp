# convcnp (landau): study A, the Nearing et al. 2022 setting, ConvCNP only, 2026-11-26

**Setup**
- Same as Nearing et al. 2022: 531 CAMELS-US basins, Kratzert periods (train WY2000–08, test WY1990–99), 3 forcing products and 27 static attributes.
- Observed streamflow from the day before (lag 1) is an extra input to the causal ConvCNP `gss`. Missing values go through the density channel, with no substitution or flag.
- **gss-AR50:** half of the streamflow input hidden during training, in runs of mean 5 days (Nearing's main setting).
- **gss-AR0:** nothing hidden during training.
- Both are also trained with the same input masking as `convcnp_gss_multi` (p_time = p_seq = 0.1). 3 seeds each.
- Tests: 0/25/50/75/100% of the streamflow input missing, in runs of mean 5 days. The masks are fixed per (basin, fraction) and identical for all runs.
- On the owner's instruction, the LSTMs were not retrained. They are compared using Nearing's published values.

**Run.** Started 06:43Z on the owner's go typed in camels. Training finished 09:38Z and evaluation 10:12Z; all 12 jobs exited with rc = 0. 3 jobs ran in parallel, using 5.7 GB.

## Median NSE

Mean of single seeds is the closest match to Nearing's single models. The gss-noQ column is `convcnp_gss_multi` on clean data; it doesn't see streamflow, so one value applies to every row.

| test: streamflow missing | gss-AR50 | gss-AR0 | gss-noQ | Nearing AR-LSTM, trained with 50% hidden | Nearing AR-LSTM, trained with 0% hidden | Nearing simulation LSTM |
|---|---|---|---|---|---|---|
| 0% | 0.844 | 0.863 | 0.755 | 0.872 | 0.879 | 0.796 |
| 25% | 0.831 | 0.738 | 0.755 | ~0.861 | ~0.857 | 0.796 |
| 50% | 0.816 | 0.631 | 0.755 | ~0.845 | ~0.831 | 0.796 |
| 75% | 0.800 | 0.514 | 0.755 | ~0.835 | ~0.798 | 0.796 |
| 100% | 0.698 | 0.229 | 0.755 | ~0.757 | ~0.553 | 0.796 |

Nearing's 0% row is exact, from `zero_lag_zero_holdout_metrics_table.txt` in their repo. The other rows were read off their Figure 1 by eye (about ±0.005).

The same test as 3-seed ensembles (ours):

| test | gss-AR50 | gss-AR0 | gss-noQ |
|---|---|---|---|
| 0% | 0.861 | 0.874 | 0.768 |
| 25% | 0.851 | 0.759 | 0.768 |
| 50% | 0.838 | 0.655 | 0.768 |
| 75% | 0.822 | 0.544 | 0.768 |
| 100% | 0.722 | 0.251 | 0.768 |

Clean, per seed:

| model | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| gss-AR50 | 0.846 | 0.845 | 0.841 |
| gss-AR0 | 0.863 | 0.860 | 0.866 |

## Paired by basin, 3-seed ensembles, 531 basins

| comparison | 0% missing | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| AR50 vs AR0: win rate | 23% | 93% | 97% | 99% | 99.8% |
| AR50 vs noQ: win rate | 97% | 97% | 93% | 92% | 35% |
| AR0 vs noQ: win rate | 98% | 48% | 14% | 5% | 0.4% |

All p < 1e-17, except AR0 vs noQ at 25% (p = 0.02).

Source: `convcnp_runs/ar_*.csv`, from `ensemble_report.py ar`.

## Verdict

**Lagged streamflow helps the ConvCNP as much as it helps the LSTM, or a little more.**
- Gain on clean data (single seeds): +0.089 for AR50 and +0.108 for AR0 over gss-noQ.
- Nearing's LSTM gains +0.076 and +0.083.
- The ConvCNP with streamflow beats the one without in 97–98% of basins.

**It does not beat the LSTM in absolute terms.**
- gss-AR50 is about 0.03–0.035 below Nearing's AR-LSTM at 0–75% missing, and about 0.06 below at 100%.
- gss-AR0 is about 0.016 below on clean data.
- These gaps track the no-streamflow gap (gss-noQ 0.755 vs 0.796). On the same split, our own LSTM-IR scores 0.807 (3-seed ensemble), against 0.768 for gss.

**The density channel does not stand in for training with gaps.**
- gss-AR0 degrades much faster than Nearing's AR0 LSTM: 0.631 vs ~0.831 at 50% missing, and 0.229 vs ~0.553 at 100%.
- Once more than 25% of streamflow is missing it falls below gss-noQ. This is the same pattern as the Daymet no-masking control.
- The AR LSTM substitutes its own prediction for missing streamflow. The ConvCNP only has the density channel, which evidently does not generalise to unseen gap rates.

**Training with 50% hidden is the robust choice for the ConvCNP too.**
- It costs 0.019 clean (it loses to AR0 in 77% of basins) and wins from 25% missing onwards.
- At 100% missing it falls 0.057 below gss-noQ (Nearing's LSTM: 0.039 below its simulation model), so a separate model without streamflow is still better when there are no observations at all.

## Comparability caveats

1. **Masks:** Nearing's test masks are drawn at random; ours are fixed per basin and fraction.
2. **Seeds:** Nearing reports single models. Our single-seed means are over 3 seeds, which spread by about ±0.003 on clean data.
3. **Figure values:** values read off the figure carry about ±0.005.
4. **Input masking:** our gss-AR and gss-noQ were also trained with 10% forcing masking. On Daymet this cost gss about 0.04 clean, so part of the gap to the LSTM may come from that. Nearing's LSTMs have no forcing masking.
5. **Basin-level comparison:** not possible. Nearing's per-basin results are only in a pickle (`time_split_statistics.pkl`), which we did not open.

## Code

Local commits on `nh-experiments`:
- 61b41e1: model, configs and scenarios, 30 tests.
- The `ar` report mode, committed today.

Queue script: `convcnp_runs/ar_queue.sh`. Nothing is pushed, and this note is not committed.
