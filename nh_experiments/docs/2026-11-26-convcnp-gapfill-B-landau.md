# convcnp (landau): study B, gap filling observed streamflow, ConvCNP only, 2026-11-26

**Setup**
- 531 CAMELS-US basins, Kratzert periods (train WY2000–08, test WY1990–99), 3 forcing products and 27 static attributes.
- **gss-gapfill:** a non-causal ConvCNP `gss`, given the same day's observed streamflow as an input channel.
  - In training, 2–6 gaps per 910-day window are hidden in that input. Gap lengths are log-uniform over 1–120 days.
  - The loss counts only hidden days in the central 374 days of each window, which have the full ±268-day context on both sides. So the model cannot copy its input.
  - No forcing masking. 3 seeds. Config `convcnp_gss_gapfill`.
- **Test:** for each basin and gap length (7, 30 or 90 days), non-overlapping gaps covering about 15% of test days.
  - Each gap has at least 7 observed days on both sides, and at least half its own days observed.
  - Masks are fixed (crc32 seed) and identical for every method.
  - Scores use the observed gap days only.
- **Comparison methods:** no LSTM was trained, as the owner asked; all come from existing runs.
  - linear and log-linear interpolation across the gap;
  - LSTM-IR simulation, the 3-seed ensemble from the missing-input study, with no streamflow input;
  - the same simulation bias-corrected with the observed/simulated volume ratio of the 7 days before and after each gap, interpolated across the gap;
  - gss-noQ: `convcnp_gss_multi`, trained with 10% forcing masking;
  - gss-AR50 from study A, which sees streamflow up to the day before only.
- **Run:** owner's go typed in camels at 10:43Z. Training 10:48–11:24Z, predictions 11:24–11:29Z, report 11:30Z; all jobs exited with rc = 0. 3 jobs, 1.86 GB each, 5.7 GB in total.

## Median NSE on gap days

3-seed ensembles; 531 basins.

| method | 7-day gaps | 30-day gaps | 90-day gaps |
|---|---|---|---|
| linear interpolation | 0.282 | 0.084 | 0.009 |
| log-linear interpolation | 0.373 | 0.120 | 0.025 |
| gss-noQ simulation | 0.798 | 0.784 | 0.774 |
| LSTM-IR simulation | 0.829 | **0.811** | **0.808** |
| LSTM-IR simulation, bias-corrected | **0.856** | **0.826** | 0.791 |
| gss-AR50 (past streamflow only) | 0.848 | 0.806 | 0.751 |
| **gss-gapfill** | 0.836 | **0.811** | 0.805 |

Mean of single seeds:

| model | 7-day | 30-day | 90-day |
|---|---|---|---|
| gss-gapfill | 0.823 | 0.793 | 0.784 |
| gss-AR50 | 0.824 | 0.774 | 0.726 |

Median absolute volume error per gap:

| method | 7-day | 30-day | 90-day |
|---|---|---|---|
| bias-corrected LSTM-IR | 0.10 | 0.12 | 0.15 |
| gss-gapfill | 0.22 | 0.20 | 0.17 |
| LSTM-IR simulation | 0.24 | 0.21 | 0.17 |
| gss-AR50 | 0.16 | 0.20 | 0.22 |

## Paired by basin: gss-gapfill vs each method

Win rate is the share of basins where gss-gapfill has the higher NSE; the median difference is gss-gapfill minus the other method.

| gss-gapfill vs | 7-day: win rate, median difference (p) | 30-day | 90-day |
|---|---|---|---|
| bias-corrected LSTM-IR | 36%, −0.020 (4e-7) | 44%, −0.010 (0.07) | 50%, 0.000 (0.46) |
| LSTM-IR simulation | 61%, +0.010 (2e-7) | 48%, −0.002 (0.53) | 40%, −0.011 (7e-7) |
| gss-AR50 | 41%, −0.007 (7e-6) | 54%, +0.004 (0.09) | 65%, +0.027 (1e-13) |
| gss-noQ | 78%, +0.038 | 67%, +0.023 | 58%, +0.013 |
| interpolation | ≥ 85% | ≥ 94% | ≥ 95% |

Source: `convcnp_runs/gapfill_*.csv` and `gapfill_report.txt`, from `eval_gapfill.py report`.

## Verdict

**The ConvCNP does not beat the practical standard.**
- On 7-day gaps the bias-corrected LSTM simulation is clearly better: gss-gapfill wins in only 36% of basins, and the bias-corrected simulation has half its volume error.
- On 30- and 90-day gaps the two are statistically tied.
- The plain LSTM simulation, which never sees streamflow, ties with gss-gapfill on 30-day gaps and beats it on 90-day gaps.

**The ConvCNP uses the surrounding streamflow only weakly.**
- It beats the no-streamflow gss by just +0.013 to +0.038, and part of that is gss-noQ's forcing-masking handicap.
- A direct check on 3 basins (seed 1): hiding all streamflow instead of only the gaps lowers gap-day NSE by only 0.03–0.07.
- On 7-day gaps it loses even to the past-only gss-AR50, which was trained on dense 5-day holdouts and learned to lean on recent flow.

**Two-sided context pays off only for long gaps.** gss-gapfill beats gss-AR50 on 90-day gaps (wins in 65% of basins, +0.027), where past-only models drift. But there it only matches the LSTM simulation.

**The underlying problem is the same as before.** The ConvCNP's base simulation skill is about 0.03 below the LSTM's, and in this configuration the streamflow context doesn't close that gap.

**Interpolation is not a competitor** beyond a few days.

**Checks:**
- The same-day streamflow input matches the target exactly.
- Hidden gap days reach the model as missing values.
- Training scores only hidden central days (unit-tested).
- Masks are identical across methods.
- The 909 missing target days per basin are neuralhydrology's warm-up period, by design.

## Possible next steps (none launched)

1. **Hybrid:** feed the LSTM-IR simulation to the ConvCNP as an input, so it learns only the correction from the surrounding observations. That gives it the LSTM's base skill. About 1.5 h per seed.
2. **More loss signal per window.** About 25% of central days are scored now. More gaps per window, or longer training, may help the model learn to use the context.
3. **Bidirectional LSTM with the same gap training,** the fair deep-learning competitor from the plan. About 1 day of engineering plus 15–18 GPU-h. Not approved.

## Code

Local commits on `nh-experiments` (convcnp repo):
- 4825273: model, config, evaluator, 32 tests.

Queue script: `convcnp_runs/gapfill_queue.sh`. Nothing is pushed, and this note is not committed in agent-coordinator.
