# ConvCNP vs LSTM on CAMELS-US: changes, experiments and results

Status on 2026-11-23. This document covers the full work from the 2020 MRes ConvCNP to the current configuration (`gss`).

It has five parts:
- what was changed and why;
- everything tried and dropped, with the numbers;
- the complete comparison against the LSTM;
- what has not been run;
- the next steps.

Paths are relative to the repository root (`/scratch2/mg963/convcnp`), unless they start with `convcnp_runs/`, which means `/scratch2/mg963/convcnp_runs/`.

**In one paragraph.** The 2020 ConvCNP scored a median NSE of 0.37, because of pipeline bugs rather than a weak idea. The rewrite fixes those bugs and runs the ConvCNP inside neuralhydrology, with the same data, loss and evaluation as the LSTM. With tuning, it reaches:
- gauged basins: median NSE 0.752, against the LSTM's 0.764 (3-seed ensembles);
- ungauged basins (PUB): 0.664 against 0.679.

So it gets within about 0.012–0.015 of the LSTM but does not beat it. It is more robust in the tail: higher mean NSE in PUB and fewer failing basins. It also trains about 5–10× faster.

Its remaining weakness is over-smoothing: peaks are too low and the flow variability is too small. Eight targeted attempts to fix that all failed. Averaging LSTM and ConvCNP runs together beats an LSTM-only ensemble, but against a 5-seed LSTM the gain is small: median 0.774 vs 0.772, better in 61% of basins.

Metric shorthand used throughout:
- **NSE**: Nash–Sutcliffe efficiency. 1 is perfect, and below 0 is worse than predicting the mean.
- **Alpha**: Alpha-NSE, the ratio of simulated to observed flow variability. 1 is ideal; below 1 means too smooth.
- **FHV / FLV**: % bias in the highest 2% and lowest 30% of flows. 0 is ideal.
- **KGE**: Kling–Gupta efficiency.
- **Median**: the median over basins.
- **Ensemble**: the average of the members' simulated discharge, with metrics recomputed. It is not an average of metrics.

---

## 1. What changed between 2020 and `gss`, and why

### 1.1 Bugs in the 2020 code (details in `REVIEW.md`)

| # | 2020 problem | Effect | Fix |
|---|---|---|---|
| 1 | `UNet()` was called with its default `in_channels=8` while 5 static dims were concatenated, so the encoder output had **3 channels** | All weather inputs squeezed through `Linear(11→3)` + sigmoid; 51k-parameter model | Per-variable Gaussian SetConv with a density channel (30 channels) → 64 hidden channels (`convcnp_nh/model.py`) |
| 2 | Training kept a random 50–165 + 150–200 of the 365 days, and forcings existed only on those days | About 23% of training days had no rain to see. The model learned to hedge, and test inputs (dense) differed from training | Dense daily forcings. Random hiding survives only as the optional `input_dropout` (default 0) |
| 3 | Non-causal UNet in training, but a causal test at the window edge | Evaluated on a setting it rarely trained on | Causal SetConv and causal dilated TCN. `seq_length` 910 with `predict_last_n` 365 means each training target sees its full 537-day history, exactly as at test time (checked at start-up) |
| 4 | Discharge standardised with one pooled mean/σ, shifted by +1.42, gamma likelihood | High-flow basins dominate the loss; arid basins collapse (NSE down to −28.7); the gamma covers negative flow | NSE* loss (each basin weighted by 1/σ_basin), regression head. `gs` also keeps Q uncentred with a softplus output, so predictions are ≥ 0 |
| 5 | Windows sampled over the concatenated multi-basin table | About 6.7% of windows mix two basins | neuralhydrology samples windows per basin |
| 6 | Rows with missing Q deleted; time taken from the row index | Gaps and lost forcings | All rows kept; missing Q masked in the loss; NaN forcings handled by the density channel |
| 7 | Normalisation statistics included the test period | Slight leakage | Training-period statistics only |
| 8 | Evaluation skipped each basin's first year and every year's last day | Minor bias | Every test day predicted as the last step of its own window |
| 9 | Metric bugs: `exp` on a never-logged Q, so train NSE was NaN; printed per-basin NSE untransformed; a double back-transform | No usable monitoring | neuralhydrology's metric code, shared with the LSTM |
| 10 | PUB checkpoint chosen on the same "ungauged" basins; SAC-SMA calibrated per basin | 2020 PUB score of 0.31 is optimistic and has no fair baseline | 5-fold CV over basins; last epoch only; validation on training basins only |
| 11 | Constant LR, no clipping, single run | Noisy, no ensemble | LR schedule {0: 1e-3, 10: 5e-4, 25: 1e-4}, gradient clipping at 1, 3 seeds |

### 1.2 Pipeline

- **neuralhydrology 1.13.0, not forked.** `convcnp_nh.register()` adds `model: convcnp` and a `convcnp:` config block, and patches neuralhydrology at run time (`convcnp_nh/__init__.py`, covered by `tests/test_convcnp.py`):
  - **Tester `predict_last_n` = 1 for the ConvCNP.** Without this, the stock tester silently skipped the first 364 test days.
  - **CMAL loss masked element-wise.** The stock loss produced NaN on fully masked batches.
  - **CMAL sampling uses the tester's `predict_last_n`.** Validation had been 365× too slow.
- **Optional training extras**, used only in ablations: `weight_decay`, `ema_decay`, and (added this week) `alpha_loss_weight`, `peak_loss_weight` / `peak_quantile`, and `loss_warmup_updates`.
- **Model option** added this week: `convcnp.local_blocks`.
- **Configs** are generated by `scripts/make_configs.py`. Data, inputs and periods are identical for both models: 531 basins; Daymet prcp, srad, tmax, tmin, vp; Kratzert's 27 static attributes.
- **Splits:**
  - MRes (gauged): train WY1981–95, test WY1996–2010.
  - Tuning: train WY1981–90, evaluate WY1991–95. All model choices were made here, so the test period was untouched.
  - Kratzert 2019 periods: sanity check for the LSTM.
  - PUB: 5-fold CV over basins (`scripts/make_pub_folds.py`, seed 0), MRes periods, pooled so every basin is scored once, out of sample (`scripts/pub_report.py`).
- **No checkpoint selection:** the last epoch is always evaluated.

### 1.3 Architecture and training

| setting | 2020 | v1 (`convcnp_mres`) | `gs` | `gss` (current) |
|---|---|---|---|---|
| Encoder | DeepSet → ConvDeepSet → **3** channels | Gaussian SetConv per variable + density, length scale 2 d | multi-scale SetConv **[0.25, 2, 8] d** | = gs |
| Static attributes | DeepSet mean-pool → 5 dims | MLP → FiLM in every TCN block | same | same |
| Core CNN | 12-layer UNet, non-causal | 7 causal dilated residual blocks, kernel 3, 64 channels, LayerNorm, GELU | + **gated** (tanh·sigmoid) blocks | = gs |
| Read-out | SetConv for mean and σ | SetConv, 2 d | scales **[0.25, 2] d** + identity path | = gs |
| Norm before head | — | LayerNorm | **none** | = gs |
| Output | gamma | linear | **softplus**, uncentred Q | + **`head_bias_init` = 1.0** |
| Loss | gamma NLL on shifted Q | NSE* | NSE* | NSE* |
| Dropout (blocks / output) | input masking, then 0 | 0.1 / 0.1 | **0.25 / 0.3** | = gs |
| Updates per epoch × epochs | 256 tasks × up to 500, best epoch picked | 1000 × 30 | **500** × 30 | = gs |
| Parameters | 51,378 | 235,718 | 323,536 | 323,536 |
| Receptive field | whole window, both directions | 537 d, past only | 537 d | 537 d |

**The `head_bias_init` fix** (`gs` → `gss`):
- **What it does.** The final head layer's weights start ×0.1, and its bias is set so that softplus(bias) = 1 normalised flow unit.
- **Why.** Without it, a softplus head sometimes starts in its flat region and never recovers ("dead start": constant output, training loss stuck at about 0.1).
- **Evidence.** 3 of 15 `gs` PUB runs were dead from epoch 1 (folds 1, 3 and 4, seed 1). With the fix, all 15 `gss` runs were healthy. It first appeared in tuning as `J_skipfix`, after a `skip` run died.

How `gs` was reached on the tuning split (median NSE, seed 1). For reference: LSTM 0.760, v1 0.715.

| step | median NSE |
|---|---|
| multi-scale | 0.719 |
| + non-negative output | 0.724 (removed the low-flow bias: FLV −144 → +14) |
| + no out-norm | 0.728 |
| + more dropout | 0.732 |
| + gated blocks | 0.735 |
| + 500 updates per epoch = `gs` | 0.742 / 0.733 / 0.728 (seeds 1/2/3) |

The individual steps are each inside single-seed noise. Only the cumulative gain, 0.715 → about 0.734, is clear.

---

## 2. Tried and dropped (tuning split, evaluated WY1991–95)

### 2.1 Before this week

Recomputed from `convcnp_runs/tune_*/test/model_epoch030/test_metrics.csv`. The overrides are in `convcnp_runs/tune_queue*.sh`.

| variant | change | median NSE (seeds) | outcome |
|---|---|---|---|
| `ms_indrop` | input dropout 0.1 | 0.650 | Hurt: Alpha 0.68, FHV −35 (hedging, as in 2020) |
| `ms_c128` | 128 channels | 0.725 | No gain; FLV −42 |
| `nn_nonorm_c128` | 128 channels, less regularisation | 0.705 | Hurt (overfits) |
| `nn_nonorm_drop_nobn` | no LayerNorm inside blocks | killed | Collapsed to a constant output |
| `nn_pln60` | 60-day target blocks, batch 256 | 0.726 | Noise |
| `I_ema` | EMA weights 0.999 | 0.739 / 0.733 / 0.730 | No gain |
| `I_wd` | AdamW, weight decay 0.05 | 0.728 / 0.729 / 0.733 | No gain |
| `I_skip` | WaveNet skip outputs | 0.738 / 0.745 / **−0.331** | Seed 3 dead start; led to `head_bias_init` |
| `J_skipfix` | skip + `head_bias_init` | 0.735 / 0.735 / 0.743 | Healthy, no gain |
| `I_all` | EMA + skip + weight decay | 0.740 / 0.734 / 0.737 | No gain |
| `K_cmal` | CMAL probabilistic head (3 components) | 0.694 / 0.681 / 0.695 | Hurt: Alpha 0.68, FHV −32. 90% interval coverage 0.81 |

### 2.2 This week: batches L, M and N, aimed at the over-smoothing

The comparison point is `gss` run in the same code base: seeds 1/2/3 = 0.726 / 0.726 / 0.723, 3-seed ensemble 0.753.

"Better than gss" is the share of basins where the variant's NSE beats `gss` seed 1. The p-value is from a paired Wilcoxon test. Full table: `convcnp_runs/pilots_LMN_seed1_summary.csv` and `..._paired.csv`, made by `scripts/ensemble_report.py pilots`.

| variant (batch) | change | median NSE | Alpha | FHV | FLV | better than gss | p |
|---|---|---|---|---|---|---|---|
| LSTM reference | — | 0.760 | 0.93 | −8.6 | −3.5 | 67% | 3e-13 |
| `gss` | — | 0.726 | 0.87 | −15.1 | +12.8 | — | — |
| `lowdrop` (L) | dropout 0.1 / 0.1 | 0.707 | 0.87 | −14.1 | +3.1 | 38% | 6e-10 |
| `sharp` (L, M) | read-out scale 0.25 d only (no 2-day kernel) | 0.736 / 0.737 / 0.725 | 0.87–0.88 | −12.6 to −14.4 | +6 to +12 | 52–55% per seed | 0.20 / 0.11 / 0.89 |
| `sharp` 3-seed ensemble | | 0.760 (vs gss 0.753) | 0.86 | −14.9 | +17.6 | 55% | 0.08 |
| `alpha03` (L) | variance-matching loss λ=0.3 | 0.697 | 0.90 | −11.7 | −26 | 31% | 2e-19 |
| `alpha10` (L) | λ=1.0 | 0.636 | **0.95** | **−8.2** | −64 | 15% | 2e-58 |
| `alpha01` (M) | λ=0.1 | −0.331 | 0 | −100 | +81 | 2% | — (dead start) |
| `alpha01w` (N) | λ=0.1 + 500-step warm-up | 0.726 | 0.88 | −14.3 | −14 | 43% | 2e-4 |
| `peak05` (N) | high-flow loss μ=0.5, top 5% of each window + warm-up | 0.689 | 0.93 | −10.8 | −36 | 26% | 1e-32 |
| `peak20` (N) | μ=2.0 | 0.643 | 0.96 | −11.2 | −57 | 13% | 2e-60 |
| `local2` (N) | 2 undilated blocks before the dilated stack (RF 545 d, 415k parameters) | 0.732 | 0.87 | −14.7 | +4 | 48% | 0.65 |
| `local2sharp` (N) | local2 + sharp | 0.731 | 0.88 | −13.6 | +7 | 51% | 0.92 |

What these show:
- **Variance and high-flow losses** fix the variability statistic (Alpha reaches or exceeds the LSTM's) by inflating variance everywhere. Low flows are pulled far too low and NSE falls. Neither moves the NSE–Alpha trade-off.
- **Less dropout hurts.**
- **Short-range blocks are neutral.**
- **`sharp`** is a small gain (about +0.007), the same sign for all three seeds but not significant. It has 128 fewer parameters, so it is a reasonable default, but it does not close the gap.
- **Warm-up** (`loss_warmup_updates`) stops the dead start that an extra loss term can cause.
- **Seed noise.** Single-seed median NSE varies by about ±0.015 between almost identical configurations (`gs` vs `gss`, seed 1: 0.742 vs 0.726). One-seed pilots cannot detect gains of about 0.01.

---

## 3. Comparison against the LSTM

The LSTM is neuralhydrology `cudalstm` with Kratzert's hyper-parameters, not re-tuned: 256 hidden units, 365-day sequences, dropout 0.4, 30 epochs.

**Sanity check.** On Kratzert's 2019 periods it gets 0.744 / 0.750 / 0.738, which matches the literature.

### 3.1 Gauged basins (MRes test WY1996–2010, 531 basins)

Sources: `convcnp_runs/summary_final_mres.csv` and `convcnp_runs/mres_mixed_control_{summary,paired}.csv` (from `scripts/ensemble_report.py mres`).

| model | median NSE | mean NSE | NSE < 0 | KGE | Alpha | FHV | FLV |
|---|---|---|---|---|---|---|---|
| LSTM seed 1 / 2 / 3 | 0.736 / 0.740 / 0.740 | — | — | — | — | — | — |
| LSTM seed 4 / 5 | 0.740 / 0.738 | 0.694 / 0.684 | 6 / 3 | 0.789 / 0.782 | 0.94 / 0.96 | −6.3 / −3.6 | — |
| LSTM seed 6 (**diverged**) | 0.722 | **−0.831** | **82** | 0.748 | 0.93 | −6.9 | — |
| LSTM ensemble, seeds 1–3 | 0.764 | 0.719 | 3 | 0.792 | 0.94 | −5.7 | +11.1 |
| LSTM ensemble, seeds 1–5 | 0.772 | 0.724 | 2 | 0.795 | 0.94 | −5.6 | +17.6 |
| LSTM ensemble, seeds 1–6 | 0.760 | 0.654 | 17 | 0.785 | 0.94 | −5.9 | +6.7 |
| ConvCNP `gs` seed 1 / 2 / 3 | 0.725 / 0.721 / 0.722 | — | — | — | — | — | — |
| ConvCNP `gs` ensemble (3) | 0.752 | 0.709 | 2 | 0.771 | 0.88 | −12.5 | +28.7 |
| **LSTM 3 + ConvCNP `gs` 3 (mixed)** | **0.774** | **0.734** | 2 | 0.792 | 0.90 | −9.9 | +27.4 |
| ConvCNP v1 ensemble (3) | 0.736 | 0.697 | 3 | 0.754 | 0.84 | −15.1 | −18.5 |
| SAC-SMA (per-basin calibrated, mean of 10) | 0.665 | 0.572 | 11 | 0.734 | 0.87 | −10.7 | +28.4 |
| ConvCNP 2020 (424 basins, single run) | 0.37 | −0.14 | — | — | — | — | — |

Paired by basin (share of basins where the first beats the second; median per-basin NSE difference; Wilcoxon p):

| comparison | wins | median difference | p |
|---|---|---|---|
| ConvCNP ens 3 vs LSTM ens 3 | 34.8% | −0.012 | 3e-10 |
| mixed vs LSTM ens 3 | 69.9% | +0.007 | 2e-23 |
| mixed vs LSTM ens 5 (fairest control) | 61.0% | +0.0035 | 6e-9 |
| mixed vs LSTM ens 6 | 62.5% | +0.006 | 3e-16 |
| LSTM ens 6 vs LSTM ens 3 | 61.6% | +0.003 | 0.01 |

Notes:
- **The 6-seed control was added on 2026-11-21:** seeds 4–6, run by `convcnp_runs/mres_queue_lstm456.sh`.
- **Seed 6 diverged late in training.** Its training loss rose to 0.16 at epoch 29 and 0.47 at epoch 30, against about 0.008 for a normal seed. It fails badly in 82 basins (worst NSE about −62), which drags the 6-seed ensemble down. The seeds 1–5 ensemble is therefore the fair comparison.
- **The LSTM recipe can fail on an unlucky seed.** No ConvCNP `gss` run has failed since `head_bias_init` was added (15 PUB runs plus 3 tuning runs).
- **The gauged ConvCNP numbers are from `gs`.** `gss` (which only adds the dead-start fix) has never been run on the gauged test; the 3 `gs` gauged seeds were all healthy.
- **Compute.** On its own, one ConvCNP seed trains in about 25 min, against about 4.5 h for one LSTM seed. A 3-seed ConvCNP ensemble (0.752) beats a single LSTM (about 0.739) at less than a third of the GPU time.

### 3.2 Ungauged basins (PUB, 5-fold CV, every basin scored once out of sample)

Sources: `convcnp_runs/pub_lstm3_vs_gss3_summary.csv` and `convcnp_runs/pub_mixed_{summary,paired}.csv` (from `scripts/ensemble_report.py pub`).

| model | median NSE | mean NSE | NSE < 0 | KGE | Alpha | FHV | FLV |
|---|---|---|---|---|---|---|---|
| LSTM seed 1 / 2 / 3 | 0.635 / 0.644 / 0.644 | 0.437 / 0.452 / 0.324 | 46 / 38 / 48 | 0.675 / 0.675 / 0.662 | 0.93–0.95 | −6 to −8 | −9 to −10 |
| LSTM ensemble (3) | **0.679** | 0.478 | 37 | **0.694** | 0.91 | −8.5 | +16.8 |
| ConvCNP `gss` seed 1 / 2 / 3 | 0.626 / 0.630 / 0.616 | 0.446 / 0.465 / 0.458 | 37 / 38 / 36 | 0.659 / 0.652 / 0.657 | 0.87 | −12 | +19 to +24 |
| ConvCNP `gss` ensemble (3) | 0.664 | **0.519** | **33** | 0.675 | 0.85 | −13.7 | +30.4 |
| **LSTM 3 + ConvCNP `gss` 3 (mixed)** | **0.696** | **0.537** | **29** | 0.694 | 0.88 | −12.6 | +26.9 |
| ConvCNP `gs` ensemble (3 dead runs included) | 0.595 | 0.525 | 19 | 0.491 | 0.66 | −33.1 | — |
| ConvCNP 2020 (single split) | 0.31 | — | — | — | — | — | — |

Paired by basin:

| comparison | wins | median difference | p |
|---|---|---|---|
| ConvCNP ens 3 vs LSTM ens 3 | 40.7% | −0.014 | 0.003 |
| mixed vs LSTM ens 3 | 63.7% | +0.011 | 2e-15 |
| mixed vs ConvCNP ens 3 | 78.9% | +0.026 | 6e-33 |

There is no 5- or 6-seed LSTM PUB control, so the PUB mixed result is partly an ensemble-size effect, as it was in the gauged setting.

### 3.3 Overall verdict

- **Like for like (same number of seeds):** the LSTM wins on median NSE by about 0.012 (gauged) and 0.015 (PUB). It is better in about 60–65% of basins. The gap is the over-smoothing: ConvCNP Alpha is 0.85–0.88 against 0.91–0.94, and FHV is about −13 against −6 to −8.
- **Where the ConvCNP is better:** the tail. It has a higher mean NSE in PUB and fewer basins below 0. No dead runs since the fix, while one LSTM seed in six diverged. And training is 5–10× cheaper.
- **Mixing the two models gives the best numbers anywhere:** 0.774 gauged, 0.696 PUB. Against the fairest control (5 LSTM seeds), the true mixing gain is small but significant: +0.0035 per basin, 61% wins.

---

## 4. Not run (configs exist, no runs)

- `convcnp_gss_mres` (gauged `gss`)
- `convcnp_gs_kratzert2019`, `convcnp_kratzert2019`
- `convcnp_cmal_mres`, `convcnp_gs_cmal_mres`
- `sharp` on the MRes test or PUB

## 5. Next steps

See the plan in the 2026-11-23 report (`/scratch2/mg963/agent-coordinator/inbox/2026-11-23-convcnp-landau.md`). In short:
1. A compute-matched gauged `gss` run with 6+ seeds.
2. A missing-forcing robustness test, where the ConvCNP's density channel should give it a real advantage.
3. A clean extra LSTM seed.

The plain-simulation gap is unlikely to close with reasonable effort.

## File index

| what | where |
|---|---|
| Review of the 2020 code | `REVIEW.md` |
| Model / patches / losses | `nh_experiments/convcnp_nh/{model.py,__init__.py}` |
| Configs (generated) | `nh_experiments/scripts/make_configs.py` → `nh_experiments/configs/` |
| Tests (23, CPU) | `nh_experiments/tests/test_convcnp.py` |
| Run summaries | `convcnp_runs/{summary_final_mres,pub_lstm3_vs_gss3_summary,mres_mixed_control_*,pub_mixed_*,pilots_LMN_seed1_*}.csv` |
| Summary scripts | `nh_experiments/scripts/{summarize,pub_report,ensemble_report,tune_report}.py` |
| Queues and logs | `convcnp_runs/*queue*.sh`, `convcnp_runs/logs/` |
