# Convolutional conditional neural processes for rainfall–runoff modelling on CAMELS-US: technical report and working notes

Status: working notes, 2026-11-27. They cover the work from 2026-11-17 to 2026-11-27 on the 2020 MRes ConvCNP. All numbers come from run outputs under `/scratch2/mg963/convcnp_runs/` and the notes listed in Appendix A. No new experiments were run for this report. Statements marked [CHECK: ...] have not been verified against a primary source.

## Summary

- **The 2020 ConvCNP's poor score was mostly caused by bugs in its data handling.** The 2020 model reached a median Nash–Sutcliffe efficiency (NSE) of 0.37 on gauged basins. We rewrote the model inside neuralhydrology, using the LSTM's data, loss and evaluation, fixed the bugs and made it causal. This raised the 3-seed ensemble median NSE to 0.736 on the same test decade. Further tuning, on a separate split, raised it to 0.752.
- **Like for like, the LSTM is still better at plain simulation.** The LSTM leads by about 0.012 in median NSE in gauged basins (0.764 against 0.752, 3-seed ensembles) and by 0.015 in ungauged basins (0.679 against 0.664). The ConvCNP over-smooths: its flows vary too little and its peaks are too low. Eight attempts to fix this, using losses, regularisation and architecture changes, all failed.
- **Missing forcings.**
  - When neither model was trained on missing data and both got the same gap-filled inputs, the ConvCNP degraded more slowly.
  - When both were retrained with the missing-data simulation of Gauch et al. (2025), the LSTM with input replacing was as good as or better than the ConvCNP. The one exception was days on which every forcing product is missing at once: there the ConvCNP was marginally better.
- **Observed streamflow as an input.** The ConvCNP gains a lot from lagged streamflow but stays about 0.03 below the published autoregressive LSTM of Nearing et al. (2022). For filling gaps in streamflow records, a non-causal ConvCNP that sees the flow on both sides of each gap ties with, or loses to, the LSTM simulation with a simple bias correction.
- **ConvCNP and LSTM ensembles combine well.** Averaging the two models' output with fixed equal weights beats an LSTM-only ensemble in gauged simulation, in ungauged basins and in gap filling. The gap-filling gain disappears when the ConvCNP member is one that does not see streamflow. This is a hybrid result, not a ConvCNP win.
- **Verdict.** There is so far no task on which the ConvCNP alone beats a matched LSTM.

---

## 1. Introduction

**What is known.**
- Long short-term memory (LSTM) networks trained on many basins at once are the reference data-driven rainfall–runoff model.
  - Kratzert et al. (2018) showed that LSTMs can simulate daily discharge in CAMELS-US basins.
  - Kratzert et al. (2019a) showed that a single regional LSTM, or an entity-aware LSTM conditioned on catchment attributes, trained on 531 CAMELS basins, beats calibrated conceptual models, including SAC-SMA, and the US National Water Model.
  - The same group showed that LSTMs also work in basins left out of training (prediction in ungauged basins, PUB; Kratzert et al., 2019b).
  - Using several meteorological products at once improves the LSTM further (Kratzert et al., 2021).
- Later work built on this base.
  - Probabilistic heads such as the countable mixture of asymmetric Laplacians (CMAL) give uncertainty estimates (Klotz et al., 2022).
  - Gauch et al. (2025) compared ways of handling missing forcing inputs.
  - Nearing et al. (2022) compared data assimilation with autoregressive use of near-real-time streamflow.
  - Kratzert et al. (2024) argue that LSTMs should be trained on many basins rather than a single one.
  - The same family of models underlies global operational flood forecasting (Nearing et al., 2024).
- Most of this work uses the neuralhydrology library (Kratzert et al., 2022). We use version 1.13.0.

**What is unknown.**
- Neural processes model a predictive distribution conditioned on an arbitrary set of observations (Garnelo et al., 2018). The convolutional conditional neural process (ConvCNP; Gordon et al., 2020) makes this translation-equivariant and handles off-grid or missing context through a "density" channel.
- ConvCNPs have worked for climate downscaling from irregular station data (Vaughan et al., 2022). In rainfall–runoff modelling their appeal is the same: forcings or streamflow observations are often missing or irregular. A neural process can condition on whatever is observed, at any time and on either side of the target, without imputing it first.
- The 2020 MRes project tried this on CAMELS-US. It reached a median NSE of 0.37, far below LSTM results, and had no LSTM baseline.

**Aims of this work.**
1. Find out why the 2020 ConvCNP performed poorly.
2. Bring it to a fair, like-for-like comparison with the LSTM.
3. Test the tasks where a ConvCNP should have a structural advantage over an LSTM:
   - missing forcings;
   - streamflow observations as extra input, either from the past (nowcasting) or from both sides of a gap (gap filling).

## 2. Data

### 2.1 Basins, inputs and target

**Basins and attributes.**
- 531 CAMELS-US basins (Newman et al., 2015; Addor et al., 2017). This is the subset used by Kratzert et al. (2019a): basins smaller than 2,000 km² and with consistent area estimates [CHECK: selection criteria as stated in Kratzert et al. 2019a]. The basin list is `nh_experiments/basins/531_basin_list.txt`.
- 27 static catchment attributes, the set used by Kratzert et al. (2019a). They cover topography, land cover, soils, geology and climate indices; the list is in `make_configs.py`, `STATIC`.

**Dynamic inputs.** Five daily forcing variables: precipitation, shortwave radiation, maximum and minimum temperature, and vapour pressure.
- **Single-product experiments:** Daymet only (Thornton et al., 1997, 2012; as distributed with CAMELS).
- **Multi-product experiments:** Daymet, Maurer (Maurer et al., 2002) and NLDAS (Xia et al., 2012), in the "extended" versions that include all five variables. These are the HydroShare versions used by Kratzert et al. (2021). With 3 products × 5 variables there are 15 dynamic inputs.

**Target.**
- Daily observed discharge `QObs(mm/d)`, clipped at zero.
- Missing target days are masked in the loss and in the metrics.
- In studies A and B, observed discharge is also an input (section 3.4).

**Changes from the 2020 data.** The 2020 work used 424 gauged training basins. It also used extra inputs: day-of-year sine and cosine, log-transformed precipitation, and `area_geospa_fabric` instead of `area_gages2`.

**Normalisation.** neuralhydrology computes statistics on the training period only. The tuned ConvCNP does not centre discharge (section 3.2).

### 2.2 Periods and splits

All periods are water years (WY), 1 October to 30 September.

| split | training | validation | test | used for |
|---|---|---|---|---|
| MRes (`mres`) | WY1981–95 | WY1991–95 (overlaps training; monitoring only) | WY1996–2010 | gauged simulation, PUB, 6-seed control, missing forcings without retraining. Same split as the 2020 MRes and the Newman et al. (2015) SAC-SMA benchmark |
| Tuning (`mres_tune`) | WY1981–90 | — | WY1991–95 | every model choice for the ConvCNP. The MRes test decade was never used for tuning |
| Kratzert 2019 (`kratzert2019`) | WY2000–08 | WY1981–89 | WY1990–99 | LSTM sanity check; missing-input studies; studies A and B. Also the periods of Gauch et al. (2025) and Nearing et al. (2022) |
| PUB | MRes periods on 4/5 of basins | training basins only | MRes test decade on the held-out 1/5 | 5-fold cross-validation over basins (`scripts/make_pub_folds.py`, seed 0; 106–107 test basins per fold). Pooled so that every basin is scored once, out of sample |

**Checkpoints.** Every reported model is the last epoch. No checkpoint was selected on validation or test data.

**Study B variant selection.** Variants are compared on the Kratzert validation period (WY1981–89) only. The tooling exists, but no variant has been selected yet.

### 2.3 Metrics and statistics

**Metrics.** All metrics are computed with neuralhydrology's metric code, which both models share.
- NSE (Nash and Sutcliffe, 1970).
- Kling–Gupta efficiency (KGE; Gupta et al., 2009).
- Alpha-NSE: the ratio of the simulated to the observed standard deviation. A value of 1 is ideal; below 1 means the simulation is too smooth.
- FHV and FLV: percentage bias in the highest 2% and the lowest 30% of the flow duration curve (Yilmaz et al., 2008).
- Study B also uses root mean squared error (RMSE) and the absolute volume error per gap: |Σsim − Σobs| / Σobs over the observed days of the gap.

**Aggregation.**
- "Median NSE" is the median over basins. Single seeds are summarised either one by one or as their mean ("mean of single seeds").
- An ensemble averages the members' simulated discharge series. The metrics are then recomputed on the averaged series; metrics are never averaged.
- A mixed ensemble is a fixed equal-weight average of the ConvCNP ensemble and the LSTM ensemble. Its weights are never fitted.

**Paired comparisons.** These are made basin by basin on the same masks.
- The win rate is the share of basins in which the first model has the higher metric.
- The median difference is the median of the per-basin differences.
- The p-value comes from a two-sided Wilcoxon signed-rank test over the 531 basins. "Significant" is used only in this statistical sense. Many differences reported here are significant but smaller than 0.01 NSE.

---

## 3. Models

### 3.1 LSTM baselines

All LSTMs are neuralhydrology models, configured in `nh_experiments/scripts/make_configs.py`. None were re-tuned.

| name | neuralhydrology model | settings | source of the configuration |
|---|---|---|---|
| LSTM (gauged, PUB, tuning) | `cudalstm`, regression head, linear output | hidden 256, initial forget-gate bias 3, output dropout 0.4, sequence 365 d, `predict_last_n` 1, batch 256, 30 epochs, Adam, learning rate 1e-3 → 5e-4 (epoch 10) → 1e-4 (epoch 25), gradient clipping at 1, NSE* loss | Kratzert et al. (2019a) hyper-parameters as used in the neuralhydrology CAMELS-US examples [CHECK: Kratzert et al. (2019a) used a 270-day sequence; 365 d follows later neuralhydrology examples]. Reproduces the published level on Kratzert periods: 0.744 / 0.750 / 0.738 for seeds 1–3 |
| LSTM-MM | as above, `nan_handling_method: masked_mean`, input embedding of fully connected ReLU layers [10, 10, 10, 10], 35 epochs | one embedding per forcing product, averaged over the products that are present | best masked-mean configuration of Gauch et al. (2025, Table A2), on top of the LSTM above |
| LSTM-IR | as above, `nan_handling_method: input_replacing`, embedding [17, 10], positional encoding size 5, 30 epochs | missing inputs set to zero, plus a binary flag per product | best input-replacing configuration of Gauch et al. (2025, Table A2) |
| Autoregressive LSTM (study A) | not trained | hidden 128, sequence 365, lag-1 streamflow input; a missing observation is replaced by the model's own prediction, plus a flag | published values from Nearing et al. (2022) |

**NSE\* loss.** NSE\* is the squared error of each sample divided by (σ_b + ε)², where σ_b is the standard deviation of discharge in basin b over the training period (Kratzert et al., 2019a). It stops high-flow basins from dominating the loss. Both models use it in every experiment.

**SAC-SMA.** The calibrated conceptual baseline on the MRes split is the SAC-SMA benchmark of Newman et al. (2015): calibrated per basin, mean of 10 calibrations (`convcnp_runs/sacsma_daymet_mres_test.csv`).

### 3.2 The ConvCNP as implemented (`convcnp_gss`)

**Code.** The model is `nh_experiments/convcnp_nh/model.py`. It is registered with neuralhydrology as `model: convcnp` through `convcnp_nh.register()`, which patches neuralhydrology at run time; neuralhydrology itself is not forked. It is an on-the-grid ConvCNP (Gordon et al., 2020) at the daily resolution of the data.

**Pipeline.** Five parts in sequence.

1. **Encoder: a Gaussian SetConv for each variable, with a density channel.**
   - For each dynamic input d with observation mask m_d(t) (0 where the value is missing):
     - density_d(t) = Σ_k m_d(t−k) ψ_d(k);
     - signal_d(t) = Σ_k m_d(t−k) x_d(t−k) ψ_d(k) / (density_d(t) + 10⁻⁶).
   - ψ_d is a Gaussian kernel with a learnable length scale and a support of 15 days.
   - The `gss` encoder is multi-scale. Each variable is smoothed at three initial length scales, 0.25, 2 and 8 days. The 0.25-day kernel is almost an identity.
   - The density and signal channels (2 × 3 × D) are projected point-wise to 64 hidden channels.
   - Missing values (NaN) set the mask to zero, so they reach the model only through a lower density. Nothing is filled in.
   - Forcing groups (products) are flattened into one list of variables. Groups matter only for neuralhydrology's simulation of missing data during training.
2. **Static attributes.** A multilayer perceptron (27 → 64 → 2 × 64 × 7) produces scale and shift parameters for feature-wise linear modulation (FiLM; Perez et al., 2018) in every block. It is initialised so that the modulation starts as the identity.
3. **ρ: residual dilated temporal convolutional network (TCN).**
   - 7 blocks with kernel 3, 64 channels and dilations 1, 2, …, 64 (Bai et al., 2018).
   - Each block applies, in order: LayerNorm across channels at each time step; FiLM; GELU; a dilated convolution; a WaveNet-style gated activation, tanh(f)·σ(g) (van den Oord et al., 2016); LayerNorm; GELU; a second dilated convolution; dropout 0.25; and a residual connection.
4. **Decoder.**
   - A Gaussian SetConv read-out at initial scales of 0.25 and 2 days, with a 15-day support.
   - It is concatenated with the unsmoothed features (an identity path).
   - There is no normalisation before the head. Then a GELU and output dropout of 0.3.
5. **Head.**
   - A neuralhydrology regression head with softplus output, so predictions are ≥ 0. Discharge is scaled but not centred (`centering: none`).
   - `head_bias_init = 1.0`: the last layer's weights are scaled by 0.1, and its bias is set so that softplus(bias) = 1 normalised flow unit.

**Causality, receptive field and windows.**
- By default every kernel looks only at t′ ≤ t. Prediction for day t then uses the same information as the LSTM.
- The receptive field is 1 + 14 + 14 + 2·2·(1 + 2 + … + 64) = 537 days.
- Training windows are 910 days long with `predict_last_n` = 365. So each training window gives 365 targets, and each target sees its full 537-day history, exactly as at test time. The model checks this condition at start-up.

**Training.** Batch 64, 30 epochs of 500 updates, Adam with the same learning-rate schedule and gradient clipping as the LSTM, NSE\* loss, last epoch.

**Size and cost.**
- 323,536 parameters on Daymet, about the same as the 256-unit LSTM (about 300k).
- A gauged Daymet seed trained in about 25 minutes alone on an RTX 3090, against about 4.5 hours for an LSTM seed.

### 3.3 Changes from the 2020 model

**Bugs found in the 2020 notebook** (full review in `REVIEW.md`; summary in the 2026-11-20 changes note):

| # | 2020 problem | effect | fix |
|---|---|---|---|
| 1 | The encoder output had 3 channels: `UNet()` used its default `in_channels=8`, while 5 static dimensions were concatenated | All weather inputs passed through `Linear(11→3)` and a sigmoid; the model had 51,378 parameters | Per-variable SetConv with a density channel, projected to 64 channels |
| 2 | Training kept only a random subset of days in each 365-day window (50–165 context plus 150–200 target days), and forcings existed only on those days | About 23% of training days had no forcing; the test input was dense | Dense daily forcings; random hiding survives only as an optional `input_dropout`, off by default |
| 3 | Non-causal UNet in training, causal test at the window edge | Tested on a configuration it rarely saw in training | Causal SetConv and TCN; 910-day windows, so every target has its full history |
| 4 | Discharge standardised with one pooled mean and standard deviation, shifted by +1.42, gamma likelihood | High-flow basins dominated the loss; arid basins collapsed (NSE down to −28.7); the gamma's support included negative discharge | NSE\* loss, regression head; later an uncentred target with softplus output |
| 5 | Windows sampled over the concatenated table of all basins | About 6.7% of windows mixed two basins | neuralhydrology samples windows per basin |
| 6 | Rows with missing discharge deleted; time taken from the row index | Gaps in the time axis; forcings lost | All rows kept; missing targets masked |
| 7 | Normalisation statistics included the test period | Slight leakage | Training-period statistics |
| 8 | Evaluation skipped each basin's first year and every year's last day | Minor bias | Every test day predicted |
| 9 | Metric bugs (training NSE always NaN; a double back-transform) | No usable monitoring | Shared neuralhydrology metrics |
| 10 | PUB checkpoint chosen on the held-out basins | The 2020 PUB score of 0.31 is optimistic | 5-fold cross-validation, last epoch, validation on training basins only |
| 11 | Constant learning rate, no clipping, single run | Noisy; no ensemble | Learning-rate schedule, clipping, 3 seeds and ensembles |

**Architecture and training, from 2020 to `gss`:**

| setting | 2020 | v1 (`convcnp_mres`) | `gs` | `gss` |
|---|---|---|---|---|
| encoder | DeepSet → ConvDeepSet → 3 channels | SetConv per variable plus density, scale 2 d | multi-scale SetConv, scales 0.25 / 2 / 8 d | = gs |
| static attributes | DeepSet mean-pooled to 5 dimensions | MLP → FiLM in every block | same | same |
| ρ | 12-layer UNet, non-causal | 7 causal dilated residual blocks, LayerNorm, GELU | + gated blocks | = gs |
| read-out | SetConv for mean and σ | SetConv, 2 d | 0.25 / 2 d plus identity path | = gs |
| normalisation before the head | — | LayerNorm | none | = gs |
| output | gamma | linear | softplus, uncentred discharge | + `head_bias_init` 1.0 |
| loss | gamma NLL | NSE\* | NSE\* | NSE\* |
| dropout (blocks / output) | input masking, later set to 0 | 0.1 / 0.1 | 0.25 / 0.3 | = gs |
| updates per epoch × epochs | 256 tasks × up to 500, best epoch picked on validation NLL | 1000 × 30 | 500 × 30 | = gs |
| parameters | 51,378 | 235,718 | 323,536 | 323,536 |
| receptive field | whole window, both directions | 537 d, past only | 537 d | 537 d |

**Why `head_bias_init` was added.**
- A softplus head can start in its flat region and never recover: the output is constant and the training loss stays at about 0.1.
- This happened in 3 of the 15 `gs` PUB runs (folds 1, 3 and 4, seed 1) and in one tuning run with skip connections.
- With the fix, every run since has been healthy: 15 PUB, 6 gauged and all later runs.
- It changes nothing else. `gss` and `gs` 3-seed ensembles on the gauged test are 0.750 and 0.752 (paired p = 0.75).

### 3.4 ConvCNP variants for missing data and streamflow inputs

1. **Missing forcings.** Missing values enter through the density channel (section 3.2). No imputation or flag is used. neuralhydrology's `nan_handling_method` options are LSTM input-layer options and are refused by the ConvCNP. During training, missing data are simulated with neuralhydrology's `nan_step_probability` and `nan_sequence_probability`, the scheme of Gauch et al. (2025). The same setting is used for the LSTMs.
2. **Lagged streamflow (study A, `convcnp_gss_ar50`, `convcnp_gss_ar0`).**
   - Observed discharge at lag 1 (`QObs(mm/d)_shift1`, from neuralhydrology `lagged_features`) is one more input channel.
   - Missing observations go through the density channel. Unlike the autoregressive LSTM, the model's own predictions are never fed back in.
   - The model stays causal, so day t sees discharge up to t−1.
   - neuralhydrology's `random_holdout_from_dynamic_features` hides contiguous runs of the discharge input during training (mean length 5 days), at a missing fraction of 50% (AR50) or 0% (AR0). This is the sampler of Nearing et al. (2022).
   - No forcing masking is applied.
3. **Gap filling (study B, `convcnp_gss_gapfill`).**
   - **Input and window.**
     - The model is non-causal. Same-day observed discharge (`QObs(mm/d)_shift0`) is an input.
     - Its receptive field of 537 days is ±268 days around each target.
     - Windows are 910 days with `predict_last_n` = 642. So the loss region is the central 374 days, [268, 642), and every scored day has full context on both sides.
   - **Gap training** (`ConvCNP.pre_model_hook`; unit-tested). In each training window:
     - 2–6 gaps are hidden in the discharge input, with lengths drawn log-uniformly from 1–120 days and placed so that they overlap the loss region.
     - The target is set to missing everywhere except on hidden days inside the loss region, so the model is never rewarded for copying its input.
   - **Validation.** In-training validation is off, because neuralhydrology's tester never hides the discharge input and would score a copy of the target.
   - No forcing masking. All other settings are as in `gss`.
   - A denser variant (`convcnp_gss_gapfill_dense`, 4–10 gaps per window) is configured but not trained.

### 3.5 Changes to neuralhydrology

All are applied at run time by `register()` and covered by tests; 32 tests pass on CPU.
1. **The tester uses `predict_last_n` = 1 for the ConvCNP.** Without this, the stock tester silently skipped the first 364 days of every test period.
2. **The CMAL loss is masked element by element.** The stock loss returned NaN on fully masked batches.
3. **CMAL sampling uses the tester's `predict_last_n`.**
4. **Optional training extras, used only in ablations:** weight decay, exponential moving average (EMA) of the weights, a variance-matching loss term, a high-flow loss term, and a loss warm-up.
5. **Evaluation scripts.**
   - **Missing-input scenarios (`eval_missing_scenarios.py`).** For speed, ConvCNP runs are evaluated in one causal pass over each basin's full series; for a causal model this is exactly equivalent to evaluating one window per day. LSTMs use neuralhydrology's standard tester.
   - **Gap filling (`eval_gapfill.py`).** It uses one non-causal pass per basin and per gap class.

---

## 4. Experiments and results

**Common settings.** Unless stated otherwise:
- 531 basins;
- three seeds per model;
- results are 3-seed ensembles, with paired statistics over 531 basins;
- the models are the ones defined in section 3.

| § | experiment | split | forcings | models (seeds) | trained for this experiment? |
|---|---|---|---|---|---|
| 4.1 | tuning and pilots | tuning | Daymet | ConvCNP variants (mostly 1 seed), LSTM reference | yes |
| 4.2 | gauged simulation | MRes | Daymet | LSTM (6), v1 (3), `gs` (3), `gss` (6), SAC-SMA | yes |
| 4.3 | ungauged basins (PUB) | PUB | Daymet | LSTM (3 × 5 folds), `gs`, `gss` (3 × 5 folds) | yes |
| 4.4 | missing forcings, no retraining | MRes | Daymet | LSTM (5), `gss` (6) | no (evaluation only) |
| 4.5 | missing inputs, three products | Kratzert | Daymet, Maurer, NLDAS | LSTM-MM, LSTM-IR, `gss` (3 each) | yes |
| 4.6 | missing inputs, Daymet only, and no-masking control | Kratzert | Daymet | LSTM-IR, `gss`, `gss` without masking (3 each) | yes |
| 4.7 | study A: lagged streamflow | Kratzert | three products | `gss`-AR50, `gss`-AR0 (3 each); LSTMs from Nearing et al. (2022) | ConvCNP only |
| 4.8 | study B: gap filling | Kratzert | three products | `gss`-gapfill (3); others from earlier runs | ConvCNP only |

### 4.1 Tuning and pilots (tuning split)

**Setup.**
- Train WY1981–90, evaluate WY1991–95, Daymet, 531 basins.
- One change at a time, mostly with seed 1.
- Overrides are in `convcnp_runs/tune_queue*.sh`; results in `convcnp_runs/tune_*/` and `pilots_LMN_seed1_*.csv`.
- The LSTM reference on this split scores 0.760.

**From v1 to `gs`** (median NSE, seed 1 unless stated):

| step | median NSE |
|---|---|
| v1 | 0.715 |
| multi-scale encoder | 0.719 (FLV −144) |
| + non-negative output (uncentred target, softplus) | 0.724 (FLV +14) |
| + no normalisation before the head | 0.728 |
| + more dropout (0.25 / 0.3) | 0.732 |
| + gated blocks | 0.735 |
| + 500 updates per epoch = `gs` | 0.742 / 0.733 / 0.728 (seeds 1–3; mean 0.734) |

- Each step is within the single-seed noise. Single-seed median NSE varies by about ±0.007 in this setting, and by up to ±0.015 between near-identical configurations.
- Only the cumulative gain, from 0.715 to about 0.734, is clear.
- The non-negative output had one clear, separate effect: it removed a large low-flow bias.

**Variants tried and dropped before 2026-11-20:**

| variant | change | median NSE | outcome |
|---|---|---|---|
| `ms_indrop` | input dropout 0.1 | 0.650 | worse; Alpha 0.68, FHV −35 |
| `ms_c128` | 128 channels | 0.725 | no gain; FLV −42 |
| `nn_nonorm_c128` | 128 channels, less regularisation | 0.705 | worse (overfits) |
| `nn_nonorm_drop_nobn` | no LayerNorm inside blocks | — | collapsed to a constant output; stopped |
| `nn_pln60` | 60-day target blocks, batch 256 | 0.726 | within noise |
| `I_ema` | EMA of weights, 0.999 | 0.739 / 0.733 / 0.730 | no gain |
| `I_wd` | AdamW, weight decay 0.05 | 0.728 / 0.729 / 0.733 | no gain |
| `I_skip` | WaveNet skip outputs | 0.738 / 0.745 / −0.331 | seed 3 dead from the start; led to `head_bias_init` |
| `J_skipfix` | skip + `head_bias_init` | 0.735 / 0.735 / 0.743 | healthy, no gain |
| `I_all` | EMA + skip + weight decay | 0.740 / 0.734 / 0.737 | no gain |
| `K_cmal` | CMAL head (3 components, Klotz et al., 2022) | 0.694 / 0.681 / 0.695 | worse; Alpha 0.68, FHV −32; 90% interval coverage 0.81 |

**Pilots L, M and N (2026-11-20), aimed at over-smoothing.**
- The reference is `gss` in the same code: seeds 0.726 / 0.726 / 0.723, ensemble 0.753.
- "Wins" is the share of basins beating `gss` seed 1, with the Wilcoxon p-value.

| variant | change | median NSE | Alpha | FHV | FLV | wins (p) |
|---|---|---|---|---|---|---|
| LSTM | — | 0.760 | 0.93 | −8.6 | −3.5 | 67% (3e-13) |
| `gss` | — | 0.726 | 0.87 | −15.1 | +12.8 | — |
| `lowdrop` | dropout 0.1 / 0.1 | 0.707 | 0.87 | −14.1 | +3.1 | 38% (6e-10) |
| `sharp`, seeds 1–3 | read-out scale 0.25 d only | 0.736 / 0.737 / 0.725 | 0.87–0.88 | −12.6 to −14.4 | +6 to +12 | 52–55% (0.11–0.89) |
| `sharp` ensemble | | 0.760 (`gss` 0.753) | 0.86 | −14.9 | +17.6 | 55% (0.08) |
| `alpha03` | variance-matching loss, λ = 0.3 | 0.697 | 0.90 | −11.7 | −26 | 31% (2e-19) |
| `alpha10` | λ = 1.0 | 0.636 | 0.95 | −8.2 | −64 | 15% (2e-58) |
| `alpha01` | λ = 0.1 | −0.331 | 0 | −100 | +81 | dead from the start |
| `alpha01w` | λ = 0.1 + 500-step loss warm-up | 0.726 | 0.88 | −14.3 | −14 | 43% (2e-4) |
| `peak05` | high-flow loss μ = 0.5 (top 5% of each window) + warm-up | 0.689 | 0.93 | −10.8 | −36 | 26% (1e-32) |
| `peak20` | μ = 2.0 | 0.643 | 0.96 | −11.2 | −57 | 13% (2e-60) |
| `local2` | 2 undilated blocks before the dilated stack | 0.732 | 0.87 | −14.7 | +4 | 48% (0.65) |
| `local2sharp` | `local2` + `sharp` | 0.731 | 0.88 | −13.6 | +7 | 51% (0.92) |

**Notes from the pilots.**
- **Variance and high-flow loss terms.** These raise Alpha to the LSTM's level or above, but only by inflating variance everywhere. Low flows are pulled far too low and NSE falls. They move the model along the trade-off between NSE and Alpha rather than improving it.
- **`sharp`.** It is the only variant with the same sign of gain for all three seeds (about +0.007), and it is not significant. The owner's stopping rule was a gain of at least 0.010 and more than 55% basin wins; no variant met it, so tuning for plain simulation stopped.

### 4.2 Gauged basins (MRes split)

**Setup.**
- Train WY1981–95, test WY1996–2010, Daymet, 531 basins.
- Each test day is predicted as the last step of its own window.
- LSTM seeds 1–6 (seeds 4–6 added as a control on 2026-11-21); ConvCNP v1 and `gs`, 3 seeds each; `gss`, 6 seeds (2026-11-23).
- Sources: `summary_final_mres.csv`, `mres_mixed_control_*.csv`, `mres_gss6_*.csv`.

| model | median NSE | mean NSE | basins with NSE < 0 | KGE | Alpha | FHV | FLV |
|---|---|---|---|---|---|---|---|
| LSTM, single seeds 1–5 | 0.736–0.740 | 0.68–0.69 (seeds 4–5) | 3–6 | 0.78–0.79 | 0.94–0.96 | −4 to −6 | — |
| LSTM seed 6 (diverged late in training) | 0.722 | −0.831 | 82 | 0.748 | 0.93 | −6.9 | — |
| LSTM ensemble, seeds 1–3 | 0.764 | 0.719 | 3 | 0.792 | 0.94 | −5.7 | +11.1 |
| LSTM ensemble, seeds 1–5 | 0.772 | 0.724 | 2 | 0.795 | 0.94 | −5.6 | +17.6 |
| ConvCNP v1 ensemble (3) | 0.736 | 0.697 | 3 | 0.754 | 0.84 | −15.1 | −18.5 |
| ConvCNP `gs` ensemble (3) | 0.752 | 0.709 | 2 | 0.771 | 0.88 | −12.5 | +28.7 |
| ConvCNP `gss`, single seeds 1–6 | 0.715–0.726 | 0.67–0.68 | 3–5 | 0.76 | 0.87–0.91 | −9 to −13 | +21 to +26 |
| ConvCNP `gss` ensemble (3) | 0.750 | 0.708 | 2 | 0.771 | 0.867 | −13.2 | +32.6 |
| ConvCNP `gss` ensemble (6) | 0.757 | 0.713 | 2 | 0.770 | 0.875 | −12.5 | +34.3 |
| mixed: LSTM 3 + `gs` 3 | 0.774 | 0.734 | 2 | 0.792 | 0.90 | −9.9 | +27.4 |
| mixed: LSTM 5 + `gss` 6 | 0.777 | 0.735 | 2 | 0.793 | 0.898 | −9.8 | +31.1 |
| SAC-SMA (Newman et al., 2015) | 0.665 | 0.572 | 11 | 0.734 | 0.87 | −10.7 | +28.4 |
| ConvCNP 2020 (424 basins, 1 run) | 0.37 | −0.14 | — | — | — | — | — |

**Paired comparisons:**

| comparison | wins | median difference | p |
|---|---|---|---|
| `gs` 3 vs LSTM 3 | 34.8% | −0.012 | 3e-10 |
| `gss` 6 vs LSTM 3 | 38.6% | −0.009 | 2e-5 |
| `gss` 6 vs LSTM 5 | 32.6% | −0.013 | 5e-13 |
| `gss` 6 vs `gss` 3 | 70.1% | +0.005 | 2e-23 |
| mixed (LSTM 3 + `gs` 3) vs LSTM 3 | 69.9% | +0.007 | 2e-23 |
| mixed (LSTM 3 + `gs` 3) vs LSTM 5 | 61.0% | +0.0035 | 6e-9 |
| mixed (LSTM 5 + `gss` 6) vs LSTM 5 | 61.0% | +0.005 | 7e-12 |

**Observations.**
1. **The rewrite closed most of the 2020 gap.** The 2020 model scored 0.37 (on 424 basins, which is not strictly comparable). The rewritten v1 scores 0.736, and tuning adds 0.016.
2. **The LSTM leads at equal seed count.** The ConvCNP wins in about 35–39% of basins.
3. **A compute-matched test failed.** Six ConvCNP seeds, at about a quarter of the LSTM's GPU time for three seeds, reach 0.757, which is below the 3-seed LSTM (0.764).
4. **The ConvCNP over-smooths consistently.** Its Alpha is 0.87–0.88 against 0.94, its FHV is −12 to −13 against −6, and extra seeds do not change this.
5. **Mixing gives a real but small gain.** The 0.764 → 0.774 improvement is mostly an ensemble-size effect. Against the fairest control (5 LSTM seeds), the mixing gain is +0.0035 to +0.005 per basin, winning in 61% of basins.
6. **LSTM seed 6 diverged late in training.** Its training loss rose to 0.47 at epoch 30, against about 0.008 normally. It is left out of the fair control. No `gss` run has failed.

### 4.3 Ungauged basins (PUB)

**Setup.**
- 5-fold cross-validation over basins on the MRes periods.
- The training-basin validation set is used for monitoring only; the last epoch is scored on held-out basins and the results are pooled over 531 basins.
- LSTM and `gss`, 3 seeds per fold (15 runs each); `gs` runs as well, three of which were dead from the start.
- Sources: `pub_lstm3_vs_gss3_summary.csv`, `pub_mixed_*.csv`.

| model | median NSE | mean NSE | NSE < 0 | KGE | Alpha | FHV |
|---|---|---|---|---|---|---|
| LSTM, single seeds | 0.635 / 0.644 / 0.644 | 0.437 / 0.452 / 0.324 | 46 / 38 / 48 | 0.66–0.68 | 0.93–0.95 | −6 to −8 |
| LSTM ensemble (3) | 0.679 | 0.478 | 37 | 0.694 | 0.91 | −8.5 |
| `gss`, single seeds | 0.626 / 0.630 / 0.616 | 0.446 / 0.465 / 0.458 | 37 / 38 / 36 | 0.65–0.66 | 0.87 | −12 |
| `gss` ensemble (3) | 0.664 | 0.519 | 33 | 0.675 | 0.85 | −13.7 |
| mixed: LSTM 3 + `gss` 3 | 0.696 | 0.537 | 29 | 0.694 | 0.88 | −12.6 |
| `gs` ensemble (3 dead runs included) | 0.595 | 0.525 | 19 | 0.491 | 0.66 | −33.1 |
| ConvCNP 2020 (one split, checkpoint chosen on test basins) | 0.31 | — | — | — | — | — |

**Paired comparisons:**

| comparison | wins | median difference | p |
|---|---|---|---|
| `gss` 3 vs LSTM 3 | 40.7% | −0.014 | 0.003 |
| mixed vs LSTM 3 | 63.7% | +0.011 | 2e-15 |
| mixed vs `gss` 3 | 78.9% | +0.026 | 6e-33 |

**Observations.**
- The LSTM leads on median NSE. The ConvCNP has the higher mean NSE (0.519 against 0.478) and fewer basins below zero (33 against 37), so its worst basins are less bad.
- There is no 5-seed LSTM control for PUB, so the mixed gain here is partly an ensemble-size effect.

### 4.4 Missing forcings without retraining (evaluation only)

**Setup.**
- Models: the gauged MRes models with no retraining, LSTM seeds 1–5 and `gss` seeds 1–6, evaluated on the MRes test decade.
- Masking: for each basin, all forcings of a day are hidden with probability p ∈ {10, 30, 50%}, independently by day, before normalisation. The mask seed is 0 and is identical for both models.
- Fill options:
  - `ffill`: gaps filled with the last observed value. The LSTM needs this, since it has no missing-data handling.
  - `nan`: gaps left missing for the ConvCNP's density channel.
- Script: `scripts/eval_missing_forcings.py`. Sources: `missing_forcing_*.csv`.

| hidden days | LSTM 5, filled | `gss` 6, filled | `gss` 6, missing values | NSE < 0 (LSTM / `gss` filled / `gss` missing) |
|---|---|---|---|---|
| 0% | 0.772 | 0.757 | — | 2 / 2 / — |
| 10% | 0.657 | 0.653 | 0.634 | 8 / 7 / 7 |
| 30% | 0.347 | 0.395 | 0.315 | 100 / 61 / 94 |
| 50% | −0.096 | 0.057 | 0.018 | 310 / 228 / 249 |

| comparison | 10% | 30% | 50% |
|---|---|---|---|
| `gss` filled vs LSTM filled: wins, median difference, p | 47.6%, −0.002, 0.41 | 68.0%, +0.027, 5e-26 | 77.6%, +0.077, 5e-44 |
| `gss` missing vs LSTM filled | 37.3%, −0.016, 5e-8 | 41.2%, −0.026, 1e-4 | 55.7%, +0.046, 0.001 |

**Observations.**
- With the same filled inputs, the ConvCNP degrades more slowly than the LSTM from 30% hidden days onwards.
- Passing gaps through the density channel without having trained on gaps is worse than filling them, at every level. The density values are then outside anything seen in training.
- Both models are unusable from 30% hidden days onwards. This motivated retraining with simulated missing data (sections 4.5 and 4.6).

### 4.5 Missing inputs with retraining: three forcing products (Gauch et al. 2025 setup)

**Setup.**
- Kratzert periods (train WY2000–08, test WY1990–99), 531 basins.
- Daymet, Maurer and NLDAS as three feature groups, plus 27 attributes.
- **Training-time masking** (Gauch et al., 2025; their tuning and experiment-2 setting), per group:
  - p_time = 0.1: each day, each product is independently missing with probability 0.1;
  - p_sequence = 0.1: each sample, each product is missing for the whole sequence with probability 0.1, keeping at least one product [CHECK: exact neuralhydrology semantics of `nan_sequence_probability`].
- **Models**, 3 seeds each (configs `lstm_mm_multi`, `lstm_ir_multi`, `convcnp_gss_multi`):
  - LSTM-MM (35 epochs);
  - LSTM-IR (30 epochs);
  - `gss`, with missing values through the density channel.
- **Test scenarios.** Masks are fixed per (basin, day, product) with mask seed 0 and are identical for all models.
  - `indep_pXX`: each product missing independently on XX% of days.
  - `allday_pXX`: all products missing together on XX% of days.
  - `keep_*`: the remaining products missing for the whole test period.
- Script: `eval_missing_scenarios.py`. Sources: `missing_scenarios_*.csv`.
- **Run.** 9 training runs on GPU 0, at most 3 in parallel, 2026-11-23 22:40Z to 2026-11-24 10:29Z.

**Median NSE, 3-seed ensembles:**

| scenario | `gss` | LSTM-MM | LSTM-IR |
|---|---|---|---|
| clean | 0.768 | 0.807 | 0.807 |
| independent 10% | 0.761 | 0.798 | 0.798 |
| independent 30% | 0.725 | 0.762 | 0.765 |
| independent 50% | 0.610 | 0.677 | 0.689 |
| all products, 10% of days | 0.747 | 0.745 | 0.764 |
| all products, 30% | 0.664 | 0.571 | 0.652 |
| all products, 50% | 0.494 | 0.350 | 0.493 |
| Daymet only | 0.720 | 0.766 | 0.746 |
| Maurer only | 0.710 | 0.745 | 0.729 |
| NLDAS only | 0.702 | 0.745 | 0.725 |
| Daymet + Maurer | 0.750 | 0.792 | 0.786 |
| Daymet + NLDAS | 0.760 | 0.794 | 0.796 |
| Maurer + NLDAS | 0.753 | 0.791 | 0.783 |

On clean data the mean of single seeds is 0.755 for `gss`, 0.793 for LSTM-MM and 0.791 for LSTM-IR.

**Paired comparisons** (wins, with p):

| scenario | `gss` vs LSTM-MM | `gss` vs LSTM-IR |
|---|---|---|
| clean | 15% (4e-56) | 16% (2e-53) |
| independent 10 / 30 / 50% | 14 / 18 / 22% | 16 / 17 / 15% |
| all products 10% | 52% (0.26) | 33% (3e-20) |
| all products 30% | 84% (2e-59) | 57% (0.001) |
| all products 50% | 86% (8e-65) | 57% (0.002) |
| one or two products kept | 14–21% | 17–31% |

**Observations.**
- **When at least one product is present each day,** the ConvCNP loses clearly to both LSTMs: 0.03–0.08 lower median NSE, winning in 14–31% of basins.
- **On days with all products missing (30–50% of days),** the ConvCNP is far better than masked mean, which has nothing to average on those days. It is only marginally better than input replacing: 57% of basins, about +0.01 in median NSE.
- **Three products help the ConvCNP less than the LSTM.** Its clean gap to the LSTM is 0.039 here, against 0.007–0.012 on Daymet with the MRes split. Part of this is the cost of the training masking (section 4.6).
- **Agreement with Gauch et al. (2025).** Their findings are only in figures, so the comparison is qualitative.
  - Their experiment 1 finds no clear winner among the LSTM methods, and ours decay in step.
  - In their experiment 2, input replacing tends to be worst when products are dropped, and in our `keep_*` scenarios it is below masked mean.
- **Differences from Gauch et al. (2025).**
  - We train one model per method, at p = 0.1, rather than a sweep over p_time. So our 30% and 50% test levels are outside the training distribution for all models.
  - The all-products scenario is ours.
  - We did not train their baselines that use only the available products.
  - Their results are in figures and in a Zenodo archive (doi 10.5281/zenodo.15008460), which we have not fetched.

### 4.6 Missing inputs with retraining: Daymet only, and the no-masking control

**Setup.**
- Kratzert periods, Daymet only (one feature group), 531 basins.
- Training masking: p_time = p_sequence = 0.1. With a single group, neuralhydrology never drops the whole sequence, so only the random 10% of days act.
- **Models**, 3 seeds each:
  - LSTM-IR (`lstm_ir_daymet`);
  - `gss` (`convcnp_gss_daymet`);
  - the control `gss` without masking (`convcnp_gss_daymet_nomask`, p = 0).
- **Tests.** Masks are fixed per (basin, day), mask seed 0.
  - `random_pXX`: XX% of days missing at random.
  - `gaps_pXX`: XX% of days missing in contiguous gaps of geometric length, mean 10 days.
- **Runs.** Main runs 2026-11-25 06:42–12:50Z; control 16:18–17:50Z.
- Sources: `missing_daymet_*.csv`, `missing_daymet_nomask_*.csv`.

**Median NSE, 3-seed ensembles:**

| scenario | `gss`, no masking | `gss`, masked | LSTM-IR |
|---|---|---|---|
| clean | 0.738 | 0.691 | 0.761 |
| random 10% | 0.663 | 0.652 | 0.713 |
| random 30% | 0.351 | 0.559 | 0.602 |
| random 50% | −0.066 | 0.429 | 0.475 |
| gaps 10% | 0.540 | 0.614 | 0.685 |
| gaps 30% | 0.117 | 0.480 | 0.547 |
| gaps 50% | −0.381 | 0.320 | 0.380 |

Clean single-seed means are 0.714 for `gss` without masking, 0.672 for masked `gss` and 0.739 for LSTM-IR.

**Paired comparisons:**
- **Masked `gss` against LSTM-IR.** `gss` wins in 13–27% of basins in every scenario (p ≤ 2e-23). Its median difference narrows from −0.059 on clean data to −0.027 at 50% random missing. It stays at −0.05 to −0.065 for contiguous gaps.
- **`gss` without masking against masked `gss`.** It wins in 85% of basins on clean data, and in 10% or fewer from 30% missing onwards.
- **`gss` without masking against LSTM-IR.** It wins in 26% of basins on clean data (p = 6e-27).

**Observations.**
- **The LSTM with input replacing beats the ConvCNP in every scenario.**
- **The masking costs the two models differently.**
  - The masking costs the ConvCNP about 0.04 clean NSE (single seeds: 0.714 against 0.672).
  - It costs the LSTM nothing measurable: masked LSTM-IR scores 0.739, against 0.744 / 0.750 / 0.738 for the unmasked LSTM seeds on the same periods.
  - Even the unmasked ConvCNP is below the masked LSTM.
- **Without masking, the ConvCNP collapses beyond about 10% missing days.** The density channel gives robustness only when the model sees gaps in training. This matches section 4.4.
- **The evidence that the ConvCNP suffers from the shorter training record is weak.** The 2026-11-25 note attributed the rest of the clean gap to the 9-year training period, comparing with "about 0.75" on the MRes split; that figure is an ensemble value. Like for like, single seeds give 0.714 here (unmasked) against 0.715–0.726 on the 15-year MRes split, while the LSTM is at about 0.74 on both. So the single-seed gap to the LSTM grows only from about 0.015 to about 0.025. No experiment separates training length from the different test decade.

### 4.7 Study A: lagged streamflow as input (Nearing et al. 2022 setting)

**Setup.**
- Kratzert periods, three forcing products (15 inputs), 27 attributes, 531 basins.
- Observed discharge at lag 1 as an input.
- **ConvCNP models**, 3 seeds each, no forcing masking:
  - `gss`-AR50: trained with 50% of the discharge input hidden in contiguous runs of mean length 5 days (Nearing's main setting);
  - `gss`-AR0: trained with nothing hidden.
- **Reference without streamflow:** gss-noQ, which is `convcnp_gss_multi` from section 4.5 (trained with 10% forcing masking).
- **LSTMs.** On the owner's instruction they were not retrained. We compare with published values from Nearing et al. (2022). Their 0% row is exact, from `zero_lag_zero_holdout_metrics_table.txt` in their repository; the other rows were read from their Fig. 1 by eye, to about ±0.005.
- **Tests.** 0, 25, 50, 75 and 100% of the discharge input missing, in runs of mean length 5 days. Masks are fixed per (basin, fraction) and identical for all runs.
- **Run.** 2026-11-26 06:43–10:12Z; 12 jobs, 3 in parallel, 5.7 GB.
- Source: `ar_*.csv` (`ensemble_report.py ar`).

**Median NSE, mean of single seeds** (the closest match to Nearing's single models):

| test: discharge missing | `gss`-AR50 | `gss`-AR0 | gss-noQ | Nearing AR-LSTM, 50% training holdout | Nearing AR-LSTM, 0% holdout | Nearing simulation LSTM |
|---|---|---|---|---|---|---|
| 0% | 0.844 | 0.863 | 0.755 | 0.872 | 0.879 | 0.796 |
| 25% | 0.831 | 0.738 | 0.755 | ~0.861 | ~0.857 | 0.796 |
| 50% | 0.816 | 0.631 | 0.755 | ~0.845 | ~0.831 | 0.796 |
| 75% | 0.800 | 0.514 | 0.755 | ~0.835 | ~0.798 | 0.796 |
| 100% | 0.698 | 0.229 | 0.755 | ~0.757 | ~0.553 | 0.796 |

The 3-seed ensembles for `gss`-AR50 are 0.861 / 0.851 / 0.838 / 0.822 / 0.722 and for `gss`-AR0 0.874 / 0.759 / 0.655 / 0.544 / 0.251; gss-noQ is 0.768.

**Paired comparisons, 3-seed ensembles** (wins):
- AR50 against AR0: 23% at 0% missing, then 93–99.8% from 25% missing onwards.
- AR50 against gss-noQ: 92–97% up to 75% missing, and 35% at 100%.
- All p < 1e-17, except AR0 against gss-noQ at 25% missing (p = 0.02).

**Observations.**
- **Lagged streamflow helps the ConvCNP a lot, but the gain is inflated.** On clean data it adds +0.089 (AR50) and +0.108 (AR0) over gss-noQ; Nearing's LSTM gains +0.076 and +0.083. But gss-noQ was trained with forcing masking and the AR models were not. On Daymet the same masking cost about 0.04 (section 4.6), so a like-for-like gain is plausibly +0.05 to +0.07. This is inferred: we have no unmasked three-product gss-noQ.
- **The ConvCNP stays below the LSTM in absolute terms.** `gss`-AR50 is 0.03–0.035 below Nearing's AR-LSTM at 0–75% missing, and about 0.06 below at 100%. `gss`-AR0 is 0.016 below on clean data. These AR models are trained as in Nearing's setting apart from the architecture, so no masking handicap explains the gap. The comparison is with published single models, so it is not paired by basin.
- **The density channel does not replace training with missing streamflow.** `gss`-AR0 degrades much faster than Nearing's AR0 LSTM: 0.631 against ~0.831 at 50% missing. Beyond 25% missing it falls below gss-noQ. The autoregressive LSTM substitutes its own prediction; the ConvCNP has only the density channel.
- **At 100% missing,** `gss`-AR50 is 0.057 below gss-noQ, against 0.039 for Nearing's LSTM relative to its simulation model. A separate model without streamflow is still better when no observations exist.
- **Caveats.**
  - Nearing's test masks are random; ours are fixed.
  - We have not opened Nearing's per-basin results, so a basin-by-basin comparison is not possible.

### 4.8 Study B: filling gaps in observed streamflow

**Setup.**
- Kratzert periods, three forcing products, 27 attributes, 531 basins.
- **Model:** `gss`-gapfill (section 3.4), 3 seeds. Trained 2026-11-26 10:48–11:24Z, 3 jobs in parallel, 1.86 GB each.
- **Test gaps.** For each basin and gap length (7, 30 or 90 days):
  - non-overlapping gaps covering about 15% of test days;
  - each gap has at least 7 observed days on each side and at least half its own days observed;
  - gaps are seeded by crc32 of (seed, basin, length) and are identical for every method.
- Scores use the observed days inside gaps only.
- **Comparison methods.** No LSTM was trained for this study; all come from earlier runs.
  - Linear and log-linear interpolation across each gap.
  - LSTM-IR simulation: the 3-seed ensemble from section 4.5, with no streamflow input.
  - The same simulation rescaled with the observed/simulated volume ratio over the 7 days before and after the gap, interpolated linearly across it. This is a common practical method and our "bias correction". The term "rescaled" below means this correction.
  - gss-noQ (section 4.5).
  - `gss`-AR50 from study A, which sees streamflow up to the day before only.
- Script: `eval_gapfill.py`. Sources: `gapfill_*.csv`, `gapfill_report*.txt`.

**Median NSE on gap days, 3-seed ensembles:**

| method | 7-day gaps | 30-day gaps | 90-day gaps |
|---|---|---|---|
| linear interpolation | 0.282 | 0.084 | 0.009 |
| log-linear interpolation | 0.373 | 0.120 | 0.025 |
| gss-noQ | 0.798 | 0.784 | 0.774 |
| LSTM-IR | 0.829 | 0.811 | 0.808 |
| LSTM-IR, rescaled | 0.856 | 0.826 | 0.791 |
| `gss`-AR50 (past only) | 0.848 | 0.806 | 0.751 |
| `gss`-gapfill | 0.836 | 0.811 | 0.805 |
| `gss`-gapfill, rescaled | 0.854 | 0.801 | 0.769 |
| gss-noQ, rescaled | 0.841 | 0.797 | 0.759 |

Single-seed means for `gss`-gapfill are 0.823 / 0.793 / 0.784. The median absolute volume error per gap is 0.10 / 0.12 / 0.15 for the rescaled LSTM-IR and 0.22 / 0.20 / 0.17 for `gss`-gapfill.

**Paired comparisons: `gss`-gapfill against each method** (wins, median difference, p):

| against | 7-day | 30-day | 90-day |
|---|---|---|---|
| LSTM-IR, rescaled | 36%, −0.020 (4e-7) | 44%, −0.010 (0.07) | 50%, 0.000 (0.46) |
| LSTM-IR | 61%, +0.010 (2e-7) | 48%, −0.002 (0.53) | 40%, −0.011 (7e-7) |
| `gss`-AR50 | 41%, −0.007 (7e-6) | 54%, +0.004 (0.09) | 65%, +0.027 (1e-13) |
| gss-noQ | 78%, +0.038 | 67%, +0.023 | 58%, +0.013 |

**Rescaling check (2026-11-26, CPU only).** The LSTM's rescaling was applied unchanged to both ConvCNPs, and each was compared with the rescaled LSTM.
- Rescaled `gss`-gapfill wins in 38–44% of basins (NSE, p ≤ 0.02); rescaled gss-noQ wins in 31–35% (p ≤ 7e-11).
- Rescaling helps the model without streamflow most: +0.041 NSE on 7-day gaps, against +0.025 for the LSTM and +0.015 for `gss`-gapfill.
- It does nothing for `gss`-gapfill on 30-day gaps.
- It lowers NSE for all three models on 90-day gaps.

**Mixed ensembles (hybrid; 2026-11-26 night, CPU only).**
- **Method.** A fixed 50/50 average of the 3-seed ConvCNP ensemble and the 3-seed LSTM-IR ensemble, with no fitted weights. Each mix is compared with the matched LSTM: raw with raw, rescaled with rescaled.
- **Control.** The same mix built with gss-noQ. If ensemble size alone explained the gain, the control would gain as much.

| method | 7-day | 30-day | 90-day |
|---|---|---|---|
| LSTM-IR | 0.829 | 0.811 | 0.808 |
| mix: `gss`-gapfill + LSTM-IR | 0.844 | 0.826 | 0.822 |
| control mix: gss-noQ + LSTM-IR | 0.823 | 0.814 | 0.804 |
| LSTM-IR, rescaled | 0.856 | 0.826 | 0.791 |
| mix, rescaled | 0.866 | 0.825 | 0.797 |
| control mix, rescaled | 0.860 | 0.824 | 0.792 |

**Paired results for the mixes:**
- **Raw mix against LSTM-IR.** NSE wins in 79 / 70 / 62% of basins (p = 3e-40, 5e-19, 2e-6). KGE wins in 70 / 62 / 51% (the last with p = 0.8).
- **Rescaled mix against the rescaled LSTM-IR.** NSE wins in 63 / 57 / 60% (p ≤ 1e-5). KGE is tied (48–52%, p ≥ 0.3).
- **The control mixes.** They win in 42–54% of basins on NSE, with median differences of 0.004 or less.

**Validation reference for later selection.** On WY1981–89, `gss`-gapfill scores 0.823 / 0.808 / 0.776 as an ensemble and 0.808 / 0.791 / 0.755 as a mean of single seeds, on the same three gap lengths. No variant has been compared against it yet.

**Observations.**
- **The ConvCNP alone does not beat the practical standard.** On 7-day gaps the rescaled LSTM simulation is clearly better and has half the volume error. On 30- and 90-day gaps the two are statistically tied. The plain LSTM simulation, which never sees streamflow, beats `gss`-gapfill on 90-day gaps.
- **The ConvCNP uses the surrounding streamflow only weakly.**
  - It gains 0.013–0.038 over gss-noQ, and part of that is gss-noQ's masking handicap.
  - In a check on 3 basins (seed 1), hiding all streamflow instead of only the gaps lowered gap-day NSE by only 0.03–0.07.
  - On 7-day gaps the past-only `gss`-AR50 is better.
- **Two-sided context pays only for long gaps.** On 90-day gaps `gss`-gapfill beats `gss`-AR50 by +0.027 (65% of basins).
- **The streamflow context the ConvCNP extracts is complementary to the LSTM.** The mix gains consistently and the control does not. The mechanism is not established. One interpretation is that `gss`-gapfill carries the information from the gap edges, while the LSTM carries a better base simulation.
- **Hybrid caveats.**
  - The mix uses LSTM output, so it is a hybrid.
  - Its pure-LSTM counterpart, a bidirectional LSTM with the same gap training, has not been trained.
  - The control member is weaker than `gss`-gapfill for reasons beyond streamflow, since it was trained with forcing masking.

### 4.9 Compute and operations

**Hardware.** All training ran on GPU 0 of the shared landau machine (RTX 3090), at most 3 jobs in parallel and a total of about 5.4–5.7 GB.

**Approximate cost per seed:**

| model and setting | time |
|---|---|
| ConvCNP, Daymet | about 25 min alone |
| ConvCNP, three products | about 1.3–1.4 h in parallel |
| ConvCNP, gap filling | about 36 min, 3 in parallel |
| LSTM (cuDNN, hidden 256) | about 4.5–5.5 h |

The missing-forcing evaluation of section 4.4 took 6 h 40 min of wall time, mostly CPU data loading.

---

## 5. Discussion and conclusions (working notes)

**Where the ConvCNP stands.**
- **The rewrite closed most of the 2020 gap.** Fixing the data pipeline and making the model causal took it from 0.37 to 0.736. Tuning added about 0.016.
- **The remaining gap to the LSTM is small but consistent:** about 0.01–0.015 median NSE in gauged and ungauged basins, at equal seed count, with the LSTM better in 60–65% of basins.
- **The gap is about twice as large in the later studies.** On the Kratzert periods with three forcing products it is 0.03–0.04, partly because of the masking cost.

**Over-smoothing seems to be structural (interpretation).**
- The ConvCNP's Alpha stays at about 0.87 against 0.94 for the LSTM. Its peaks are underestimated by about twice as much as the LSTM's.
- More seeds, less dropout, sharper read-out kernels, short-range blocks, and variance and peak losses did not move the trade-off between NSE and Alpha.
- **Candidate explanations, none tested:**
  - Convolutional features are linear filters of the input history with fixed dilation patterns. An LSTM can switch state (for example, snowpack filled or empty) more sharply.
  - The Gaussian read-out smooths in time by construction.
  - The receptive field of 537 days with a fixed pattern of dilations may average over events.

**The density channel is not, by itself, a missing-data method.**
- In three separate tests, a ConvCNP that had not seen gaps in training did worse with missing values passed through the density channel than with filled values, or collapsed:
  - missing forcings without retraining (section 4.4);
  - Daymet without masking (section 4.6);
  - `gss`-AR0 on missing streamflow (section 4.7).
- Training with simulated gaps fixes this. But on clean data it costs the ConvCNP about 0.04 NSE and the LSTM with input replacing nothing measurable.
- **Why the cost differs is not known.** One hypothesis is that the LSTM's flag input lets it separate the two regimes, whereas the ConvCNP must use the same filters for densities near 1 and below 1.

**Where the ConvCNP was better.**
1. **Filled forcings at high missing rates without retraining** (section 4.4).
2. **Aligned missing days across products, against masked mean** (section 4.5). Against input replacing this edge is small: 57% of basins.
3. **Fewer failing basins in PUB.**
4. **Training stability.** No dead or diverged runs since `head_bias_init`, against one diverged LSTM seed in six.
5. **Training cost.** About 3–10 times cheaper per seed, depending on the setting.

None of these is a like-for-like win on a standard benchmark.

**Streamflow context.**
- The ConvCNP clearly uses lagged streamflow (study A), but less effectively than the autoregressive LSTM.
- In gap filling it does not exploit the two-sided context well enough to beat a rescaled simulation.
- The mixed-ensemble result (section 4.8) shows that the information it does extract is useful and not already in the LSTM. This is the most promising thread for the ConvCNP. It should be reported as a hybrid until a pure-LSTM counterpart (a bidirectional LSTM with gap training) has been tested.

**Mixing.** Fixed equal-weight mixes of LSTM and ConvCNP ensembles were better than the matched LSTM ensemble in every setting tested: gauged, PUB and gap filling. The gauged and PUB gains are small, at +0.004 to +0.011 per basin, and partly reflect ensemble size. The gap-filling gain has a control that rules out ensemble size.

**Limitations.**
- **Tuning.** All ConvCNP tuning was done on the Daymet MRes tuning split. The three-product and streamflow variants inherit `gss` settings that were not re-tuned, and neither were the LSTMs.
- **Seeds.** Single-seed pilots cannot resolve differences of about 0.01.
- **Study A is not paired by basin.** It compares with published values, partly read from figures.
- **Study B has no external benchmark.** Its baselines are our own choices; the strongest deep-learning competitor (a bidirectional LSTM) is missing, and donor-gauge methods were not tried.
- **Sensitivity to the training record is unresolved.** The difference between the 9-year and 15-year splits is small for single ConvCNP seeds (section 4.6), and the splits also differ in test decade.

**Candidate next steps.** None has been launched; GPU runs need the owner's go typed in the camels session.
1. **`convcnp_gss_gapfill_dense`** (4–10 gaps per window). Selection on validation only; about 40 min for 3 seeds.
2. **A ConvCNP given the LSTM simulation as an input (hybrid).** It would learn only the correction from the surrounding observations. It needs LSTM-IR predictions for the training and validation periods first.
3. **A bidirectional LSTM with the same gap training.** This is the fair pure-LSTM competitor for study B: about 1 day of engineering plus 15–18 GPU-hours.
4. **An unmasked three-product gss-noQ.** It would remove the masking confound in studies A and B; about 1.4 h per seed.
5. **Study C (short records, a neural-process context encoder over a basin's short record).** Planned in the 2026-11-25 note, not approved. It is the closest to the original motivation for neural processes.

---

## References

Checked against Crossref, arXiv or proceedings records on 2026-11-27. Fields marked [CHECK] could not be confirmed there.

- Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P. (2017). The CAMELS data set: catchment attributes and meteorology for large-sample studies. Hydrology and Earth System Sciences, 21, 5293–5313.
- Bai, S., Kolter, J. Z. and Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271.
- Garnelo, M., et al. (2018). Conditional neural processes. Proceedings of the 35th International Conference on Machine Learning, PMLR 80, 1704–1713.
- Gauch, M., Kratzert, F., Klotz, D., Nearing, G., Cohen, D., Gilon, O., et al. (2025). How to deal w___ missing input data. Hydrology and Earth System Sciences, 29, 6221–6235.
- Gordon, J., Bruinsma, W. P., Foong, A. Y. K., Requeima, J., Dubois, Y. and Turner, R. E. (2020). Convolutional conditional neural processes. International Conference on Learning Representations (ICLR).
- Gupta, H. V., Kling, H., Yilmaz, K. K. and Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: implications for improving hydrological modelling. Journal of Hydrology, 377, 80–91.
- Klotz, D., et al. (2022). Uncertainty estimation with deep learning for rainfall–runoff modeling. Hydrology and Earth System Sciences, 26, 1673–1693.
- Kratzert, F., Klotz, D., Brenner, C., Schulz, K. and Herrnegger, M. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. Hydrology and Earth System Sciences, 22, 6005–6022.
- Kratzert, F., et al. (2019a). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. Hydrology and Earth System Sciences, 23, 5089–5110.
- Kratzert, F., et al. (2019b). Toward improved predictions in ungauged basins: exploiting the power of machine learning. Water Resources Research, 55, 11344–11354.
- Kratzert, F., et al. (2021). A note on leveraging synergy in multiple meteorological data sets with deep learning for rainfall–runoff modeling. Hydrology and Earth System Sciences, 25, 2685–2703.
- Kratzert, F., Gauch, M., Nearing, G. and Klotz, D. (2022). NeuralHydrology — a Python library for deep learning research in hydrology. Journal of Open Source Software, 7(71), 4050.
- Kratzert, F., et al. (2024). HESS Opinions: Never train a Long Short-Term Memory (LSTM) network on a single basin. Hydrology and Earth System Sciences, 28, 4187–4201.
- Maurer, E. P., Wood, A. W., Adam, J. C., Lettenmaier, D. P. and Nijssen, B. (2002). A long-term hydrologically based dataset of land surface fluxes and states for the conterminous United States. Journal of Climate, 15, 3237–3251.
- Nash, J. E. and Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I — a discussion of principles. Journal of Hydrology, 10, 282–290.
- Nearing, G. S., Klotz, D., Frame, J. M., Gauch, M., Gilon, O., Kratzert, F., Sampson, A. K., Shalev, G. and Nevo, S. (2022). Technical note: Data assimilation and autoregression for using near-real-time streamflow observations in long short-term memory networks. Hydrology and Earth System Sciences, 26, 5493–5513.
- Nearing, G., et al. (2024). Global prediction of extreme floods in ungauged watersheds. Nature, 627, 559–563.
- Newman, A. J., et al. (2015). Development of a large-sample watershed-scale hydrometeorological data set for the contiguous USA: data set characteristics and assessment of regional variability in hydrologic model performance. Hydrology and Earth System Sciences, 19, 209–223.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V. and Courville, A. (2018). FiLM: visual reasoning with a general conditioning layer. Proceedings of the AAAI Conference on Artificial Intelligence, 32.
- Thornton, P. E., Running, S. W. and White, M. A. (1997). Generating surfaces of daily meteorological variables over large regions of complex terrain. Journal of Hydrology, 190, 214–251.
- Thornton, P. E., Thornton, M. M., Mayer, B. W., Wilhelmi, N., Wei, Y. and Cook, R. B. (2012). Daymet: daily surface weather on a 1 km grid for North America, 1980–2012. ORNL DAAC, Oak Ridge, Tennessee, USA. (The dataset entry cited by Addor et al., 2017.)
- van den Oord, A., et al. (2016). WaveNet: a generative model for raw audio. arXiv:1609.03499.
- Vaughan, A., Tebbutt, W., Hosking, J. S. and Turner, R. E. (2022). Convolutional conditional neural processes for local climate downscaling. Geoscientific Model Development, 15, 251–268.
- Xia, Y., et al. (2012). Continental-scale water and energy flux analysis and validation for the North American Land Data Assimilation System project phase 2 (NLDAS-2): 1. Intercomparison and application of model products. Journal of Geophysical Research: Atmospheres, 117, D03109 [CHECK: article number], doi:10.1029/2011JD016048.
- Yilmaz, K. K., Gupta, H. V. and Wagener, T. (2008). A process-based diagnostic approach to model evaluation: application to the NWS distributed hydrologic model. Water Resources Research, 44(9), W09417 [CHECK: article number], doi:10.1029/2007WR006716.

---

## Appendix A. Sources and files

| what | where |
|---|---|
| review of the 2020 code | `REVIEW.md` |
| 2020 MRes report | `AI4ER_MResProject_MGM_submission.pdf` (untracked) |
| changes from 2020 to `gss` | `/scratch2/mg963/agent-coordinator/inbox/2026-11-20-convcnp-changes.md`; `nh_experiments/CHANGES_AND_RESULTS.md` |
| model and neuralhydrology patches | `nh_experiments/convcnp_nh/{model.py,__init__.py}` |
| configurations | `nh_experiments/scripts/make_configs.py` → `nh_experiments/configs/` |
| tests (32, CPU) | `nh_experiments/tests/test_convcnp.py` |
| evaluation and reports | `nh_experiments/scripts/{eval_missing_forcings,eval_missing_scenarios,eval_gapfill,ensemble_report,pub_report,summarize}.py` |
| study notes | `agent-coordinator/inbox/2026-11-2{3,4,5,6,7}-convcnp-*-landau.md`; study A and B copies in `nh_experiments/docs/` |
| run outputs and summary tables | `/scratch2/mg963/convcnp_runs/` (`*.csv`, `*_report*.txt`, `logs/`) |
| queue scripts | `/scratch2/mg963/convcnp_runs/*queue*.sh` |
