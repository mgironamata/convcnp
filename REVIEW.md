# Review: ConvCNPs for CAMELS rainfall–runoff (2026 revisit)

Scope: `convcnp/` package, `convcnp_clean.ipynb` (the notebook that produced the report's numbers), and the MRes report PDF.
Older notebooks/scripts (`convcnp_regression_*`, `sequential_inference.ipynb`, `train.py`, `data_hydro*.py`, `data_365.py`) are the earlier GP-toy prototypes and were not used for the final results.

## 0. What the results actually are

- The final experiment is in `convcnp_clean.ipynb` (cells 15–68). Checkpoint `_experiments/2020-08-02_00-31-53_k5fold-01-gamma-staticfs-after-encoder-unet`, "best epoch 85". `_experiments/`, the CAMELS CSVs and `train_test_splits/` are **not** on this machine.
- The CDF figure in cell 68 reproduces report Table 4.1 exactly. On the gauged-basin, temporal-test split (PGB, 424 basins, WY1996–2010):
  - **median NSE 0.37**
  - mean −0.14
  - min −28.7
  - PUB median 0.31 (Table 4.2)
- **There is no LSTM baseline anywhere.** The report compares against SAC-SMA (Newman et al.) only. For context, an LSTM or EA-LSTM on CAMELS-531 gets roughly 0.70–0.76 median NSE in the gauged setting (Kratzert 2019 HESS, different periods), and about 0.69 in PUB (Kratzert 2019 WRR, k-fold). So the gap is about 0.35 NSE: big enough to point at structural problems, not tuning.
- The task really is pure simulation. `context_mask = target_mask = [0,1,1,...]`, so discharge is never an input. I checked for discharge leakage through the masks and found none.

## 1. Problems that most likely explain the gap (ranked)

### 1.1 The encoder squeezes all meteorology into 3 channels (bug, confirmed)
- The notebook computes `rho_in_channels = 8 + static_embedding_dims = 13` but never uses it. It calls `rho=UNet()`, whose default is `in_channels=8`.
- In `ConvCNP.__init__` (`convcnp/convcnp_architectures.py`, the `encoder_out_channels = self.rho.in_channels - self.static_embedding_dims` line), this gives `encoder_out_channels = 8 − 5 = 3`.
- The SetConv is linear, then a single `Linear(11→3)` and a sigmoid follow. So precipitation, temperature, radiation, vapour pressure and day-of-year all get folded into **3 sigmoid channels per grid point** before the CNN sees them.
- I instantiated the model exactly as in cell 22: 51,378 parameters, matching the notebook output, and `encoder.out_channels == 3`.
- The whole model is also tiny (51k parameters vs about 300k for a 256-unit LSTM). The UNet's widest layer is 64 channels.

### 1.2 Training sees an incomplete forcing record; testing sees a complete one
- In `HydroGenerator.generate_task` with `extrapolate=False`, each 365-day window keeps only `num_train + num_test` randomly chosen days: 50–165 plus 150–200, i.e. 200–365 days, about 77% on average.
- Because of `observe_at_target=True`, forcings exist only on those days. On the remaining ~23% of days there is **no precipitation at all**, so many targets are unpredictable. The model learns to hedge and smooth, which matches the systematic peak under-prediction in Fig. B.1/B.2 and cell 70.
- The test (`generate_test_task`, 364 context + 1 target) gives a dense 365-day record, which is a different input distribution from the one the model trained on.

### 1.3 Training is non-causal, but the test is causal and at the window edge
- In training, targets sit anywhere in the window, so the CNN (a non-causal UNet) learns to use **future** forcings.
- At test, the target is the last day (x = 364/365), with no future context. It also sits at the grid boundary, where the UNet's zero padding and reflect padding (`pad_concat`) distort things.
- The model is therefore evaluated almost exclusively on a configuration it rarely trained on.

### 1.4 Loss scaling ignores per-basin magnitude
- `OBS_RUN` is standardised with one global μ/σ pooled over all basins, then shifted by +1.42 for the gamma likelihood.
- High-flow basins dominate the NLL. In low-flow or arid basins, every value sits close to 1.0, so the model barely distinguishes them, and the NSE there (a per-basin, variance-normalised metric) collapses. This is where the −28 to −1 NSE tail comes from.
- Kratzert's NSE* loss weights each basin by 1/σ_basin precisely to avoid this.
- The gamma is also defined on a shifted variable. Support (0,1) in shifted space corresponds to **negative discharge**, and a gamma's shape is not invariant to an arbitrary shift. This is one reason FLV/FMS blow up to ±1e8–1e11 (log of negative or near-zero flows) and some basins come out NaN.

### 1.5 Weak feature encoders
- Dynamic features go through a DeepSet: each scalar becomes a pair (value, feature_id/8), passes through an MLP, and is **mean-pooled** across features into 10 dims.
- Static attributes are handled the same way: 27 attributes are mean-pooled into **5 dims**, then broadcast as constant channels.
- That design is meant for missing covariates. CAMELS forcings and attributes are complete, so concatenating channels, plus an MLP or FiLM for the statics, would be strictly more expressive.
- EA-LSTM uses the full 27-dim static vector to gate every timestep.

### 1.6 Data-pipeline errors
- **Windows cross basins.** `randoms = np.random.randint(0, len(df) - timeslice)` indexes into the concatenated multi-basin frame, so about 365/5479 ≈ 6.7% of windows mix the end of one basin with the start of the next. The attributes come from the first basin.
- **Gaps from dropped rows.** Rows with missing Q (`OBS_RUN < 0`) are dropped. That also deletes their forcings and compresses the time axis, because x comes from the row index, not the date. Better to keep the rows and mask the loss.
- **Normalisation statistics** for forcings, Q and attributes are computed over all periods and basins, test included. This is minor.
- **Rows dropped in evaluation.** The first year of every basin is skipped (`sorted(years)[1:]`), and the last day of each year is dropped (`df[s_ind:e_ind]` excludes `e_ind`). Also minor.

### 1.7 Training regime
- Constant LR of 1e-3, no schedule, no gradient clipping. Single run, no ensemble.
- `train(gen, ...)` iterates the single-process generator, not the `DataLoader` built in cell 44, at about 6 min/epoch.
- Diagnostics were blind: train NSE was always `nan` (see 2.2).

## 2. Evaluation bugs (affect reported numbers or diagnostics, not training)

1. **The per-basin `NSE=` printed in cell 62 is meaningless.** It calls `calc_nse(observations, pred_mu)`, comparing back-transformed observations in mm/day against a **non-back-transformed** `pred_mu`. The stored `nse_index` is correct and is the one in the report.
2. **`rev_transform_tensor` (cell 28) applies `torch.exp`**, but Q was never log-transformed. That is why train NSE was always `nan` and NSE could not be monitored.
3. **Cell 68 back-transforms observations twice** (`rev_transform(df_hru['OBS_RUN'])` on values that were already back-transformed). The `nse_array` it produces is unused, but it is wrong.
4. **FLV/FMS are not interpretable** because of negative or near-zero predictions (see 1.4). The SAC-SMA FLV of −1.9e10 in the report shows `metrics.py` misbehaves on zero-flow basins regardless.
5. **The PUB test is not a clean PUB test.** The checkpoint was selected on `df_test_catchment`, which is the *same* "ungauged" basins (in the training period). Also, the SAC-SMA numbers in Table 4.2 come from per-basin-calibrated models, so they are not a PUB benchmark.

## 3. Recommended plan

1. **Get a real baseline on the identical setup.** Run an LSTM with `neuralhydrology` on the same 531 basins, Daymet forcings, WY1981–1995 train / 1996–2010 test, and the same 27 attributes. The number to beat should come from your own pipeline, not from a paper with different periods.
2. **Rebuild the data pipeline:**
   - Index windows per basin.
   - Keep all forcing days and mask missing Q in the loss.
   - Normalise using training-period statistics only.
3. **Make the task match the evaluation:**
   - Use dense forcings over a window of warm-up plus prediction days (e.g. 730 days), and compute the loss on every day after the warm-up.
   - Choose causal (causal or dilated convolutions, LSTM-comparable) or non-causal (centred windows, evaluated the same way), and use the same choice for train and test.
   - Random context subsampling becomes an *augmentation* ablation, not the default.
4. **Architecture:**
   - Remove the 3-channel bottleneck: 64–128 channels, and pass `in_channels` to the CNN correctly.
   - Concatenate forcings as channels.
   - Condition on static attributes via FiLM or an MLP on the full vector.
   - Use a receptive field of at least 365 days (dilated TCN or a deeper UNet).
   - LR schedule, gradient clipping, and an ensemble of a few seeds.
5. **Loss:**
   - First a Gaussian NLL or NSE* on per-basin-scaled Q, to compare point accuracy with the LSTM.
   - Then a proper positive likelihood on raw mm/day Q (log-normal, gamma with ε, or a CMAL/mixture as in neuralhydrology).
6. **Evaluation:**
   - Fix the metric bugs above and report median and mean NSE/KGE.
   - For PUB, use k-fold over basins with validation basins separate from test basins.
7. **Be honest about where ConvCNPs add value.** With dense, regular daily forcings, the SetConv encoder is essentially an identity onto a grid, so the model reduces to a CNN with an RBF read-out. Matching the LSTM there is a reasonable goal; beating it is not guaranteed. The NP machinery earns its keep with **irregular or missing data**: missing forcing days, sparse or intermittent gauges, and assimilating past observed discharge as context (`context_mask` with Q = 1) for forecasting or gap-filling. Those are the experiments where a ConvCNP is most likely to show something an LSTM can't easily do.
