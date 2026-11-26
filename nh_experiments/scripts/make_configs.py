#!/usr/bin/env python
"""Generate the experiment configs in nh_experiments/configs/ from one shared definition.

Everything that is not model-specific (basins, forcings, inputs, attributes, periods, loss,
metrics) is shared, so LSTM vs ConvCNP comparisons differ only in the model block.

  python nh_experiments/scripts/make_configs.py
"""
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = "/scratch2/mg963/data/camels_us/CAMELS_US"
RUN_DIR = "/scratch2/mg963/convcnp_runs"
BASINS = str(ROOT / "basins" / "531_basin_list.txt")

DYNAMIC = ["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)"]
# the 27 static attributes of Kratzert et al. (2019, HESS) / neuralhydrology CAMELS-US tutorials
STATIC = ["elev_mean", "slope_mean", "area_gages2", "frac_forest", "lai_max", "lai_diff", "gvf_max",
          "gvf_diff", "soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity", "soil_conductivity",
          "max_water_content", "sand_frac", "silt_frac", "clay_frac", "carbonate_rocks_frac",
          "geol_permeability", "p_mean", "pet_mean", "aridity", "frac_snow", "high_prec_freq",
          "high_prec_dur", "low_prec_freq", "low_prec_dur", "p_seasonality"]
METRICS = ["NSE", "KGE", "Alpha-NSE", "Beta-NSE", "Pearson-r", "FHV", "FLV", "FMS", "Peak-Timing"]

# --- periods (neuralhydrology uses dd/mm/yyyy) ----------------------------------------------------
PERIODS = {
    # Kratzert et al. 2019 (HESS), the published LSTM/EA-LSTM benchmark setup
    "kratzert2019": dict(train_start_date="01/10/1999", train_end_date="30/09/2008",
                         validation_start_date="01/10/1980", validation_end_date="30/09/1989",
                         test_start_date="01/10/1989", test_end_date="30/09/1999"),
    # the MRes split (= Newman et al. 2015 SAC-SMA calibration/validation split, first/last 15 WY).
    # Validation here re-uses the last 5 training years: monitoring only, NO model selection
    # (we always evaluate the last epoch). Use the *_tune configs for hyper-parameter search.
    "mres": dict(train_start_date="01/10/1980", train_end_date="30/09/1995",
                 validation_start_date="01/10/1990", validation_end_date="30/09/1995",
                 test_start_date="01/10/1995", test_end_date="30/09/2010"),
    # hyper-parameter tuning inside the MRes training period (test period never touched)
    "mres_tune": dict(train_start_date="01/10/1980", train_end_date="30/09/1990",
                      validation_start_date="01/10/1990", validation_end_date="30/09/1995",
                      test_start_date="01/10/1990", test_end_date="30/09/1995"),
}

COMMON = dict(
    run_dir=RUN_DIR,
    data_dir=DATA_DIR,
    dataset="camels_us",
    forcings=["daymet"],
    train_basin_file=BASINS, validation_basin_file=BASINS, test_basin_file=BASINS,
    dynamic_inputs=DYNAMIC,
    static_attributes=STATIC,
    target_variables=["QObs(mm/d)"],
    clip_targets_to_zero=["QObs(mm/d)"],
    loss="NSE",
    metrics=METRICS,
    optimizer="Adam",
    clip_gradient_norm=1.0,
    device="cuda:0",
    num_workers=8,
    seed=1,
    validate_every=5,
    validate_n_random_basins=100,
    save_weights_every=5,
    log_interval=5,
    log_tensorboard=True,
    log_n_figures=0,
    save_validation_results=False,
    cache_validation_data=True,
)

# Kratzert et al. 2019-style LSTM (neuralhydrology `cudalstm`, standard published hyper-parameters)
LSTM = dict(
    model="cudalstm", head="regression", output_activation="linear",
    hidden_size=256, initial_forget_bias=3, output_dropout=0.4,
    seq_length=365, predict_last_n=1,
    batch_size=256, epochs=30,
    learning_rate={0: 1e-3, 10: 5e-4, 25: 1e-4},
)

# Causal ConvCNP. Receptive field with these settings = 1 + 14 + 14 + 2*2*(1+2+...+64) = 537 days,
# so seq_length >= 537 + 365 - 1 = 901 gives every training target exactly its test-time history.
CONVCNP = dict(
    model="convcnp", head="regression", output_activation="linear",
    output_dropout=0.1,
    seq_length=910, predict_last_n=365,
    batch_size=64, epochs=30, max_updates_per_epoch=1000,
    learning_rate={0: 1e-3, 10: 5e-4, 25: 1e-4},
    convcnp=dict(channels=64, kernel_size=3, n_blocks=7, dilation_base=2, dropout=0.1,
                 encoder_kernel_size=15, decoder_kernel_size=15, init_length_scale=2.0,
                 causal=True, static_hidden=64, input_dropout=0.0),
)

# Probabilistic variant: countable mixture of asymmetric Laplacians (Klotz et al. 2022)
CONVCNP_CMAL = dict(CONVCNP, head="cmal", n_distributions=3, loss="CMALLoss",
                    n_samples=100, negative_sample_handling="clip", mc_dropout=False)

# Tuned ConvCNP ("gs"), selected on the tuning split (train WY1981-90, eval WY1991-95):
# multi-scale SetConv (incl. near-identity 0.25 d), identity path in the read-out, non-negative output
# (uncentred target + softplus), no per-timestep norm before the head, more dropout, gated TCN blocks,
# half the updates per epoch (later epochs overfit). See REVIEW.md / nh_experiments/README.md.
CONVCNP_GS = dict(
    CONVCNP,
    output_dropout=0.3,
    max_updates_per_epoch=500,
    output_activation="softplus",
    custom_normalization={"QObs(mm/d)": {"centering": "none"}},
    convcnp=dict(CONVCNP["convcnp"], dropout=0.25, encoder_scales=[0.25, 2.0, 8.0],
                 decoder_scales=[0.25, 2.0], decoder_identity=True, out_norm=False, gated=True),
)

# Tuned ConvCNP with stabilised initialisation ("gss"): identical to CONVCNP_GS plus head_bias_init=1.0
# (softplus head starts at ~1 normalised flow unit instead of possibly in its flat region). Added after
# 3/15 PUB runs of CONVCNP_GS were dead from epoch 1 (training loss stuck ~0.1).
CONVCNP_GSS = dict(CONVCNP_GS, convcnp=dict(CONVCNP_GS["convcnp"], head_bias_init=1.0))

# Probabilistic tuned ConvCNP: identical to CONVCNP_GS except the head (3-component countable mixture
# of asymmetric Laplacians, Klotz et al. 2022) and its NLL loss. Point prediction for NSE = sample mean.
CONVCNP_GS_CMAL = dict(CONVCNP_GS, head="cmal", n_distributions=3, loss="CMALLoss",
                       n_samples=100, negative_sample_handling="clip", mc_dropout=False)
CONVCNP_GS_CMAL.pop("output_activation")

# --- Missing inputs (Gauch et al. 2025, HESS 29, 6221-6235, "How to deal w___ missing input data") ---------
# Three forcing products as feature groups (Kratzert et al. 2021 / Gauch et al. 2025), Kratzert 2019 periods
# (train WY2000-08, validation WY1981-89, test WY1990-99), and the paper's training-time missing-data
# simulation p_time = p_sequence = 0.1 per group (their tuning / experiment-2 setting). The LSTM blocks are the
# paper's best configurations (Table A2): masked mean -- ReLU embedding [10, 10, 10, 10], no positional
# encoding, 35 epochs; input replacing -- ReLU embedding [17, 10], positional encoding 5, 30 epochs.
MULTI = dict(
    forcings=["daymet", "maurer_extended", "nldas_extended"],
    dynamic_inputs=[
        [f"{v}_daymet" for v in DYNAMIC],
        [f"{v}_maurer_extended" for v in DYNAMIC],
        [f"{v}_nldas_extended" for v in ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]],
    ],
    nan_step_probability=0.1, nan_sequence_probability=0.1,
)
# Single-product variant (Daymet only, one feature group). With one group neuralhydrology never drops the whole
# sequence (it keeps at least one group), so only p_time = 0.1 random days act; identical for both models.
DAYMET_NAN = dict(forcings=["daymet"], dynamic_inputs=[list(DYNAMIC)],
                  nan_step_probability=0.1, nan_sequence_probability=0.1)
LSTM_MM = dict(LSTM, nan_handling_method="masked_mean", epochs=35,
               dynamics_embedding=dict(type="fc", hiddens=[10, 10, 10, 10], activation="relu", dropout=0.0))
LSTM_IR = dict(LSTM, nan_handling_method="input_replacing", nan_handling_pos_encoding_size=5, epochs=30,
               dynamics_embedding=dict(type="fc", hiddens=[17, 10], activation="relu", dropout=0.0))

# --- Near-real-time streamflow as input (Nearing et al. 2022, HESS 26, 5493-5513) ----------------------------------
# 531 basins, Kratzert 2019 periods, all three forcing products (15 inputs, no forcing masking), observed Q at lag 1 as
# an extra input. Missing Q: contiguous runs with mean length 5 d (neuralhydrology `random_holdout_from_dynamic_features`,
# their sampler); they train at 50 % (main setting) and 0 %. The ConvCNP sees Q through its density channel; the
# LSTMs are not re-trained here -- we compare against their published numbers.
AR = dict(
    forcings=MULTI["forcings"], dynamic_inputs=[v for g in MULTI["dynamic_inputs"] for v in g],
    lagged_features={"QObs(mm/d)": [1]}, autoregressive_inputs=["QObs(mm/d)_shift1"],
    random_holdout_from_dynamic_features={"QObs(mm/d)_shift1": {"missing_fraction": 0.5, "mean_missing_length": 5}},
)
AR0 = dict(AR, random_holdout_from_dynamic_features={})

# --- Study B: gap-filling observed streamflow ------------------------------------------------------------------
# Non-causal gss with observed Q at the same time step (QObs(mm/d)_shift0) as an input. Training hides 2-6 gaps per
# window (lengths log-uniform in 1-120 days) in that input and scores only the hidden days (convcnp.gap_inputs,
# see ConvCNP.pre_model_hook). Receptive field 537 days = +-268 around each day, so with seq_length 910 and
# predict_last_n 642 the loss region is the central 374 days [268, 642), each with full two-sided context.
# No forcing masking (like convcnp_gss_ar50/ar0). In-training validation is off: neuralhydrology's tester never hides
# Q, so it would score a copy of the target (evaluation is scripts/eval_gapfill.py).
CONVCNP_GSS_GAPFILL = dict(
    CONVCNP_GSS, predict_last_n=642, validate_every=None,
    convcnp=dict(CONVCNP_GSS["convcnp"], causal=False, gap_inputs=["QObs(mm/d)_shift0"],
                 gap_len_range=[1, 120], gap_count_range=[2, 6]))
GAPFILL = dict(AR, lagged_features={"QObs(mm/d)": [0]}, autoregressive_inputs=["QObs(mm/d)_shift0"],
               random_holdout_from_dynamic_features={})

EXPERIMENTS = {
    "convcnp_gss_ar50": (CONVCNP_GSS, "kratzert2019", AR),
    "convcnp_gss_ar0": (CONVCNP_GSS, "kratzert2019", AR0),
    "convcnp_gss_gapfill": (CONVCNP_GSS_GAPFILL, "kratzert2019", GAPFILL),
    "lstm_mm_multi": (LSTM_MM, "kratzert2019", MULTI),
    "lstm_ir_multi": (LSTM_IR, "kratzert2019", MULTI),
    "convcnp_gss_multi": (CONVCNP_GSS, "kratzert2019", MULTI),
    "lstm_ir_daymet": (LSTM_IR, "kratzert2019", DAYMET_NAN),
    "convcnp_gss_daymet": (CONVCNP_GSS, "kratzert2019", DAYMET_NAN),
    # control: identical to convcnp_gss_daymet but without training-time masking
    "convcnp_gss_daymet_nomask": (CONVCNP_GSS, "kratzert2019",
                                  dict(DAYMET_NAN, nan_step_probability=0.0, nan_sequence_probability=0.0)),
    "lstm_kratzert2019": (LSTM, "kratzert2019"),
    "convcnp_kratzert2019": (CONVCNP, "kratzert2019"),
    "lstm_mres": (LSTM, "mres"),
    "convcnp_mres": (CONVCNP, "mres"),
    "convcnp_cmal_mres": (CONVCNP_CMAL, "mres"),
    "convcnp_mres_tune": (CONVCNP, "mres_tune"),
    "lstm_mres_tune": (LSTM, "mres_tune"),
    "convcnp_gs_mres": (CONVCNP_GS, "mres"),
    "convcnp_gs_mres_tune": (CONVCNP_GS, "mres_tune"),
    "convcnp_gs_kratzert2019": (CONVCNP_GS, "kratzert2019"),
    "convcnp_gs_cmal_mres": (CONVCNP_GS_CMAL, "mres"),
    "convcnp_gss_mres": (CONVCNP_GSS, "mres"),
    "convcnp_gss_mres_tune": (CONVCNP_GSS, "mres_tune"),
    "convcnp_gs_cmal_mres_tune": (CONVCNP_GS_CMAL, "mres_tune"),
}

# Prediction in ungauged basins (PUB): 5-fold CV over basins (folds from scripts/make_pub_folds.py),
# MRes periods. Test basins are unseen in space and time; validation only monitors training basins.
PUB_K = 5
for _k in range(1, PUB_K + 1):
    _fold = dict(train_basin_file=str(ROOT / "basins" / f"pub_k{PUB_K}_fold{_k}_train.txt"),
                 validation_basin_file=str(ROOT / "basins" / f"pub_k{PUB_K}_fold{_k}_train.txt"),
                 test_basin_file=str(ROOT / "basins" / f"pub_k{PUB_K}_fold{_k}_test.txt"),
                 validate_n_random_basins=50)
    EXPERIMENTS[f"lstm_pub_fold{_k}"] = (LSTM, "mres", _fold)
    EXPERIMENTS[f"convcnp_gs_pub_fold{_k}"] = (CONVCNP_GS, "mres", _fold)
    EXPERIMENTS[f"convcnp_gss_pub_fold{_k}"] = (CONVCNP_GSS, "mres", _fold)

HEADER = """# Generated by nh_experiments/scripts/make_configs.py -- edit that script, not this file.
# Experiment: {name}   periods: {periods}
"""


def main():
    yaml = YAML()
    yaml.default_flow_style = False
    out_dir = ROOT / "configs"
    out_dir.mkdir(exist_ok=True)
    for name, spec in EXPERIMENTS.items():
        model, periods = spec[0], spec[1]
        extra = spec[2] if len(spec) > 2 else {}
        cfg = {"experiment_name": name}
        cfg.update(COMMON)
        cfg.update(PERIODS[periods])
        cfg.update(model)
        cfg.update(extra)
        path = out_dir / f"{name}.yml"
        with path.open("w") as fp:
            fp.write(HEADER.format(name=name, periods=periods))
            yaml.dump(cfg, fp)
        print("wrote", path)


if __name__ == "__main__":
    main()
