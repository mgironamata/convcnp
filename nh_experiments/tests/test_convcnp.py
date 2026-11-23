"""Unit tests for the neuralhydrology ConvCNP (no CAMELS data needed).

Run: python -m pytest -q nh_experiments/tests
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convcnp_nh import register  # noqa: E402

register()
from neuralhydrology.utils.config import Config  # noqa: E402
from convcnp_nh.model import ConvCNP  # noqa: E402

DYN = ["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)"]
STAT = ["p_mean", "aridity", "area_gages2"]


def make_cfg(**over):
    d = {
        "model": "convcnp", "head": "regression", "output_activation": "linear",
        "dynamic_inputs": DYN, "static_attributes": STAT, "target_variables": ["QObs(mm/d)"],
        "seq_length": 200, "predict_last_n": 50, "output_dropout": 0.0,
        "convcnp": {"channels": 16, "n_blocks": 4, "kernel_size": 3, "encoder_kernel_size": 7,
                    "decoder_kernel_size": 5, "dropout": 0.0},
    }
    conv = over.pop("convcnp", {})
    d.update(over)
    d["convcnp"].update(conv)
    return Config(d)


def make_batch(B=2, T=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x_d": {k: torch.randn(B, T, 1, generator=g) for k in DYN},
            "x_s": torch.randn(B, len(STAT), generator=g)}


def clone(batch):
    return {"x_d": {k: v.clone() for k, v in batch["x_d"].items()}, "x_s": batch["x_s"].clone()}


@pytest.mark.parametrize("head,n_out", [("regression", 1), ("gmm", 3 * 3), ("cmal", 4 * 3)])
def test_shapes(head, n_out):
    extra = {"n_distributions": 3} if head != "regression" else {}
    model = ConvCNP(make_cfg(head=head, **extra)).eval()
    out = model(make_batch())
    key = "y_hat" if head == "regression" else {"gmm": "mu", "cmal": "mu"}[head]
    assert out[key].shape[:2] == (2, 200)
    assert torch.isfinite(out[key]).all()


def test_causal_no_future_leakage():
    model = ConvCNP(make_cfg()).eval()
    b = make_batch()
    t = 120
    b2 = clone(b)
    for k in DYN:
        b2["x_d"][k][:, t + 1:] += 10 * torch.randn_like(b2["x_d"][k][:, t + 1:])
    with torch.no_grad():
        y1, y2 = model(b)["y_hat"], model(b2)["y_hat"]
    assert torch.allclose(y1[:, : t + 1], y2[:, : t + 1], atol=1e-5), "output depends on future inputs"
    assert not torch.allclose(y1[:, t + 1:], y2[:, t + 1:]), "perturbation had no effect at all"


def test_receptive_field_is_exact():
    # (float64: the oldest in-RF step only reaches the output through every block's edge tap)
    model = ConvCNP(make_cfg(convcnp={"init_length_scale": 1000.0})).eval().double()
    rf = model.receptive_field
    t = 199
    assert rf <= t

    def batch():
        b = make_batch()
        return {"x_d": {k: v.double() for k, v in b["x_d"].items()}, "x_s": b["x_s"].double()}

    with torch.no_grad():
        base = model(batch())["y_hat"][:, t]
        b_out = batch(); b_out["x_d"]["prcp(mm/day)"][:, t - rf] += 100.0   # just outside RF
        b_in = batch(); b_in["x_d"]["prcp(mm/day)"][:, t - rf + 1] += 100.0  # oldest step inside RF
        assert (model(b_out)["y_hat"][:, t] - base).abs().max() < 1e-12
        assert (model(b_in)["y_hat"][:, t] - base).abs().max() > 1e-10


def test_nan_inputs_are_masked():
    model = ConvCNP(make_cfg()).eval()
    b = make_batch()
    b["x_d"]["tmax(C)"][:, 50:80] = float("nan")
    with torch.no_grad():
        y = model(b)["y_hat"]
    assert torch.isfinite(y).all()


def test_static_conditioning_matters_after_training_step():
    model = ConvCNP(make_cfg())
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    b = make_batch()
    loss = model(b)["y_hat"].pow(2).mean()
    loss.backward(); opt.step()
    model.eval()
    b2 = clone(b); b2["x_s"] = b2["x_s"] + 1.0
    with torch.no_grad():
        assert not torch.allclose(model(b)["y_hat"], model(b2)["y_hat"])


def test_unknown_option_rejected():
    with pytest.raises(ValueError):
        ConvCNP(make_cfg(convcnp={"chanels": 8}))


def test_noncausal_uses_future():
    model = ConvCNP(make_cfg(convcnp={"causal": True, "kernel_size": 3}))
    model_nc = ConvCNP(make_cfg(convcnp={"causal": False, "kernel_size": 3})).eval()
    b = make_batch(); b2 = clone(b)
    b2["x_d"]["prcp(mm/day)"][:, 150] += 100.0
    with torch.no_grad():
        assert not torch.allclose(model_nc(b)["y_hat"][:, 140], model_nc(b2)["y_hat"][:, 140])


def test_tester_evaluates_every_day_with_predict_last_n_1(tmp_path):
    from neuralhydrology.evaluation.tester import RegressionTester
    basins = tmp_path / "basins.txt"; basins.write_text("01000001\n")
    cfg = make_cfg(predict_last_n=50, loss="NSE", metrics=["NSE"], test_basin_file=str(basins),
                   test_start_date="01/10/1995", test_end_date="30/09/1996", device="cpu",
                   data_dir=str(tmp_path), dataset="camels_us")
    tester = RegressionTester.__new__(RegressionTester)
    try:  # BaseTester.__init__ sets self.cfg first; it then needs a fitted scaler, which we don't have
        tester.__init__(cfg, run_dir=tmp_path, period="test", init_model=False)
    except FileNotFoundError:
        pass
    assert tester.cfg.predict_last_n == 1
    assert cfg.predict_last_n == 50, "training config must not be modified"


def _original_cmal(prediction, y, eps=1e-8):
    mask = ~torch.isnan(y).any(1).any(1)
    y = y[mask]; m, b, t, p = (prediction[k][mask] for k in ("mu", "b", "tau", "pi"))
    error = y - m
    ll = torch.log(t) + torch.log(1.0 - t) - torch.log(b) - torch.max(t * error, (t - 1.0) * error) / b
    return -torch.mean(torch.sum(torch.logsumexp(torch.log(p + eps) + ll, dim=2), dim=1))


def test_cmal_loss_elementwise_masking():
    from neuralhydrology.training.loss import MaskedCMALLoss
    g = torch.Generator().manual_seed(0)
    B, T, K = 4, 30, 3
    pred = {"mu": torch.randn(B, T, K, generator=g), "b": torch.rand(B, T, K, generator=g) + 0.1,
            "tau": torch.rand(B, T, K, generator=g) * 0.8 + 0.1,
            "pi": torch.softmax(torch.randn(B, T, K, generator=g), -1)}
    y = torch.randn(B, T, 1, generator=g)
    loss = MaskedCMALLoss.__new__(MaskedCMALLoss); loss.eps = 1e-8
    # identical to the original for a single predicted step
    p1 = {k: v[:, -1:] for k, v in pred.items()}
    assert torch.allclose(loss._get_loss(p1, {"y": y[:, -1:]}), _original_cmal(p1, y[:, -1:]), atol=1e-6)
    # NaNs in every sample: original -> NaN, patched -> finite
    y_nan = y.clone(); y_nan[:, 5] = float("nan")
    assert torch.isnan(_original_cmal(pred, y_nan))
    assert torch.isfinite(loss._get_loss(pred, {"y": y_nan}))


def test_multiscale_and_decoder_identity_causal_and_rf():
    over = {"encoder_scales": [0.25, 2.0, 8.0], "decoder_scales": [0.25, 2.0], "decoder_identity": True}
    model = ConvCNP(make_cfg(convcnp=over)).eval()
    assert model.encoder.n_out == 3 * len(DYN) and model.decoder.n_out == 2 * 16
    b = make_batch(); t = 120; b2 = clone(b)
    for k in DYN:
        b2["x_d"][k][:, t + 1:] += 10.0
    with torch.no_grad():
        y1, y2 = model(b)["y_hat"], model(b2)["y_hat"]
    assert torch.allclose(y1[:, : t + 1], y2[:, : t + 1], atol=1e-5)
    # receptive field still exact (float64, flat kernels)
    m = ConvCNP(make_cfg(convcnp={**over, "encoder_scales": [1000.0, 500.0], "decoder_scales": [1000.0]})).eval().double()
    rf, t = m.receptive_field, 199

    def batch():
        bb = make_batch()
        return {"x_d": {k: v.double() for k, v in bb["x_d"].items()}, "x_s": bb["x_s"].double()}
    with torch.no_grad():
        base = m(batch())["y_hat"][:, t]
        bo = batch(); bo["x_d"]["prcp(mm/day)"][:, t - rf] += 100.0
        bi = batch(); bi["x_d"]["prcp(mm/day)"][:, t - rf + 1] += 100.0
        assert (m(bo)["y_hat"][:, t] - base).abs().max() < 1e-12
        assert (m(bi)["y_hat"][:, t] - base).abs().max() > 1e-10


def test_out_norm_off_runs_and_is_causal():
    model = ConvCNP(make_cfg(convcnp={"out_norm": False, "encoder_scales": [0.25, 2.0]})).eval()
    b = make_batch(); b2 = clone(b); t = 100
    for k in DYN:
        b2["x_d"][k][:, t + 1:] -= 5.0
    with torch.no_grad():
        assert torch.allclose(model(b)["y_hat"][:, : t + 1], model(b2)["y_hat"][:, : t + 1], atol=1e-5)


def test_gated_blocks_causal_and_rf_unchanged():
    m0 = ConvCNP(make_cfg())
    m = ConvCNP(make_cfg(convcnp={"gated": True})).eval()
    assert m.receptive_field == m0.receptive_field
    b = make_batch(); b2 = clone(b); t = 90
    for k in DYN:
        b2["x_d"][k][:, t + 1:] *= -3.0
    with torch.no_grad():
        assert torch.allclose(m(b)["y_hat"][:, : t + 1], m(b2)["y_hat"][:, : t + 1], atol=1e-5)


def test_block_norm_none_causal():
    m = ConvCNP(make_cfg(convcnp={"block_norm": "none", "out_norm": False})).eval()
    assert not any(isinstance(mod, torch.nn.LayerNorm) for mod in m.modules())
    b = make_batch(); b2 = clone(b); t = 80
    for k in DYN:
        b2["x_d"][k][:, t + 1:] += 7.0
    with torch.no_grad():
        assert torch.allclose(m(b)["y_hat"][:, : t + 1], m(b2)["y_hat"][:, : t + 1], atol=1e-5)
    with pytest.raises(ValueError):
        ConvCNP(make_cfg(convcnp={"block_norm": "batch"}))


def test_skip_outputs_causal_and_rf_unchanged():
    over = {"skip": True, "gated": True, "decoder_identity": True, "decoder_scales": [0.25, 2.0]}
    m = ConvCNP(make_cfg(convcnp=over)).eval()
    assert m.receptive_field == ConvCNP(make_cfg()).receptive_field
    assert all(b.skip_conv is not None for b in m.blocks)
    b = make_batch(); b2 = clone(b); t = 110
    for k in DYN:
        b2["x_d"][k][:, t + 1:] += 4.0
    with torch.no_grad():
        y1, y2 = m(b)["y_hat"], m(b2)["y_hat"]
    assert y1.shape == (2, 200, 1)
    assert torch.allclose(y1[:, : t + 1], y2[:, : t + 1], atol=1e-5)


def test_training_extras_config_keys():
    cfg = make_cfg(ema_decay=0.99, weight_decay=0.05, optimizer="AdamW", learning_rate={0: 1e-3})
    assert cfg.ema_decay == 0.99 and cfg.weight_decay == 0.05
    import neuralhydrology.training.basetrainer as bt
    opt = bt.get_optimizer(ConvCNP(cfg), cfg)
    assert isinstance(opt, torch.optim.AdamW) and opt.param_groups[0]["weight_decay"] == 0.05


def test_head_bias_init_softplus_starts_at_target_level():
    cfg = make_cfg(output_activation="softplus", convcnp={"head_bias_init": 1.0, "skip": True, "gated": True})
    m = ConvCNP(cfg).eval()
    with torch.no_grad():
        y = m(make_batch())["y_hat"]
    assert torch.isfinite(y).all() and (y > 0).all()
    assert 0.3 < y.mean().item() < 3.0      # starts near softplus(bias) = 1, not in the flat region


def _fake_scaler():
    import numpy as np
    import xarray as xr
    ds = lambda v: xr.Dataset({"QObs(mm/d)": xr.DataArray(np.float32(v))})
    return {"xarray_feature_center": ds(0.0), "xarray_feature_scale": ds(1.0)}


def test_cmal_sampling_uses_tester_predict_last_n():
    from neuralhydrology.utils import samplingutils
    import neuralhydrology.evaluation.tester as tmod
    cfg_train = make_cfg(head="cmal", n_distributions=3, loss="CMALLoss", n_samples=7,
                         negative_sample_handling="clip", mc_dropout=False, predict_last_n=50)
    model = ConvCNP(cfg_train).eval()
    seen = {}
    orig = samplingutils._get_frequency_last_n

    def spy(predict_last_n, *a, **k):
        seen["pln"] = predict_last_n
        return orig(predict_last_n, *a, **k)

    samplingutils._get_frequency_last_n = spy
    try:
        tester = tmod.UncertaintyTester.__new__(tmod.UncertaintyTester)
        tester.cfg = make_cfg(head="cmal", n_distributions=3, loss="CMALLoss", n_samples=7,
                              negative_sample_handling="clip", mc_dropout=False, predict_last_n=1)
        from neuralhydrology.training import get_loss_obj
        tester.loss_obj = get_loss_obj(tester.cfg)
        b = make_batch()
        b["y"] = torch.rand(2, 200, 1)
        b["per_basin_target_stds"] = torch.ones(2, 1, 1)
        model.sample = lambda data, n: samplingutils.sample_cmal(model, data, n,
                                                                  scaler=_fake_scaler())
        with torch.no_grad():
            preds, _ = tester._get_predictions_and_loss(model, b)
    finally:
        samplingutils._get_frequency_last_n = orig
    assert seen["pln"] == 1, seen
    assert preds["y_hat"].shape[1] == 1          # one predicted step per window, not 50
    assert model.cfg.predict_last_n == 50        # model config restored


def test_window_alpha_penalty():
    import torch
    from convcnp_nh import window_alpha_penalty
    torch.manual_seed(0)
    y = torch.randn(4, 100, 1)
    assert window_alpha_penalty(y.clone(), y).item() < 1e-4            # perfect variance -> 0
    p = window_alpha_penalty(0.5 * y, y).item()                         # half the std -> (0.5-1)^2
    assert abs(p - 0.25) < 1e-2
    y2 = y.clone(); y2[:, :80] = float("nan")                           # <30 valid points -> ignored
    assert window_alpha_penalty(0.5 * y, y2).item() == 0.0
    yh = (0.5 * y).requires_grad_()
    window_alpha_penalty(yh, y).backward()
    assert torch.isfinite(yh.grad).all()


def test_local_blocks_receptive_field_and_causality():
    m = ConvCNP(make_cfg(convcnp={"local_blocks": 2, "init_length_scale": 1000.0})).eval().double()
    m0 = ConvCNP(make_cfg())
    assert len(m.blocks) == len(m0.blocks) + 2
    assert m.receptive_field == m0.receptive_field + 2 * 2 * (3 - 1)
    rf, t = m.receptive_field, 199
    def batch():
        b = make_batch()
        return {"x_d": {k: v.double() for k, v in b["x_d"].items()}, "x_s": b["x_s"].double()}
    with torch.no_grad():
        base = m(batch())["y_hat"][:, t]
        b_out = batch(); b_out["x_d"]["prcp(mm/day)"][:, t - rf] += 100.0
        b_in = batch(); b_in["x_d"]["prcp(mm/day)"][:, t - rf + 1] += 100.0
        assert (m(b_out)["y_hat"][:, t] - base).abs().max() < 1e-12
        assert (m(b_in)["y_hat"][:, t] - base).abs().max() > 1e-10


def test_window_peak_penalty():
    from convcnp_nh import window_peak_penalty
    torch.manual_seed(0)
    y = torch.rand(3, 100, 1); stds = torch.ones_like(y)
    assert window_peak_penalty(y.clone(), y, stds, eps=0.0).item() < 1e-12
    # error only on the low flows -> no peak penalty; error on the top 5% -> penalty
    low = y.clone(); low[y < 0.5] += 1.0
    assert window_peak_penalty(low, y, stds, eps=0.0).item() < 1e-12
    high = y.clone(); high[y >= torch.quantile(y.squeeze(-1), 0.95, dim=1, keepdim=True).unsqueeze(-1)] -= 0.5
    assert abs(window_peak_penalty(high, y, stds, eps=0.0).item() - 0.25) < 1e-6
    y2 = y.clone(); y2[:, :80] = float("nan")                 # <30 valid -> ignored, NaNs tolerated
    assert window_peak_penalty(high, y2, stds).item() == 0.0


def test_nse_loss_extra_terms_and_warmup():
    from neuralhydrology.training import get_loss_obj
    y = torch.rand(2, 365, 1); stds = torch.ones_like(y)
    pred, gt = {"y_hat": 0.5 * y}, {"y": y, "per_basin_target_stds": stds}
    base = get_loss_obj(make_cfg(loss="NSE", predict_last_n=365, seq_length=400))(pred, gt)[0].item()
    L = get_loss_obj(make_cfg(loss="NSE", predict_last_n=365, seq_length=400, peak_loss_weight=1.0,
                              loss_warmup_updates=2))
    vals = [L(pred, gt)[0].item() for _ in range(3)]
    assert abs(vals[0] - base) < 1e-7 and abs(vals[1] - base) < 1e-7 and vals[2] > base + 1e-4
    with torch.no_grad():                                      # validation: extra terms on, no counting
        assert L(pred, gt)[0].item() > base + 1e-4
