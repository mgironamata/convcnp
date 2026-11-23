"""On-the-grid ConvCNP for rainfall-runoff modelling, as a neuralhydrology model.

Design (fixes the issues found in REVIEW.md):

* **Encoder = SetConv on the daily grid.** Each dynamic input channel d gets a learnable Gaussian
  kernel psi_d. With observation mask m_d(t) (0 where an input is missing / NaN),
      density_d(t) = sum_k m_d(t-k) psi_d(k)
      signal_d(t)  = sum_k m_d(t-k) x_d(t-k) psi_d(k) / (density_d(t) + eps)
  i.e. exactly the ConvCNP SetConv (Gordon et al. 2020), discretised at the data resolution
  and computed per channel (no 3-channel bottleneck, no feature mean-pooling). The 2*D
  channels (density + signal) are projected point-wise to `channels` hidden channels.
* **rho = residual dilated TCN**, FiLM-conditioned on the full static attribute vector.
* **Decoder = SetConv read-out** (learnable Gaussian smoothing over grid features) followed by
  any neuralhydrology head (regression / gmm / cmal / umal).
* **Causal by default**: every kernel only looks at t' <= t, so the prediction for day t
  never uses future forcings. This makes the model directly comparable with an LSTM, and,
  provided ``seq_length - predict_last_n + 1 >= receptive_field``, every training target sees
  exactly the same history it would see at test time (neuralhydrology tests each day as the
  last step of its own window).
* **Missing data**: NaNs in the dynamic inputs are handled by the density channel; the
  ``input_dropout`` option randomly hides inputs during training (ConvCNP-style context
  subsampling, as an augmentation rather than the default task).

Hyper-parameters live in the ``convcnp:`` block of the run config (see ``register.py``).
"""
from __future__ import annotations

import logging
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)

DEFAULTS = {
    "channels": 64,              # hidden channels of rho
    "kernel_size": 3,            # TCN kernel size
    "n_blocks": 7,               # residual blocks, dilation = dilation_base ** i
    "dilation_base": 2,
    "local_blocks": 0,           # extra undilated (dilation 1) blocks before the dilated stack (fast local dynamics)
    "dropout": 0.1,              # dropout inside the TCN blocks
    "encoder_kernel_size": 15,   # support (days) of the SetConv encoder kernel
    "decoder_kernel_size": 15,   # support (days) of the SetConv read-out kernel
    "init_length_scale": 2.0,    # initial Gaussian length scale (days)
    "encoder_scales": None,      # list of initial length scales per variable (multi-scale SetConv);
                                 # None -> [init_length_scale]. Include ~0.25 for a near-identity path.
    "decoder_scales": None,      # same for the read-out; None -> [init_length_scale]
    "decoder_identity": False,   # also feed the un-smoothed rho features to the head
    "block_norm": "layer",       # normalisation inside TCN blocks: "layer" (per-timestep LayerNorm) or "none"
    "skip": False,               # WaveNet-style skip outputs: per-block 1x1 projection of the gated activation,
                                 # summed over blocks and fed to the read-out next to the residual stream
    "head_bias_init": None,      # if set (e.g. 1.0): init the head bias so softplus(bias) = this value and
                                 # shrink head weights x0.1, so a softplus head cannot start in its flat region
    "gated": False,              # WaveNet-style gated activation tanh(f)*sigmoid(g) in the TCN blocks
    "out_norm": True,            # per-timestep LayerNorm before the head (False keeps feature amplitude)
    "causal": True,
    "static_hidden": 64,         # hidden size of the static-attribute MLP (FiLM generator)
    "input_dropout": 0.0,        # prob. of hiding each (time, variable) input during training
}


def _get_convcnp_cfg(cfg: Config) -> dict:
    user = {}
    try:
        user = dict(cfg.convcnp or {})
    except AttributeError:  # register() not called -> no `convcnp` property
        pass
    unknown = set(user) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown convcnp options: {sorted(unknown)}. Valid: {sorted(DEFAULTS)}")
    out = dict(DEFAULTS)
    out.update(user)
    return out


class GaussianSetConv(nn.Module):
    """Depthwise Gaussian convolution on a regular grid with learnable per-channel length scales.

    ``init_length_scales`` may be a float or a list of S floats (multi-scale): every input channel
    is convolved with S kernels, giving S*C output channels ordered [scale_0 (all C), scale_1, ...].
    If ``normalise`` is True, returns [density, signal / density] (2*S*C channels), using ``mask``;
    otherwise returns the plain smoothed signal (S*C channels).
    """

    def __init__(self, n_channels: int, kernel_size: int, init_length_scales, causal: bool,
                 normalise: bool):
        super().__init__()
        if not causal and kernel_size % 2 == 0:
            kernel_size += 1  # symmetric kernel needs odd support
        scales = list(init_length_scales) if isinstance(init_length_scales, (list, tuple)) else [init_length_scales]
        self.n_scales = len(scales)
        self.n_channels = n_channels
        self.n_out = n_channels * self.n_scales
        self.kernel_size = kernel_size
        self.causal = causal
        self.normalise = normalise
        init = torch.tensor([math.log(s) for s in scales for _ in range(n_channels)], dtype=torch.float32)
        self.log_length_scale = nn.Parameter(init)
        if causal:
            lags = torch.arange(kernel_size - 1, -1, -1, dtype=torch.float32)  # weight j <-> lag K-1-j
        else:
            half = (kernel_size - 1) // 2
            lags = torch.arange(-half, half + 1, dtype=torch.float32)
        self.register_buffer("lags", lags, persistent=False)

    def _kernel(self) -> torch.Tensor:
        ls = self.log_length_scale.exp()[:, None]                      # [C, 1]
        w = torch.exp(-0.5 * (self.lags[None, :] / ls) ** 2)           # [C, K]
        return w[:, None, :]                                           # [C, 1, K] depthwise

    def _conv(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kernel_size
        pad = (k - 1, 0) if self.causal else ((k - 1) // 2, (k - 1) // 2)
        if self.n_scales > 1:
            x = x.repeat(1, self.n_scales, 1)
        return F.conv1d(F.pad(x, pad), self._kernel(), groups=self.n_out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x, mask: [B, C, T]
        if not self.normalise:
            return self._conv(x)
        if mask is None:
            mask = torch.ones_like(x)
        density = self._conv(mask)
        signal = self._conv(x * mask) / (density + 1e-6)
        return torch.cat([density, signal], dim=1)


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels at each time step (does not mix information across time)."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T]
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, causal: bool,
                 gated: bool = False, norm: str = "layer", skip: bool = False):
        super().__init__()
        self.causal = causal
        self.gated = gated
        self.pad_total = (kernel_size - 1) * dilation
        if not causal and self.pad_total % 2:
            raise ValueError("Non-causal TCN needs (kernel_size-1)*dilation to be even (use odd kernel_size).")
        if norm not in ("layer", "none"):
            raise ValueError(f"block_norm must be 'layer' or 'none', got {norm!r}")
        make_norm = (lambda: ChannelLayerNorm(channels)) if norm == "layer" else (lambda: nn.Identity())
        self.norm1 = make_norm()
        self.conv1 = nn.Conv1d(channels, 2 * channels if gated else channels, kernel_size, dilation=dilation)
        self.norm2 = make_norm()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.skip_conv = nn.Conv1d(channels, channels, kernel_size=1) if skip else None

    def _pad(self, x):
        if self.causal:
            return F.pad(x, (self.pad_total, 0))
        return F.pad(x, (self.pad_total // 2, self.pad_total // 2))

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        h = self.norm1(x)
        h = h * gamma[..., None] + beta[..., None]           # FiLM with static attributes
        h = self.conv1(self._pad(F.gelu(h)))
        if self.gated:
            f, g = h.chunk(2, dim=1)
            h = torch.tanh(f) * torch.sigmoid(g)
        skip = self.skip_conv(h) if self.skip_conv is not None else None
        h = self.conv2(self._pad(F.gelu(self.norm2(h))))
        out = x + self.dropout(h)
        return (out, skip) if self.skip_conv is not None else out


class ConvCNP(BaseModel):
    """Causal on-the-grid ConvCNP (SetConv encoder -> FiLM-TCN -> SetConv read-out -> head)."""

    module_parts = ["encoder", "input_proj", "static_net", "blocks", "decoder", "head"]

    def __init__(self, cfg: Config):
        super().__init__(cfg=cfg)
        p = _get_convcnp_cfg(cfg)
        self.p = p
        if isinstance(cfg.dynamic_inputs, dict) or (cfg.dynamic_inputs and isinstance(cfg.dynamic_inputs[0], list)):
            raise ValueError("ConvCNP supports a flat list of dynamic_inputs (single frequency) only.")
        self.dynamic_inputs = list(cfg.dynamic_inputs)
        n_dyn = len(self.dynamic_inputs)
        if cfg.evolving_attributes:
            raise ValueError("ConvCNP does not support evolving_attributes; add them to dynamic_inputs instead.")
        n_static = len(cfg.static_attributes) + len(cfg.hydroatlas_attributes)
        if cfg.use_basin_id_encoding:
            n_static += cfg.number_of_basins
        C = p["channels"]
        self.dilations = [1] * int(p["local_blocks"]) + [p["dilation_base"] ** i for i in range(p["n_blocks"])]

        enc_scales = p["encoder_scales"] or [p["init_length_scale"]]
        dec_scales = p["decoder_scales"] or [p["init_length_scale"]]
        self.encoder = GaussianSetConv(n_dyn, p["encoder_kernel_size"], enc_scales, p["causal"], normalise=True)
        self.input_proj = nn.Conv1d(2 * self.encoder.n_out, C, kernel_size=1)

        self.n_static = n_static
        self.static_net = None
        if n_static > 0:
            self.static_net = nn.Sequential(
                nn.Linear(n_static, p["static_hidden"]), nn.GELU(),
                nn.Linear(p["static_hidden"], 2 * C * len(self.dilations)))
            nn.init.zeros_(self.static_net[-1].weight)      # start as identity FiLM
            nn.init.zeros_(self.static_net[-1].bias)

        self.blocks = nn.ModuleList([
            TCNBlock(C, p["kernel_size"], d, p["dropout"], p["causal"], p["gated"], p["block_norm"], p["skip"])
            for d in self.dilations])
        C_feat = 2 * C if p["skip"] else C                  # residual stream (+ summed skip outputs)
        self.decoder = GaussianSetConv(C_feat, p["decoder_kernel_size"], dec_scales, p["causal"], normalise=False)
        n_head = self.decoder.n_out + (C_feat if p["decoder_identity"] else 0)
        self.out_norm = ChannelLayerNorm(n_head) if p["out_norm"] else nn.Identity()
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=n_head, n_out=self.output_size)
        if p["head_bias_init"] is not None:
            lin = [m for m in self.head.modules() if isinstance(m, nn.Linear)][-1]
            with torch.no_grad():
                lin.weight.mul_(0.1)
                lin.bias.fill_(math.log(math.expm1(float(p["head_bias_init"]))))   # inverse softplus

        self._check_receptive_field(cfg)

    # ------------------------------------------------------------------------------------------------
    @property
    def receptive_field(self) -> int:
        """Number of past days (incl. the current one) that influence a prediction (causal mode)."""
        p = self.p
        rf = 1 + (self.encoder.kernel_size - 1) + (self.decoder.kernel_size - 1)
        rf += sum(2 * (p["kernel_size"] - 1) * d for d in self.dilations)
        return rf

    def _check_receptive_field(self, cfg: Config):
        seq, pln = cfg.seq_length, cfg.predict_last_n
        if not isinstance(seq, int) or not isinstance(pln, int):
            return
        rf = self.receptive_field
        LOGGER.info(f"ConvCNP receptive field: {rf} days (causal={self.p['causal']}), "
                    f"seq_length={seq}, predict_last_n={pln}")
        if self.p["causal"] and seq - pln + 1 < rf:
            LOGGER.warning(f"seq_length - predict_last_n + 1 = {seq - pln + 1} < receptive field {rf}: early "
                           f"training targets see zero-padded history, unlike at test time. Increase seq_length "
                           f"to >= {rf + pln - 1} for exact train/test consistency.")
        if not self.p["causal"]:
            LOGGER.warning("Non-causal ConvCNP: predictions use future forcings within the window. Not comparable "
                           "with an LSTM, and neuralhydrology evaluates each day at the window end (no future).")

    # ------------------------------------------------------------------------------------------------
    def forward(self, data: dict) -> Dict[str, torch.Tensor]:
        x_d = data["x_d"]
        x = torch.cat([x_d[k] for k in self.dynamic_inputs], dim=-1)        # [B, T, D]
        mask = torch.isfinite(x).to(x.dtype)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.training and self.p["input_dropout"] > 0:
            mask = mask * torch.bernoulli(torch.full_like(mask, 1.0 - self.p["input_dropout"]))
        x, mask = x.transpose(1, 2), mask.transpose(1, 2)                  # [B, D, T]

        h = self.input_proj(self.encoder(x, mask))                          # [B, C, T]

        B, C = h.shape[0], h.shape[1]
        if self.static_net is not None:
            statics = [data[k] for k in ("x_s", "x_one_hot") if k in data]
            film = self.static_net(torch.cat(statics, dim=-1)).view(B, len(self.blocks), 2, C)
            gammas, betas = 1.0 + film[:, :, 0], film[:, :, 1]
        else:
            gammas = h.new_ones(B, len(self.blocks), C)
            betas = h.new_zeros(B, len(self.blocks), C)

        skip_sum = None
        for i, block in enumerate(self.blocks):
            if self.p["skip"]:
                h, s = block(h, gammas[:, i], betas[:, i])
                skip_sum = s if skip_sum is None else skip_sum + s
            else:
                h = block(h, gammas[:, i], betas[:, i])
        if skip_sum is not None:
            h = torch.cat([h, F.gelu(skip_sum / math.sqrt(len(self.blocks)))], dim=1)

        h_dec = self.decoder(h)
        if self.p["decoder_identity"]:
            h_dec = torch.cat([h, h_dec], dim=1)
        h = F.gelu(self.out_norm(h_dec))                                    # [B, C', T]
        h = h.transpose(1, 2)                                               # [B, T, C]
        pred = {"convcnp_features": h}
        pred.update(self.head(self.dropout(h)))
        return pred
