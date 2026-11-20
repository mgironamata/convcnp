"""Plug a ConvCNP into neuralhydrology without forking it.

Call :func:`register` once before using neuralhydrology's training/evaluation entry points.
It (1) adds a ``convcnp`` property to ``Config`` so the run YAML may contain a ``convcnp:`` block
(neuralhydrology rejects unknown keys otherwise), and (2) wraps ``get_model`` in the trainer and
tester so ``model: convcnp`` resolves to :class:`convcnp_nh.model.ConvCNP`.
"""
import copy

from neuralhydrology.utils.config import Config

_REGISTERED = False


def register():
    global _REGISTERED
    if _REGISTERED:
        return
    if not hasattr(Config, "convcnp"):
        Config.convcnp = property(lambda self: self._cfg.get("convcnp", {}) or {})

    import neuralhydrology.modelzoo as modelzoo
    import neuralhydrology.training.basetrainer as basetrainer
    import neuralhydrology.evaluation.tester as tester
    from .model import ConvCNP

    original = modelzoo.get_model

    def get_model(cfg):
        if cfg.model.lower() == "convcnp":
            return ConvCNP(cfg=cfg)
        return original(cfg)

    modelzoo.get_model = get_model
    basetrainer.get_model = get_model
    tester.get_model = get_model

    # Evaluation protocol. neuralhydrology scores only the last step of each evaluation window, but
    # with predict_last_n = N it builds windows whose *whole* N-day target block lies inside the
    # period, silently dropping the first N-1 days of the validation/test period (and storing N
    # predictions per day). The ConvCNP is causal with receptive field <= seq_length, so its
    # last-step prediction does not depend on predict_last_n: evaluate with predict_last_n = 1,
    # i.e. one window ending on every day of the period -- the same protocol as the LSTM.
    # Training (loss over the last N steps) is unaffected.
    original_init = tester.BaseTester.__init__

    def tester_init(self, cfg, run_dir, period="test", init_model=True):
        if cfg.model.lower() == "convcnp" and isinstance(cfg.predict_last_n, int) and cfg.predict_last_n > 1:
            cfg_dict = copy.deepcopy(cfg.as_dict())
            cfg_dict["predict_last_n"] = 1
            cfg = Config(cfg_dict)
        original_init(self, cfg, run_dir, period, init_model)

    tester.BaseTester.__init__ = tester_init

    # Probabilistic heads: neuralhydrology's samplers read predict_last_n from *model.cfg*. During
    # in-training validation the model still carries the training config (predict_last_n = 365), so
    # it would draw samples for 365 steps per window and keep only the last (365x the work). Let the
    # sampler see the tester's config (predict_last_n = 1) while sampling.
    original_unc = tester.UncertaintyTester._get_predictions_and_loss

    def unc_get_predictions_and_loss(self, model, data):
        if getattr(model, "cfg", None) is not None and model.cfg.model.lower() == "convcnp" \
                and model.cfg.predict_last_n != self.cfg.predict_last_n:
            model_cfg = model.cfg
            model.cfg = self.cfg
            try:
                return original_unc(self, model, data)
            finally:
                model.cfg = model_cfg
        return original_unc(self, model, data)

    tester.UncertaintyTester._get_predictions_and_loss = unc_get_predictions_and_loss

    # CMAL loss. neuralhydrology's MaskedCMALLoss drops every *sample* that has a missing target
    # anywhere in its predicted sequence and sums the NLL over time. With predict_last_n = 365 this
    # discards almost all samples in basins with gaps (an all-masked batch gives NaN) and scales the
    # loss with N. Use element-wise masking + mean over valid points instead. For predict_last_n = 1
    # (LSTM default) this is mathematically identical to the original.
    import neuralhydrology.training.loss as nh_loss
    import torch

    def cmal_get_loss(self, prediction, ground_truth, **kwargs):
        y = ground_truth["y"]                                      # [B, T, 1]
        valid = ~torch.isnan(y[..., 0])                            # [B, T]
        if not valid.any():
            return (prediction["mu"] * 0.0).sum()
        error = torch.nan_to_num(y, nan=0.0) - prediction["mu"]    # [B, T, K]
        t, b = prediction["tau"], prediction["b"]
        log_like = (torch.log(t) + torch.log(1.0 - t) - torch.log(b)
                    - torch.max(t * error, (t - 1.0) * error) / b)
        result = torch.logsumexp(torch.log(prediction["pi"] + self.eps) + log_like, dim=2)  # [B, T]
        return -result[valid].mean()

    nh_loss.MaskedCMALLoss._get_loss = cmal_get_loss

    _register_training_extras()
    _REGISTERED = True


def _register_alpha_loss():
    """``alpha_loss_weight`` (lambda): NSE* loss + lambda * mean_over_windows (std(y_hat)/std(y) - 1)^2.

    The variance ratio is computed per training window over its ``predict_last_n`` target block (365 d for
    the ConvCNP), so it penalises over-smoothing (Alpha-NSE < 1) directly. Windows with fewer than
    ``min_valid`` observed targets are ignored; with ``predict_last_n = 1`` (LSTM) the term is inactive.
    """
    import torch
    import neuralhydrology.training.loss as nh_loss

    if not hasattr(Config, "alpha_loss_weight"):
        Config.alpha_loss_weight = property(lambda self: self._cfg.get("alpha_loss_weight", None))

    cls = nh_loss.MaskedNSELoss
    original_init, original_get_loss = cls.__init__, cls._get_loss

    def __init__(self, cfg, eps: float = 0.1):
        original_init(self, cfg, eps=eps)
        self._alpha_w = float(cfg.alpha_loss_weight or 0.0)

    def _get_loss(self, prediction, ground_truth, **kwargs):
        nse = original_get_loss(self, prediction, ground_truth, **kwargs)
        lam = getattr(self, "_alpha_w", 0.0)
        y, y_hat = ground_truth["y"], prediction["y_hat"]
        if lam <= 0 or y.shape[1] < 2:
            return nse
        return nse + lam * window_alpha_penalty(y_hat, y)

    cls.__init__, cls._get_loss = __init__, _get_loss


def window_alpha_penalty(y_hat, y, min_valid: int = 30, eps: float = 1e-3):
    """mean over windows of (std(y_hat)/std(y) - 1)^2, using only observed targets. Shapes [bs, n, 1]."""
    import torch
    mask = (~torch.isnan(y)).to(y_hat.dtype)
    yv = torch.nan_to_num(y)
    n = mask.sum(dim=1)                                           # [bs, 1]
    def std(x):
        mu = (x * mask).sum(dim=1) / n.clamp(min=1)
        return torch.sqrt((((x - mu.unsqueeze(1)) * mask) ** 2).sum(dim=1) / n.clamp(min=1) + 1e-8)
    ratio = std(y_hat) / (std(yv) + eps)
    ok = n >= min_valid
    if not ok.any():
        return y_hat.sum() * 0.0
    return ((ratio - 1.0) ** 2)[ok].mean()


def _register_training_extras():
    """Model-agnostic training options (usable for the LSTM too):

    * ``weight_decay``: passed to Adam/AdamW (neuralhydrology passes none; torch's AdamW default is 0.01).
    * ``ema_decay``: exponential moving average of the weights, updated after every optimizer step. At
      the end of each epoch the EMA weights are swapped into the model, so checkpoints, validation and
      the final test all use them; the raw weights are restored before the next epoch's training.
    """
    import torch
    import neuralhydrology.training.basetrainer as basetrainer

    if not hasattr(Config, "weight_decay"):
        Config.weight_decay = property(lambda self: self._cfg.get("weight_decay", None))
    if not hasattr(Config, "ema_decay"):
        Config.ema_decay = property(lambda self: self._cfg.get("ema_decay", None))

    original_get_optimizer = basetrainer.get_optimizer

    def get_optimizer(model, cfg):
        if cfg.weight_decay is None:
            return original_get_optimizer(model=model, cfg=cfg)
        name = cfg.optimizer.lower()
        cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(name)
        if cls is None:
            raise NotImplementedError(f"weight_decay not supported for optimizer {cfg.optimizer}")
        return cls(model.parameters(), lr=cfg.learning_rate[0], weight_decay=float(cfg.weight_decay))

    basetrainer.get_optimizer = get_optimizer

    _register_alpha_loss()

    Trainer = basetrainer.BaseTrainer
    original_init_training = Trainer.initialize_training
    original_train_epoch = Trainer._train_epoch

    def initialize_training(self):
        original_init_training(self)
        self._ema = None
        decay = self.cfg.ema_decay
        if decay:
            self._ema_decay = float(decay)
            self._ema = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self._ema_raw = None
            opt_step = self.optimizer.step

            def step_with_ema(*args, **kwargs):
                out = opt_step(*args, **kwargs)
                with torch.no_grad():
                    for k, v in self.model.state_dict().items():
                        if v.dtype.is_floating_point:
                            self._ema[k].mul_(self._ema_decay).add_(v.detach(), alpha=1.0 - self._ema_decay)
                        else:
                            self._ema[k].copy_(v)
                return out

            self.optimizer.step = step_with_ema

    def _train_epoch(self, epoch):
        ema = getattr(self, "_ema", None)
        if ema is not None and self._ema_raw is not None:          # restore raw weights for training
            self.model.load_state_dict(self._ema_raw)
            self._ema_raw = None
        original_train_epoch(self, epoch)
        if ema is not None:                                         # evaluate / save with EMA weights
            self._ema_raw = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(ema)

    Trainer.initialize_training = initialize_training
    Trainer._train_epoch = _train_epoch
