#!/usr/bin/env python
"""Train / evaluate neuralhydrology models (incl. the ConvCNP) with identical data + evaluation.

Examples
--------
  # train (one seed); the run directory is created under `run_dir` from the config
  python nh_experiments/run.py train configs/lstm_mres.yml --gpu 0 --seed 1

  # several seeds (sequentially) -> later ensemble them with scripts/summarize.py
  python nh_experiments/run.py train configs/convcnp_mres.yml --gpu 0 --seed 1 2 3

  # evaluate the last epoch on the test period
  python nh_experiments/run.py eval /scratch2/mg963/convcnp_runs/<run_dir> --period test --gpu 0

  # quick CPU smoke test with overrides
  python nh_experiments/run.py train configs/convcnp_mres.yml --gpu -1 \
      --override epochs=1 max_updates_per_epoch=5 data_dir=/scratch2/mg963/data/camels_us_synthetic

`train` also runs test-period evaluation afterwards unless --no-test is given.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ruamel.yaml import YAML

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from convcnp_nh import register  # noqa: E402

register()

from neuralhydrology.evaluation.evaluate import start_evaluation  # noqa: E402
from neuralhydrology.training.train import start_training  # noqa: E402
from neuralhydrology.utils.config import Config  # noqa: E402

LOGGER = logging.getLogger("run")


def _parse_override(items: list[str]) -> dict:
    yaml = YAML(typ="safe")
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        parsed = yaml.load(value)
        # nested keys, e.g. convcnp.channels=128
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = parsed
    return out


def _resolve_config_path(path: str) -> Path:
    p = Path(path)
    if not p.exists() and (HERE / p).exists():
        p = HERE / p
    return p.resolve()


def _set_device(cfg: Config, gpu: int | None):
    if gpu is None:
        return
    cfg.device = f"cuda:{gpu}" if gpu >= 0 else "cpu"


def cmd_train(args):
    yaml = YAML(typ="safe")
    cfg_path = _resolve_config_path(args.config)
    base = yaml.load(cfg_path.read_text())
    overrides = _parse_override(args.override)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key].update(val)
        else:
            base[key] = val

    seeds = args.seed or [base.get("seed")]
    run_dirs = []
    for seed in seeds:
        d = dict(base)
        if seed is not None:
            d["seed"] = int(seed)
            d["experiment_name"] = f"{base['experiment_name']}_seed{seed}"
        cfg = Config(d)
        _set_device(cfg, args.gpu)
        if args.dry_run:
            _dry_run(cfg)
            continue
        start_training(cfg)
        run_dirs.append(cfg.run_dir)
        print(f"[run.py] finished training: {cfg.run_dir}", flush=True)
        if not args.no_test:
            _evaluate(cfg.run_dir, "test", None, args.gpu)
    print("[run.py] run directories:\n" + "\n".join(str(r) for r in run_dirs))


def _dry_run(cfg: Config):
    """Print the resolved settings (and the ConvCNP receptive field) without training."""
    keys = ["experiment_name", "model", "head", "loss", "seed", "device", "seq_length", "predict_last_n",
            "batch_size", "epochs", "max_updates_per_epoch", "learning_rate", "train_start_date", "train_end_date",
            "test_start_date", "test_end_date", "data_dir", "run_dir", "output_dropout", "output_activation",
            "weight_decay", "ema_decay", "alpha_loss_weight", "peak_loss_weight", "loss_warmup_updates"]
    for k in keys:
        try:
            print(f"  {k:24s} {getattr(cfg, k)}")
        except (AttributeError, ValueError):
            pass
    if cfg.model.lower() == "convcnp":
        from convcnp_nh.model import ConvCNP
        model = ConvCNP(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        ok = cfg.seq_length - cfg.predict_last_n + 1 >= model.receptive_field
        print(f"  {'convcnp':24s} {model.p}")
        print(f"  receptive field {model.receptive_field} d, params {n_params:,}, "
              f"train/test-consistent: {ok}")
    print()


def _evaluate(run_dir: Path, period: str, epoch: int | None, gpu: int | None):
    cfg = Config(Path(run_dir) / "config.yml")
    _set_device(cfg, gpu)
    start_evaluation(cfg=cfg, run_dir=Path(run_dir), epoch=epoch, period=period)
    print(f"[run.py] evaluated {run_dir} on {period}", flush=True)


def cmd_eval(args):
    for run_dir in args.run_dir:
        _evaluate(Path(run_dir), args.period, args.epoch, args.gpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train")
    p.add_argument("config")
    p.add_argument("--gpu", type=int, default=None, help="GPU id; <0 = CPU; default = config 'device'")
    p.add_argument("--seed", type=int, nargs="*", help="one or more seeds (run sequentially)")
    p.add_argument("--override", nargs="*", default=[], help="config overrides key=value (nested: a.b=value)")
    p.add_argument("--no-test", action="store_true", help="skip test-period evaluation after training")
    p.add_argument("--dry-run", action="store_true", help="print resolved config + receptive field, don't train")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval")
    p.add_argument("run_dir", nargs="+")
    p.add_argument("--period", default="test", choices=["train", "validation", "test"])
    p.add_argument("--epoch", type=int, default=None, help="default: last epoch")
    p.add_argument("--gpu", type=int, default=None)
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
