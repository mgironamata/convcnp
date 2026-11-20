#!/bin/bash
# Ordered experiment plan. Run stages individually, e.g.:
#   GPU=0 bash nh_experiments/experiments.sh lstm_repro
# Stages:
#   lstm_repro   LSTM on Kratzert et al. (2019) periods      -> sanity check vs. published numbers
#   lstm_mres    LSTM on the MRes split (WY1981-95 / 96-2010) -> the baseline to beat
#   convcnp_mres causal ConvCNP on the MRes split
#   sacsma       SAC-SMA benchmark on the MRes test period (needs modelOutput_daymet zip)
#   summary      summary table (per seed + seed ensembles)
#   tune         small ConvCNP grid on the tuning split (test period untouched)
#   cmal         probabilistic ConvCNP (CMAL head)
#   convcnp_repro ConvCNP on Kratzert periods
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/scratch2/mg963/conda_envs/envs/nh/bin/python
GPU=${GPU:-0}
SEEDS=${SEEDS:-"1 2 3"}
RUNS=/scratch2/mg963/convcnp_runs
DATA=/scratch2/mg963/data/camels_us

case "${1:-}" in
  lstm_repro)    $PY nh_experiments/run.py train nh_experiments/configs/lstm_kratzert2019.yml --gpu $GPU --seed $SEEDS ;;
  lstm_mres)     $PY nh_experiments/run.py train nh_experiments/configs/lstm_mres.yml --gpu $GPU --seed $SEEDS ;;
  convcnp_mres)  $PY nh_experiments/run.py train nh_experiments/configs/convcnp_mres.yml --gpu $GPU --seed $SEEDS ;;
  convcnp_repro) $PY nh_experiments/run.py train nh_experiments/configs/convcnp_kratzert2019.yml --gpu $GPU --seed $SEEDS ;;
  cmal)          $PY nh_experiments/run.py train nh_experiments/configs/convcnp_cmal_mres.yml --gpu $GPU --seed $SEEDS ;;
  sacsma)
    $PY nh_experiments/scripts/sacsma_benchmark.py --model-output $DATA/unpacked/modelOutput_daymet \
        --data-dir $DATA/CAMELS_US --start 1995-10-01 --end 2010-09-30 --seeds mean \
        --out $RUNS/sacsma_daymet_mres_test.csv ;;
  summary)
    BM=""; [ -f $RUNS/sacsma_daymet_mres_test.csv ] && BM="--benchmark $RUNS/sacsma_daymet_mres_test.csv"
    echo "### MRes split (test WY1996-2010)"
    $PY nh_experiments/scripts/summarize.py $RUNS/lstm_mres_seed* $RUNS/convcnp_mres_seed* $BM --out $RUNS/summary_mres.csv
    if ls -d $RUNS/lstm_kratzert2019_seed* >/dev/null 2>&1; then
      echo; echo "### Kratzert et al. 2019 periods (test WY1990-1999)"
      $PY nh_experiments/scripts/summarize.py $RUNS/*_kratzert2019_seed* --out $RUNS/summary_kratzert2019.csv
    fi ;;
  tune)
    # one seed per setting; compare on the validation (= tuning test) period WY1991-1995
    # (8 blocks -> receptive field 1049 d -> seq_length >= 1049 + 365 - 1 = 1413)
    for ov in "convcnp.channels=64" "convcnp.channels=128" "convcnp.channels=128 convcnp.n_blocks=8 seq_length=1415" \
              "convcnp.dropout=0.2 output_dropout=0.2" "convcnp.input_dropout=0.1" "convcnp.static_hidden=128"; do
      tag=$(echo "$ov" | tr ' =.{}:,' '_______' | tr -s '_')
      $PY nh_experiments/run.py train nh_experiments/configs/convcnp_mres_tune.yml --gpu $GPU --seed 1 \
          --override $ov experiment_name=convcnp_tune_$tag
    done
    $PY nh_experiments/run.py train nh_experiments/configs/lstm_mres_tune.yml --gpu $GPU --seed 1
    $PY nh_experiments/scripts/summarize.py $RUNS/convcnp_tune_* $RUNS/lstm_mres_tune_seed* --out $RUNS/summary_tune.csv ;;
  *) sed -n 2,16p "$0"; exit 1 ;;
esac
