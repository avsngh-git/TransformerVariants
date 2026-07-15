#!/usr/bin/env bash
# Retrain V5 after the numerical and RoFormer-formulation correction.

set -euo pipefail

CONDA_ENV="transformer_lab"
DATA_DIR="data/processed/fineweb-1B"
SEEDS=(42 137 2024)

for seed in "${SEEDS[@]}"; do
    checkpoint_dir="checkpoints/linear_main_1B_fixed_s${seed}"
    log_file="logs/v5_fixed_s${seed}.log"

    echo "===== fixed V5 seed ${seed} started at $(date --iso-8601=seconds) =====" | tee "$log_file"
    conda run -n "$CONDA_ENV" python scripts/train.py \
        --variant linear \
        --scale main \
        --seed "$seed" \
        --data_dir "$DATA_DIR" \
        --max_steps 15000 \
        --max_lr 3e-4 \
        --warmup_steps 500 \
        --micro_batch_size 8 \
        --grad_accum_steps 8 \
        --grad_clip 1.0 \
        --eval_interval 500 \
        --checkpoint_interval 2500 \
        --log_interval 10 \
        --checkpoint_dir "$checkpoint_dir" \
        --compile 2>&1 | tee -a "$log_file"
    echo "===== fixed V5 seed ${seed} finished at $(date --iso-8601=seconds) =====" | tee -a "$log_file"
done
