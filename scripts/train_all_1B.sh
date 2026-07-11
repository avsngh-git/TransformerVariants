#!/bin/bash
# ============================================================================
# Full Controlled Experiment: All variants at main scale, 1B tokens, 3 seeds
# ============================================================================
#
# This trains all 10 variants (V0-V6 including sub-variants) with 3 random seeds
# each on 1B tokens from FineWeb-Edu. This is the main benchmark run.
#
# Token budget: 1,000,000,000 (1B) — Chinchilla-optimal for ~51M params
# Steps: 15,000 (effective batch = 8 * 8 * 1024 = 65,536 tokens/step)
# Estimated time per run: ~8-9 hours (main scale, L4 GPU)
# Total runs: 30 (10 variants × 3 seeds)
# Total estimated time: ~240-270 hours (~10-11 days)
#
# Prerequisites:
#   1. Prepare data: bash scripts/prepare_1B_data.sh
#   2. Verify data exists at: data/processed/fineweb-1B/
#
# Usage:
#   bash scripts/train_all_1B.sh 2>&1 | tee training_all_1B.log
#
# To resume after interruption:
#   - Check which runs completed in the log
#   - Comment out completed runs below
#   - Re-run the script
# ============================================================================

set -e

CONDA_ENV="transformer_lab"
TRAIN_SCRIPT="scripts/train.py"
DATA_DIR="data/processed/fineweb-1B"

# Training hyperparameters (same for all variants — controlled experiment)
MAX_STEPS=15000
MAX_LR="3e-4"
WARMUP_STEPS=500
MICRO_BATCH=8
GRAD_ACCUM=8
GRAD_CLIP=1.0
EVAL_INTERVAL=500
CHECKPOINT_INTERVAL=2500
LOG_INTERVAL=10

# Variants to train
DENSE_VARIANTS="vanilla modern alibi gqa swa swa_interleaved linear"
MOE_VARIANTS="moe moe_interleaved moe_deep"
ALL_VARIANTS="$DENSE_VARIANTS $MOE_VARIANTS"

# Seeds
SEEDS="42 137 2024"

# Count total runs
total_runs=0
for v in $ALL_VARIANTS; do
  for s in $SEEDS; do
    total_runs=$((total_runs + 1))
  done
done

echo "============================================================"
echo "Full Controlled Experiment: 1B tokens, main scale, 3 seeds"
echo "============================================================"
echo "  Variants: $(echo $ALL_VARIANTS | wc -w)"
echo "  Seeds: $(echo $SEEDS | wc -w)"
echo "  Total runs: $total_runs"
echo "  Steps per run: $MAX_STEPS"
echo "  Token budget per run: ~1B (65K tokens/step × 15K steps)"
echo "  Data: $DATA_DIR"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: Data directory not found: $DATA_DIR"
  echo "Run: bash scripts/prepare_1B_data.sh"
  exit 1
fi

run_num=0

for variant in $ALL_VARIANTS; do
  for seed in $SEEDS; do
    run_num=$((run_num + 1))

    # Determine compile flag (MoE variants skip compile)
    COMPILE_FLAG=""
    case $variant in
      moe|moe_interleaved|moe_deep) COMPILE_FLAG="" ;;
      *) COMPILE_FLAG="--compile" ;;
    esac

    # Determine activation (only matters for vanilla)
    ACTIVATION_FLAG=""
    if [ "$variant" = "vanilla" ]; then
      ACTIVATION_FLAG="--activation gelu"
    fi

    # Checkpoint dir includes seed for multi-seed runs
    CKPT_DIR="checkpoints/${variant}_main_1B_s${seed}"

    echo ">>> [$run_num/$total_runs] $variant — seed $seed"
    echo "    Checkpoint: $CKPT_DIR"
    echo "    Started: $(date)"

    # Check if this run already completed (final checkpoint exists at max_steps)
    FINAL_CKPT="$CKPT_DIR/checkpoint_step_$(printf '%06d' $MAX_STEPS).pt"
    if [ -f "$FINAL_CKPT" ]; then
      echo "    SKIPPING: Already completed ($FINAL_CKPT found)"
      echo ""
      continue
    fi

    # Check if we can resume from a partial run
    RESUME_FLAG=""
    if [ -f "$CKPT_DIR/checkpoint_latest.pt" ]; then
      echo "    RESUMING from $CKPT_DIR/checkpoint_latest.pt"
      RESUME_FLAG="--resume $CKPT_DIR/checkpoint_latest.pt"
    fi

    conda run -n $CONDA_ENV python $TRAIN_SCRIPT \
      --variant $variant \
      --scale main \
      --seed $seed \
      --data_dir $DATA_DIR \
      --max_steps $MAX_STEPS \
      --max_lr $MAX_LR \
      --warmup_steps $WARMUP_STEPS \
      --micro_batch_size $MICRO_BATCH \
      --grad_accum_steps $GRAD_ACCUM \
      --grad_clip $GRAD_CLIP \
      --eval_interval $EVAL_INTERVAL \
      --checkpoint_interval $CHECKPOINT_INTERVAL \
      --log_interval $LOG_INTERVAL \
      --checkpoint_dir $CKPT_DIR \
      $COMPILE_FLAG \
      $ACTIVATION_FLAG \
      $RESUME_FLAG

    echo "    Finished: $(date)"
    echo ""
  done
done

echo "============================================================"
echo "All training complete!"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run evaluation pipeline:"
echo "     conda run -n $CONDA_ENV python scripts/evaluate.py \\"
echo "       --checkpoints checkpoints/*_main_1B_s*/ \\"
echo "       --output reports/1B_comparison/ \\"
echo "       --data_dir $DATA_DIR"
echo "  2. Launch dashboard:"
echo "     REPORT_DIR=reports/1B_comparison/ streamlit run dashboard/app.py"
