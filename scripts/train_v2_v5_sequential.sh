#!/bin/bash
# Sequential training of V2–V5 + V4-interleaved at main and stretch scales.
# Each run uses --compile for ~15-25% speedup.
# Estimated total time: ~50-70 hours on L4 GPU.
#
# Usage: bash scripts/train_v2_v5_sequential.sh 2>&1 | tee training_v2_v5.log

set -e

CONDA_ENV="transformer_lab"
TRAIN_SCRIPT="scripts/train.py"
DATA_DIR_MAIN="data/processed/wikitext-100M"
DATA_DIR_STRETCH="data/processed/wikitext-120M"

echo "============================================="
echo "Sequential Training: V2-V5 at main + stretch"
echo "Started: $(date)"
echo "============================================="

# --- V2: ALiBi ---
echo ""
echo ">>> [1/10] V2 ALiBi — main scale"
echo "    Started: $(date)"
conda run -n $CONDA_ENV python $TRAIN_SCRIPT \
  --variant alibi --scale main --compile \
  --data_dir $DATA_DIR_MAIN
echo "    Finished: $(date)"

echo ""
echo ">>> [2/10] V2 ALiBi — stretch scale"
echo "    Started: $(date)"
conda run -n $CONDA_ENV python $TRAIN_SCRIPT \
  --variant alibi --scale stretch --compile \
  --data_dir $DATA_DIR_STRETCH
echo "    Finished: $(date)"

# --- V3: GQA ---
echo ""
echo ">>> [3/10] V3 GQA — main scale"
echo "    Started: $(date)"
conda run -n $CONDA_ENV python $TRAIN_SCRIPT \
  --variant gqa --scale main --compile \
  --data_dir $DATA_DIR_MAIN
echo "    Finished: $(date)"

echo ""
echo ">>> [4/10] V3 GQA — stretch scale"
echo "    Started: $(date)"
conda run -n $CONDA_ENV python $TRAIN_SCRIPT \
  --variant gqa --scale stretch --compile \
  --data_dir $DATA_DIR_STRETCH
echo "    Finished: $(date)"

# --- V4: SWA ---
echo ""
echo ">>> [5/10] V4 SWA — main scale"
echo "    Started: $(date)"
conda run -n $CONDA_ENV python $TRAIN_SCRIPT \
  --variant swa --scale main --compile \
  --data_dir $DATA_DIR_MAIN
echo "    Finished: $(date)"

echo ""
echo ">>> [6/10] V4 SWA — stretch scale"
echo "    Started: $(date)"
conda run -n $CONDA_ENV python $TRAIN_SCRIPT \
  --variant swa --scale stretch --compile \
  --data_dir $DATA_DIR_STRETCH
echo "    Finished: $(date)"

# --- V4-interleaved ---
echo ""
echo ">>> [7/10] V4-interleaved — main scale"
echo "    Started: $(date)"
conda run -n $CONDA_ENV python $TRAIN_SCRIPT \
  --variant swa_interleaved --scale main --compile \
  --data_dir $DATA_DIR_MAIN
echo "    Finished: $(date)"

echo ""
echo ">>> [8/10] V4-interleaved — stretch scale"
echo "    Started: $(date)"
conda run -n $CONDA_ENV python $TRAIN_SCRIPT \
  --variant swa_interleaved --scale stretch --compile \
  --data_dir $DATA_DIR_STRETCH
echo "    Finished: $(date)"

# --- V5: Linear (Linformer) ---
echo ""
echo ">>> [9/10] V5 Linear — main scale"
echo "    Started: $(date)"
conda run -n $CONDA_ENV python $TRAIN_SCRIPT \
  --variant linear --scale main --compile \
  --data_dir $DATA_DIR_MAIN
echo "    Finished: $(date)"

echo ""
echo ">>> [10/10] V5 Linear — stretch scale"
echo "    Started: $(date)"
conda run -n $CONDA_ENV python $TRAIN_SCRIPT \
  --variant linear --scale stretch --compile \
  --data_dir $DATA_DIR_STRETCH
echo "    Finished: $(date)"

echo ""
echo "============================================="
echo "All training complete!"
echo "Finished: $(date)"
echo "============================================="
