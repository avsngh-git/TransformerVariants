#!/usr/bin/env bash
# Train V2 (ALiBi) and V3 (GQA) variants.
#
# Usage:
#   ./scripts/train_v2_v3.sh <scale> [data_dir]
#
# Arguments:
#   scale     One of: debug, main, stretch
#             - debug:   Fast iteration (4 layers, 256d, 2000 steps)
#             - main:    Primary benchmark (~51M params, 5000 steps)
#             - stretch: Near-memory-limit (~124M params, 5000 steps)
#
#   data_dir  Path to tokenized shard directory (default: data/wikitext)
#
# Examples:
#   ./scripts/train_v2_v3.sh debug
#   ./scripts/train_v2_v3.sh main data/openwebtext
#   ./scripts/train_v2_v3.sh stretch data/wikitext --compile
#
# Notes:
#   - Both variants require CUDA and the flash_attn library
#   - ALiBi (V2) replaces RoPE with linear position biases
#   - GQA (V3) uses fewer KV heads (n_head/4) for ~6% param reduction
#   - Add --compile flag for ~15-25% speedup with torch.compile

set -euo pipefail

SCALE="${1:-debug}"
DATA_DIR="${2:-data/wikitext}"
shift 2 2>/dev/null || true  # consume positional args, remaining go as extra flags
EXTRA_FLAGS="$@"

# Validate scale
if [[ "$SCALE" != "debug" && "$SCALE" != "main" && "$SCALE" != "stretch" ]]; then
    echo "Error: scale must be one of: debug, main, stretch"
    echo "Usage: $0 <scale> [data_dir] [extra flags...]"
    exit 1
fi

echo "=============================================="
echo " Training V2 (ALiBi) and V3 (GQA)"
echo " Scale: $SCALE"
echo " Data:  $DATA_DIR"
echo "=============================================="
echo

# --- Train V2: ALiBi ---
echo ">>> Starting V2 (ALiBi) training at $SCALE scale..."
echo "    Position encoding: linear biases (no RoPE)"
echo "    Backend: flash_attn"
echo
python scripts/train.py \
    --variant alibi \
    --scale "$SCALE" \
    --data_dir "$DATA_DIR" \
    $EXTRA_FLAGS

echo
echo ">>> V2 (ALiBi) training complete."
echo

# --- Train V3: GQA ---
echo ">>> Starting V3 (GQA) training at $SCALE scale..."
echo "    Attention: grouped-query (n_kv_head = n_head/4)"
echo "    Backend: flash_attn"
echo
python scripts/train.py \
    --variant gqa \
    --scale "$SCALE" \
    --data_dir "$DATA_DIR" \
    $EXTRA_FLAGS

echo
echo ">>> V3 (GQA) training complete."
echo
echo "=============================================="
echo " Both variants trained successfully!"
echo " Checkpoints: checkpoints/alibi_$SCALE/"
echo "              checkpoints/gqa_$SCALE/"
echo "=============================================="
