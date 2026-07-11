#!/bin/bash
# ============================================================================
# Prepare 1B tokens from FineWeb-Edu for the controlled experiment
# ============================================================================
#
# Downloads and tokenizes 1B tokens from FineWeb-Edu (sample-10BT)
# into uint16 binary shards. Uses the streaming pipeline for constant
# memory usage regardless of corpus size.
#
# Output: data/processed/fineweb-1B/
#   - train_000000.bin ... train_000099.bin (~100 shards of 10M tokens each)
#   - val_000000.bin (small validation split, ~1% = ~10M tokens)
#   - manifest.json (shard metadata compatible with ShardedDataLoader)
#
# Estimated time: 30-60 minutes depending on network speed
# Estimated disk: ~2 GB
#
# Usage:
#   bash scripts/prepare_1B_data.sh 2>&1 | tee data_prep_1B.log
#
# To resume interrupted preparation:
#   bash scripts/prepare_1B_data.sh  (automatically resumes via progress.json)
# ============================================================================

set -e

CONDA_ENV="transformer_lab"
OUTPUT_DIR="data/processed/fineweb-1B"
MAX_TOKENS=1000000000  # 1 billion tokens
CONFIG="configs/data/fineweb_edu.yaml"

echo "============================================================"
echo "Data Preparation: 1B tokens from FineWeb-Edu"
echo "============================================================"
echo "  Source: HuggingFaceFW/fineweb-edu (sample-10BT)"
echo "  Max tokens: $(printf "%'d" $MAX_TOKENS)"
echo "  Output: $OUTPUT_DIR"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# Check if data already exists and is complete
if [ -f "$OUTPUT_DIR/manifest.json" ]; then
  echo "Data directory exists with manifest. Checking if complete..."
  TOKEN_COUNT=$(python -c "
import json
with open('$OUTPUT_DIR/manifest.json') as f:
    m = json.load(f)
train_tokens = m.get('train_tokens', 0)
val_tokens = m.get('val_tokens', 0)
print(train_tokens + val_tokens)
" 2>/dev/null || echo "0")
  
  if [ "$TOKEN_COUNT" -ge 900000000 ]; then
    echo "Dataset already prepared ($TOKEN_COUNT tokens). Skipping."
    exit 0
  else
    echo "Dataset incomplete ($TOKEN_COUNT tokens). Resuming..."
  fi
fi

# Run the streaming pipeline
conda run -n $CONDA_ENV python scripts/prepare_streaming.py \
  --config $CONFIG \
  --max-tokens $MAX_TOKENS \
  --output-dir $OUTPUT_DIR \
  --resume

echo ""
echo "============================================================"
echo "Data preparation complete!"
echo "  Output: $OUTPUT_DIR"
echo "  Finished: $(date)"
echo "============================================================"
echo ""
echo "Verify with:"
echo "  ls -la $OUTPUT_DIR/*.bin | wc -l"
echo "  cat $OUTPUT_DIR/manifest.json | python -m json.tool | head -20"
