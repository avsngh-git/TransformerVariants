#!/bin/bash
# Train all 3 scales of the Modern (V1) model

echo "=== Training Modern V1 — Debug (16M) ==="
conda run -n transformer_lab python scripts/train.py \
    --variant modern \
    --data_dir data/processed/wikitext-full \
    --scale debug \
    --max_steps 2000 \
    --micro_batch_size 8 \
    --grad_accum_steps 4 \
    --warmup_steps 100 \
    --log_interval 100 \
    --eval_interval 500 \
    --checkpoint_interval 1000 \
    --dtype bfloat16 \
    --compile

echo ""
echo "=== Training Modern V1 — Main (51M) ==="
conda run -n transformer_lab python scripts/train.py \
    --variant modern \
    --data_dir data/processed/wikitext-100M \
    --scale main \
    --max_steps 3000 \
    --micro_batch_size 8 \
    --grad_accum_steps 8 \
    --warmup_steps 200 \
    --log_interval 200 \
    --eval_interval 1000 \
    --checkpoint_interval 1500 \
    --dtype bfloat16 \
    --compile

echo ""
echo "=== Training Modern V1 — Stretch (124M) ==="
conda run -n transformer_lab python scripts/train.py \
    --variant modern \
    --data_dir data/processed/wikitext-120M \
    --scale stretch \
    --max_steps 2000 \
    --micro_batch_size 4 \
    --grad_accum_steps 8 \
    --warmup_steps 200 \
    --log_interval 100 \
    --eval_interval 500 \
    --checkpoint_interval 1000 \
    --dtype bfloat16 \
    --compile

echo ""
echo "=== All V1 training complete ==="
