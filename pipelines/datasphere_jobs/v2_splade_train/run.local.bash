#!/bin/bash
# Local test run with small dataset (quick sanity check)
# Uses pre-packaged data if available, falls back to ir_datasets

set -euo pipefail

echo "=== SPLADEv2 Training — Local Test ==="

DATA_ARG=""
if [ -f "triplets.jsonl.gz" ]; then
    echo "Using pre-packaged data: triplets.jsonl.gz"
    DATA_ARG="--data triplets.jsonl.gz"
else
    echo "No triplets.jsonl.gz found, will use ir_datasets (slower)"
fi

# Small run: 1000 samples, 1 epoch, no gradient accumulation
python main.py \
    $DATA_ARG \
    --output model.pt \
    --log training_log.jsonl \
    --checkpoint-dir checkpoints \
    --max-samples 1000 \
    --epochs 1 \
    --batch-size 8 \
    --grad-accum 1 \
    --lr 2e-5 \
    --warmup-steps 10 \
    --num-checkpoints 2

echo ""
echo "=== Done ==="
echo "Model:  model.pt"
echo "Log:    training_log.jsonl"
echo "Ckpts:  checkpoints/"
