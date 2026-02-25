#!/bin/bash
# Local test run — downloads data via ir_datasets (no prepare_data needed)

set -euo pipefail

echo "=== SPLADEv2 Training — Local Test (ir_datasets) ==="

# Small run: 1000 samples, 1 epoch, no gradient accumulation
python main.py \
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
