#!/bin/bash
# Resume training from a checkpoint (local example)
# Usage: ./run.resume.bash checkpoints/checkpoint_step_5000.pt

set -euo pipefail

CHECKPOINT="${1:?Usage: $0 <checkpoint.pt>}"

echo "=== SPLADEv2 Training — Resume from $CHECKPOINT ==="

python main.py \
    --output model.pt \
    --log training_log.jsonl \
    --checkpoint-dir checkpoints \
    --resume "$CHECKPOINT" \
    --max-samples 500000 \
    --epochs 1 \
    --batch-size 32 \
    --grad-accum 4 \
    --lr 2e-5 \
    --warmup-steps 6000 \
    --num-checkpoints 5 \
    --fp16

echo ""
echo "=== Resume Training Complete ==="
