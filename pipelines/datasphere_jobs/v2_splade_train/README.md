# V2 SPLADEv2 Training — DataSphere Job

Trains a SPLADE model on MS MARCO passage triples using the correct SPLADEv2 recipe.

## Quick Start

```bash
# Local test (small dataset, ~2 min on GPU)
bash run.local.bash

# DataSphere cloud training (500K samples, ~2-3 hours on A100)
bash run.bash

# Resume from checkpoint
bash run.resume.bash checkpoints/checkpoint_step_5000.pt
```

## Key Improvements over v1

| Issue | v1 (Broken) | v2 (Fixed) |
|-------|------------|------------|
| Scoring | L2-normalized cosine sim | Raw dot product |
| Temperature | τ=0.1 (causes overflow) | τ=1.0 (none) |
| Special tokens | `[unused]`, `.`, `,` leak | Masked in vocab dimension |
| Learning rate | 5e-6 (too low) | 2e-5 (official) |
| LR schedule | None | Linear warmup + decay |
| Regularization | L1 (suboptimal) | FLOPS + quadratic warmup |
| Training data | 96K samples | 500K samples |
| Grad accumulation | None | 4× (effective bs=128) |

## Model Architecture

- **Base model**: `distilbert-base-uncased` (DistilBERT with MLM head)
- **Forward**: `log1p(ReLU(MLM_logits) × attention_mask × vocab_mask)`, max-pooled over sequence
- **Vocab size**: 30,522 (BERT WordPiece)
- **Special tokens masked**: [PAD], [CLS], [SEP], [MASK], [UNK], [unused0-99]

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup steps | 6000 |
| Batch size | 32 × 4 = 128 effective |
| Max sequence length | 128 |
| Epochs | 1 |
| Training samples | 500,000 |
| FP16 | Yes |
| FLOPS λ_q | 3e-4 |
| FLOPS λ_d | 1e-4 |
| FLOPS warmup (T) | 10,000 steps |
| Gradient clipping | max_norm=1.0 |

## Output Files

| File | Description |
|------|-------------|
| `model.pt` | Final model weights (state_dict only, for inference) |
| `training_log.jsonl` | Detailed training log (JSON Lines format) |
| `checkpoints/` | Intermediate checkpoints (full state for resume) |

## Training Log Format

The `training_log.jsonl` file contains one JSON object per line:

```jsonl
{"type": "config", "timestamp": 1234567890, "lr": 2e-5, ...}
{"type": "step", "step": 100, "avg_loss": 3.45, "avg_ce_loss": 3.44, "lr": 1.5e-6, ...}
{"type": "eval", "step": 1000, "queries": [{"query": "...", "top_tokens": [...], "nnz": 150}]}
{"type": "checkpoint", "step": 3000, "path": "checkpoints/checkpoint_step_3000.pt"}
{"type": "final", "total_steps": 15625, "total_time_sec": 7200, "final_loss": 0.85}
```

## Checkpoint Files

Checkpoints are saved at ~5 evenly-spaced intervals. Each checkpoint contains:
- `model_state_dict` — model weights
- `optimizer_state_dict` — AdamW state
- `scheduler_state_dict` — LR scheduler state
- `scaler_state_dict` — FP16 GradScaler state
- `step`, `epoch`, `loss`, `config`

To resume: `python main.py --resume checkpoints/checkpoint_step_5000.pt ...`

Additionally, **inference-only weights** are saved alongside each checkpoint as `model_step_NNNN.pt` — these can be directly used with the v1_custom_splade inference pipeline.

## Using Trained Model for Inference

Copy the output `model.pt` to the inference job:

```bash
cp model.pt ../v1_custom_splade/model.pt
cd ../v1_custom_splade
bash run.local.bash
```

The v2 model is compatible with the v1 inference pipeline because the SPLADE
class architecture is identical (the special token mask is applied during
training, causing those token weights to be near-zero naturally).
