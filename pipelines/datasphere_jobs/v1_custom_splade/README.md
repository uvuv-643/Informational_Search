# V1 Custom SPLADE — MS MARCO Re-Ranking Benchmark

Custom SPLADE model trained on MS MARCO passage triples, evaluated via BM25 top-1000 re-ranking.

## Model Architecture

- **Base model**: `distilbert-base-uncased` (DistilBERT with MLM head)
- **Forward**: `ReLU → attention masking → max pooling → clamp(max=10) → log1p`
- **Vocab size**: 30,522 (BERT WordPiece)

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Base model | `distilbert-base-uncased` |
| Dataset | `msmarco-passage/train/triples-small` |
| Training samples | 96,000 (3 × 32,000) |
| Batch size | 32 |
| Epochs | 1 |
| Learning rate | 5e-6 |
| Optimizer | AdamW |
| Loss | Contrastive (cross-entropy) + L1 regularization |
| λ_query (L1) | 1e-3 |
| λ_doc (L1) | 1e-4 |
| Temperature (τ) | 0.1 |
| Max sequence length | 128 |
| Mixed precision | FP16 |
| Gradient clipping | max_norm=1.0 |
| Normalization | L2-normalize before scoring |

## Weights File

Place `model.pt` (the `state_dict()` checkpoint) in this directory before running.
The file is produced by `report/train.ipynb` via `torch.save(model.state_dict(), 'model.pt')`.

## Usage

```bash
# Local test
python main.py \
    --collection ../collection.tar.gz \
    --model model.pt \
    --max-queries 10 \
    --output metrics.json

# DataSphere Jobs
cp collection.tar.gz v1_custom_splade/
cp model.pt v1_custom_splade/
cd v1_custom_splade
datasphere project job execute -p <PROJECT_ID> -c config.yaml
```

## Expected Metrics

MRR@10 ≈ 0.078 (per report/inference.ipynb benchmarks)
