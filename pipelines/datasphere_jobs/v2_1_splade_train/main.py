#!/usr/bin/env python3
"""SPLADEv2 Training Pipeline — DataSphere Jobs.

Trains a SPLADE model (distilbert-base-uncased + MLM head) on MS MARCO passage
triples using the correct SPLADEv2 recipe:
  - Raw dot-product scoring (no L2 normalization, no temperature)
  - FLOPS regularization with quadratic warmup
  - Special token masking in forward pass
  - Linear warmup + decay LR schedule
  - Gradient accumulation for effective batch size 128
  - Periodic checkpoints + detailed training log

Usage:
    python main.py --output model.pt --log training_log.jsonl
    python main.py --output model.pt --resume checkpoint_step_5000.pt
    python main.py --output model.pt --max-samples 100000 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

BASE_MODEL_NAME = "distilbert-base-uncased"
VOCAB_SIZE = 30522

# Special token IDs in BERT WordPiece vocabulary
# [PAD]=0, [unused1-99]=1..99, [UNK]=100, [CLS]=101, [SEP]=102, [MASK]=103
# Punctuation tokens (999-1066): . , - ! ? ; : ' ) ( | # & / + = @ { } ~
# These leak into sparse representations and waste capacity on non-semantic terms.
SPECIAL_TOKEN_IDS = sorted(set(
    [0, 100, 101, 102, 103]
    + list(range(1, 100))       # [unused0] .. [unused99]
    + list(range(999, 1067))    # all punctuation/symbol tokens
))

TEST_QUERIES = [
    "what is python programming",
    "how to lose weight",
    "best restaurants in paris",
    "covid vaccine side effects",
    "machine learning tutorial",
    "climate change causes",
    "how to cook pasta",
    "bitcoin price prediction",
    "yoga benefits health",
    "electric cars pros cons",
]


# ═══════════════════════════════════════════════════════════════════
# Training Configuration
# ═══════════════════════════════════════════════════════════════════

class TrainConfig:
    """All hyperparameters in one place for reproducibility.

    Plain class instead of dataclass for Python 3.9 compatibility
    with DataSphere's module loader.
    """

    def __init__(self, **kwargs):
        # Model
        self.model_name = kwargs.get("model_name", BASE_MODEL_NAME)
        self.max_length = kwargs.get("max_length", 128)

        # Data (0 = use entire dataset, ~39.8M triplets)
        self.dataset_name = kwargs.get("dataset_name", "msmarco-passage/train/triples-small")
        self.max_samples = kwargs.get("max_samples", 0)

        # Optimization
        self.lr = kwargs.get("lr", 2e-5)
        self.weight_decay = kwargs.get("weight_decay", 0.01)
        self.warmup_steps = kwargs.get("warmup_steps", 6000)
        self.batch_size = kwargs.get("batch_size", 32)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 4)
        self.epochs = kwargs.get("epochs", 1)
        self.max_grad_norm = kwargs.get("max_grad_norm", 1.0)
        self.fp16 = kwargs.get("fp16", True)

        # FLOPS regularization (reduced from 3e-4/1e-4 to avoid over-sparsification)
        self.flops_lambda_q = kwargs.get("flops_lambda_q", 1e-4)
        self.flops_lambda_d = kwargs.get("flops_lambda_d", 5e-5)
        self.flops_warmup_steps = kwargs.get("flops_warmup_steps", 10_000)

        # Checkpointing
        self.num_checkpoints = kwargs.get("num_checkpoints", 5)
        self.log_every = kwargs.get("log_every", 100)
        self.eval_every = kwargs.get("eval_every", 1000)

        # Resume
        self.resume_from = kwargs.get("resume_from", None)

    def to_dict(self):
        """Serialize config to dict for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ═══════════════════════════════════════════════════════════════════
# SPLADE Model
# ═══════════════════════════════════════════════════════════════════

class SPLADE(nn.Module):
    """SPLADEv2: DistilBERT MLM → ReLU → special-token mask → max-pool → log1p.

    Matches the official SPLADE encode() from naver/splade but adds explicit
    special-token zeroing in the vocabulary dimension to prevent [CLS]/[SEP]/
    [PAD]/[unused*] tokens from leaking into the sparse representation.
    """

    def __init__(self, model_name: str = BASE_MODEL_NAME):
        super().__init__()
        self.bert_mlm = AutoModelForMaskedLM.from_pretrained(model_name)
        self.vocab_size = self.bert_mlm.config.vocab_size

        # Pre-compute vocab mask: 1 for normal tokens, 0 for special tokens
        vocab_mask = torch.ones(self.vocab_size)
        vocab_mask[SPECIAL_TOKEN_IDS] = 0.0
        self.register_buffer("vocab_mask", vocab_mask)  # (vocab_size,)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse vocab-size representation.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            Sparse representation (batch_size, vocab_size), non-negative.
        """
        outputs = self.bert_mlm(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (bs, seq_len, vocab_size)

        # ReLU activation → non-negative
        relu_logits = F.relu(logits)

        # Mask padding positions (sequence dimension)
        relu_logits = relu_logits * attention_mask.unsqueeze(-1)

        # Mask special tokens (vocabulary dimension)
        relu_logits = relu_logits * self.vocab_mask.unsqueeze(0).unsqueeze(0)

        # Max pooling over sequence length
        pooled, _ = torch.max(relu_logits, dim=1)  # (bs, vocab_size)

        # Log saturation: log(1 + x) compresses large values
        return torch.log1p(pooled)


# ═══════════════════════════════════════════════════════════════════
# FLOPS Regularization
# ═══════════════════════════════════════════════════════════════════

def flops_loss(batch_rep: torch.Tensor) -> torch.Tensor:
    """FLOPS regularization from Paria et al., 2020.

    Penalizes the sum of squared mean activations per vocab dimension.
    This encourages the model to distribute activations sparsely across
    the vocabulary rather than concentrating on a few high-frequency terms.

    L_FLOPS = sum_j ( mean_i |r_ij| )^2

    Args:
        batch_rep: (batch_size, vocab_size) — sparse representations

    Returns:
        Scalar FLOPS loss.
    """
    return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)


class RegWeightScheduler:
    """Quadratic warmup scheduler for regularization weight.

    lambda_t = lambda_ * min(1, (t / T)^2)

    Starts at 0 and ramps up to lambda_ over T steps.
    """

    def __init__(self, lambda_: float, T: int):
        self.lambda_ = lambda_
        self.T = T
        self.t = 0

    def step(self) -> float:
        self.t += 1
        ratio = min(1.0, (self.t / self.T) ** 2)
        return self.lambda_ * ratio

    @property
    def current(self) -> float:
        ratio = min(1.0, (self.t / self.T) ** 2)
        return self.lambda_ * ratio


# ═══════════════════════════════════════════════════════════════════
# Loss Function
# ═══════════════════════════════════════════════════════════════════

def compute_loss(
    q_reps: torch.Tensor,
    p_reps: torch.Tensor,
    n_reps: torch.Tensor,
    lambda_q: float,
    lambda_d: float,
) -> tuple[torch.Tensor, dict]:
    """SPLADEv2 training loss: cross-entropy on in-batch negatives + FLOPS.

    Scoring is raw dot product — NO L2 normalization, NO temperature.

    Args:
        q_reps: (batch_size, vocab_size) — query representations
        p_reps: (batch_size, vocab_size) — positive doc representations
        n_reps: (batch_size, vocab_size) — negative doc representations
        lambda_q: FLOPS regularization weight for queries
        lambda_d: FLOPS regularization weight for documents

    Returns:
        (total_loss, detail_dict) where detail_dict has component losses.
    """
    batch_size = q_reps.size(0)

    # Concatenate positive and negative docs: (2 * batch_size, vocab_size)
    all_docs = torch.cat([p_reps, n_reps], dim=0)

    # Raw dot-product scores: (batch_size, 2 * batch_size)
    # No normalization, no temperature
    scores = torch.matmul(q_reps, all_docs.T)

    # Labels: positive doc for query i is at index i
    labels = torch.arange(batch_size, device=q_reps.device, dtype=torch.long)
    ce_loss = F.cross_entropy(scores, labels)

    # FLOPS regularization
    flops_q = flops_loss(q_reps)
    flops_d = (flops_loss(p_reps) + flops_loss(n_reps)) / 2.0

    reg_loss = lambda_q * flops_q + lambda_d * flops_d
    total_loss = ce_loss + reg_loss

    details = {
        "ce_loss": ce_loss.item(),
        "flops_q": flops_q.item(),
        "flops_d": flops_d.item(),
        "reg_loss": reg_loss.item(),
        "total_loss": total_loss.item(),
        "lambda_q": lambda_q,
        "lambda_d": lambda_d,
    }
    return total_loss, details


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

def _is_gzip_file(path):
    """Detect gzip by magic bytes (0x1f 0x8b), not file extension.

    DataSphere renames input files (e.g. triplets.jsonl.gz -> /job/input_data_y5z),
    so we cannot rely on the extension.
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(2)
            return magic == b"\x1f\x8b"
    except Exception:
        return False


def load_triplets_from_file(data_path, max_samples):
    """Load pre-packaged triplets from JSONL(.gz) file.

    Format: one JSON object per line with keys "q", "p", "n".
    Supports both plain JSONL and gzip-compressed JSONL.
    Gzip detection uses magic bytes, not file extension (DataSphere renames files).
    """
    import gzip

    is_gz = _is_gzip_file(data_path)
    use_all = (max_samples <= 0)
    print(f"Loading triplets from {data_path} (gzip={is_gz}, max={'ALL' if use_all else max_samples})...")
    data = []
    opener = gzip.open if is_gz else open
    with opener(data_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading triplets", total=None if use_all else max_samples)):
            if not use_all and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append((obj["q"], obj["p"], obj["n"]))
    print(f"  -> {len(data):,} triplets loaded from file")
    return data


def load_triplets_from_ir_datasets(max_samples):
    """Load triplets via ir_datasets (downloads if needed, ~20-30 min first time)."""
    import ir_datasets

    dataset = ir_datasets.load("msmarco-passage/train/triples-small")

    print("Loading queries...")
    queries = {q.query_id: q.text for q in tqdm(dataset.queries_iter(), desc="Queries")}
    print(f"  -> {len(queries):,} queries")

    print("Loading documents...")
    docs = {d.doc_id: d.text for d in tqdm(dataset.docs_iter(), desc="Docs")}
    print(f"  -> {len(docs):,} documents")

    use_all = (max_samples <= 0)
    print(f"Loading triplets (max={'ALL' if use_all else max_samples})...")
    data = []
    for i, item in enumerate(tqdm(dataset.docpairs_iter(), desc="Triplets")):
        if not use_all and i >= max_samples:
            break
        data.append((
            queries[item.query_id],
            docs[item.doc_id_a],
            docs[item.doc_id_b],
        ))
    print(f"  -> {len(data):,} triplets loaded from ir_datasets")
    return data


class MSMARCOTriplesDataset(Dataset):
    """MS MARCO passage triples dataset.

    Supports two loading modes:
      1. Pre-packaged JSONL(.gz) file (fast, for DataSphere upload)
      2. ir_datasets (downloads from web, slow first time)

    Tokenizes on-the-fly, keeps raw text in memory.
    """

    def __init__(self, tokenizer, max_length=128, max_samples=0, data_path=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if data_path and os.path.exists(data_path):
            self.data = load_triplets_from_file(data_path, max_samples)
        else:
            if data_path:
                print(f"WARNING: {data_path} not found, falling back to ir_datasets")
            self.data = load_triplets_from_ir_datasets(max_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query_text, pos_text, neg_text = self.data[idx]

        query = self.tokenizer(
            query_text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        pos_doc = self.tokenizer(
            pos_text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        neg_doc = self.tokenizer(
            neg_text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "q_ids": query["input_ids"].squeeze(0),
            "q_mask": query["attention_mask"].squeeze(0),
            "p_ids": pos_doc["input_ids"].squeeze(0),
            "p_mask": pos_doc["attention_mask"].squeeze(0),
            "n_ids": neg_doc["input_ids"].squeeze(0),
            "n_mask": neg_doc["attention_mask"].squeeze(0),
        }


# ═══════════════════════════════════════════════════════════════════
# Training Log Writer
# ═══════════════════════════════════════════════════════════════════

class TrainingLogger:
    """Writes JSON Lines log for post-hoc analysis and debugging.

    Each line is a JSON object with one of these types:
      - "config":     full training configuration
      - "step":       per-step metrics (loss components, LR, etc.)
      - "eval":       test query encoding results
      - "checkpoint": checkpoint save event
      - "final":      final training summary
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._fh = open(log_path, "a", encoding="utf-8")

    def log(self, record: dict):
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def log_config(self, config):
        self.log({"type": "config", "timestamp": time.time(), **config.to_dict()})

    def log_step(self, step: int, epoch: int, **metrics):
        self.log({"type": "step", "timestamp": time.time(), "step": step, "epoch": epoch, **metrics})

    def log_eval(self, step: int, query_results: list):
        self.log({"type": "eval", "timestamp": time.time(), "step": step, "queries": query_results})

    def log_checkpoint(self, step: int, path: str, loss: float):
        self.log({"type": "checkpoint", "timestamp": time.time(), "step": step, "path": path, "loss": loss})

    def log_final(self, total_steps: int, total_time: float, final_loss: float):
        self.log({
            "type": "final", "timestamp": time.time(),
            "total_steps": total_steps, "total_time_sec": total_time, "final_loss": final_loss,
        })

    def close(self):
        self._fh.close()


# ═══════════════════════════════════════════════════════════════════
# Test Query Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_test_queries(
    model: SPLADE, tokenizer, device: torch.device, top_k: int = 20,
) -> list[dict]:
    """Encode test queries and return top-K tokens with weights.

    This shows the learned sparse representation quality.
    Good model: semantic expansion tokens with weights 0.5-10.
    Bad model: punctuation / [unused] tokens dominating.
    """
    model.eval()
    results = []

    for query in TEST_QUERIES:
        tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        q_rep = model(
            tokens["input_ids"].to(device),
            tokens["attention_mask"].to(device),
        )  # (1, vocab_size)

        topk = torch.topk(q_rep[0], k=top_k)
        top_tokens = [
            (tokenizer.decode([idx.item()]), round(weight.item(), 4))
            for idx, weight in zip(topk.indices, topk.values)
        ]

        # Stats
        nnz = (q_rep[0] > 0.01).sum().item()
        max_val = q_rep[0].max().item()
        mean_nz = q_rep[0][q_rep[0] > 0.01].mean().item() if nnz > 0 else 0.0

        results.append({
            "query": query,
            "top_tokens": top_tokens,
            "nnz": nnz,
            "max_weight": round(max_val, 4),
            "mean_nz_weight": round(mean_nz, 4),
        })

    model.train()
    return results


def print_eval_results(results: list[dict]):
    """Pretty-print test query evaluation results."""
    print("\n" + "=" * 70)
    print("TEST QUERY EVALUATION")
    print("=" * 70)
    for r in results:
        tokens_str = ", ".join(f"{tok}:{w:.2f}" for tok, w in r["top_tokens"][:10])
        print(f"\n  Q: {r['query']}")
        print(f"  Top-10: [{tokens_str}]")
        print(f"  nnz={r['nnz']}, max={r['max_weight']:.2f}, mean_nz={r['mean_nz_weight']:.2f}")
    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════
# Checkpoint Management
# ═══════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: SPLADE,
    optimizer,
    scheduler,
    scaler,
    step: int,
    epoch: int,
    loss: float,
    config: TrainConfig,
    path: str,
):
    """Save a full training checkpoint (model + optimizer + scheduler state)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "config": config.to_dict(),
    }, path)
    print(f"  💾 Checkpoint saved: {path} (step={step}, loss={loss:.4f})")


def load_checkpoint(path: str, model: SPLADE, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    """Load a training checkpoint. Returns (step, epoch, loss)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"  📂 Checkpoint loaded: {path} (step={ckpt['step']}, loss={ckpt.get('loss', '?')})")
    return ckpt["step"], ckpt["epoch"], ckpt.get("loss", 0.0)


def save_model_weights(model: SPLADE, path: str):
    """Save only model state_dict (for inference)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  📦 Model weights saved: {path}")


def _cleanup_old_checkpoints(checkpoint_dir, keep=2):
    """Keep only the last `keep` checkpoints (+ their model_step_* weights).

    Pairs checkpoint_step_NNNN.pt and model_step_NNNN.pt are treated as one unit.
    checkpoint_final.pt is always kept.
    """
    import glob
    import re

    # Find all checkpoint_step_*.pt files, extract step numbers
    pattern = os.path.join(checkpoint_dir, "checkpoint_step_*.pt")
    ckpts = glob.glob(pattern)
    if len(ckpts) <= keep:
        return  # nothing to clean

    # Sort by step number
    def _step(path):
        m = re.search(r"checkpoint_step_(\d+)\.pt$", path)
        return int(m.group(1)) if m else 0

    ckpts.sort(key=_step)

    # Delete all but the last `keep`
    to_delete = ckpts[:-keep]
    for ckpt in to_delete:
        step_num = _step(ckpt)
        weights = os.path.join(checkpoint_dir, f"model_step_{step_num}.pt")
        for f in [ckpt, weights]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  🗑️  Removed old checkpoint: {os.path.basename(f)}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPLADEv2 training on MS MARCO")
    p.add_argument("--output", required=True, help="Output model.pt path (final weights)")
    p.add_argument("--log", required=True, help="Training log JSONL path")
    p.add_argument("--checkpoint-dir", default="checkpoints", help="Dir for intermediate checkpoints")
    p.add_argument("--resume", default=None, help="Resume from checkpoint .pt file")
    p.add_argument("--max-samples", type=int, default=0, help="Max training triplets (0 = all ~39.8M)")
    p.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--warmup-steps", type=int, default=6000, help="LR warmup steps")
    p.add_argument("--data", default=None, help="Pre-packaged triplets.jsonl.gz (from prepare_data.py). Falls back to ir_datasets if not provided.")
    p.add_argument("--num-checkpoints", type=int, default=5, help="Number of intermediate checkpoints")
    p.add_argument("--fp16", action="store_true", default=True, help="Use FP16 mixed precision")
    p.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Build config
    config = TrainConfig(
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        num_checkpoints=args.num_checkpoints,
        fp16=args.fp16,
        resume_from=args.resume,
    )

    # Initialize logger
    logger = TrainingLogger(args.log)
    logger.log_config(config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    print(f"Tokenizer: {config.model_name} (vocab_size={tokenizer.vocab_size})")

    # Dataset
    data_path = args.data
    if data_path:
        print(f"\nLoading dataset from pre-packaged file: {data_path}")
    else:
        print(f"\nLoading dataset via ir_datasets: {config.dataset_name}")
    dataset = MSMARCOTriplesDataset(
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_samples=config.max_samples,
        data_path=data_path,
    )
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    total_steps_per_epoch = len(dataloader)
    total_steps = total_steps_per_epoch * config.epochs
    effective_steps = total_steps // config.gradient_accumulation_steps
    print(f"\nTraining plan:")
    print(f"  Samples:          {len(dataset):,}")
    print(f"  Batch size:       {config.batch_size} × {config.gradient_accumulation_steps} accum = {config.batch_size * config.gradient_accumulation_steps} effective")
    print(f"  Steps/epoch:      {total_steps_per_epoch:,}")
    print(f"  Total steps:      {total_steps:,} ({effective_steps:,} optimizer steps)")
    print(f"  Epochs:           {config.epochs}")

    # Model
    print(f"\nInitializing SPLADE model: {config.model_name}")
    model = SPLADE(config.model_name).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,} (trainable: {trainable_count:,})")

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=effective_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16 and device.type == "cuda")

    # FLOPS regularization schedulers
    reg_q = RegWeightScheduler(config.flops_lambda_q, config.flops_warmup_steps)
    reg_d = RegWeightScheduler(config.flops_lambda_d, config.flops_warmup_steps)

    # Resume from checkpoint
    start_step = 0
    start_epoch = 0
    if config.resume_from and os.path.exists(config.resume_from):
        start_step, start_epoch, _ = load_checkpoint(
            config.resume_from, model, optimizer, scheduler, scaler, device,
        )
        # Fast-forward regularization schedulers
        for _ in range(start_step):
            reg_q.step()
            reg_d.step()
        print(f"  Resuming from step {start_step}, epoch {start_epoch}")

    # Checkpoint schedule: evenly spaced across training
    checkpoint_interval = max(1, total_steps // max(1, config.num_checkpoints))
    print(f"\n  Checkpoint interval: every {checkpoint_interval:,} steps")
    print(f"  Log interval:       every {config.log_every} steps")
    print(f"  Eval interval:      every {config.eval_every} steps")

    # Initial evaluation
    print("\n--- Initial model evaluation ---")
    eval_results = evaluate_test_queries(model, tokenizer, device)
    print_eval_results(eval_results)
    logger.log_eval(step=start_step, query_results=eval_results)

    # ─── Training ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")

    model.train()
    global_step = start_step
    optimizer_step = start_step // config.gradient_accumulation_steps
    running_loss = 0.0
    running_ce = 0.0
    running_reg = 0.0
    running_count = 0
    train_start_time = time.time()
    last_loss = 0.0

    for epoch in range(start_epoch, config.epochs):
        print(f"\n── Epoch {epoch + 1}/{config.epochs} ──")

        pbar = tqdm(
            enumerate(dataloader), total=total_steps_per_epoch,
            desc=f"Epoch {epoch + 1}", dynamic_ncols=True,
        )

        for batch_idx, batch in pbar:
            # Skip already-processed steps when resuming
            if epoch == start_epoch and batch_idx < (start_step % total_steps_per_epoch):
                continue

            global_step += 1

            # Step regularization schedulers
            lq = reg_q.step()
            ld = reg_d.step()

            # Forward pass
            with torch.cuda.amp.autocast(enabled=config.fp16 and device.type == "cuda"):
                q_reps = model(batch["q_ids"].to(device), batch["q_mask"].to(device))
                p_reps = model(batch["p_ids"].to(device), batch["p_mask"].to(device))
                n_reps = model(batch["n_ids"].to(device), batch["n_mask"].to(device))
                loss, loss_details = compute_loss(q_reps, p_reps, n_reps, lq, ld)
                loss = loss / config.gradient_accumulation_steps

            # Backward
            scaler.scale(loss).backward()

            # Optimizer step (every gradient_accumulation_steps)
            if global_step % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            # Accumulate metrics
            running_loss += loss_details["total_loss"]
            running_ce += loss_details["ce_loss"]
            running_reg += loss_details["reg_loss"]
            running_count += 1
            last_loss = loss_details["total_loss"]

            # Progress bar
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{loss_details['total_loss']:.3f}",
                "ce": f"{loss_details['ce_loss']:.3f}",
                "reg": f"{loss_details['reg_loss']:.4f}",
                "lr": f"{current_lr:.2e}",
            })

            # Log metrics
            if global_step % config.log_every == 0 and running_count > 0:
                avg_loss = running_loss / running_count
                avg_ce = running_ce / running_count
                avg_reg = running_reg / running_count

                logger.log_step(
                    step=global_step,
                    epoch=epoch,
                    avg_loss=round(avg_loss, 6),
                    avg_ce_loss=round(avg_ce, 6),
                    avg_reg_loss=round(avg_reg, 6),
                    last_loss=round(loss_details["total_loss"], 6),
                    last_ce=round(loss_details["ce_loss"], 6),
                    last_flops_q=round(loss_details["flops_q"], 4),
                    last_flops_d=round(loss_details["flops_d"], 4),
                    lambda_q=round(lq, 8),
                    lambda_d=round(ld, 8),
                    lr=current_lr,
                    optimizer_step=optimizer_step,
                    elapsed_sec=round(time.time() - train_start_time, 1),
                )

                tqdm.write(
                    f"  [step {global_step:>6d}] "
                    f"loss={avg_loss:.4f} (ce={avg_ce:.4f} reg={avg_reg:.5f}) "
                    f"lr={current_lr:.2e} λ_q={lq:.6f} λ_d={ld:.6f}"
                )
                running_loss = 0.0
                running_ce = 0.0
                running_reg = 0.0
                running_count = 0

            # Evaluate test queries
            if global_step % config.eval_every == 0:
                eval_results = evaluate_test_queries(model, tokenizer, device)
                print_eval_results(eval_results)
                logger.log_eval(step=global_step, query_results=eval_results)
                model.train()

            # Save checkpoint (keep only last num_checkpoints)
            if global_step % checkpoint_interval == 0:
                ckpt_path = os.path.join(
                    args.checkpoint_dir,
                    f"checkpoint_step_{global_step}.pt",
                )
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    global_step, epoch, last_loss, config, ckpt_path,
                )
                # Also save inference-only weights
                weights_path = os.path.join(
                    args.checkpoint_dir,
                    f"model_step_{global_step}.pt",
                )
                save_model_weights(model, weights_path)
                logger.log_checkpoint(step=global_step, path=ckpt_path, loss=last_loss)

                # Cleanup: keep only the last N checkpoints
                _cleanup_old_checkpoints(args.checkpoint_dir, keep=config.num_checkpoints)

    # ─── Final ──────────────────────────────────────────────────
    total_time = time.time() - train_start_time
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"  Total steps:  {global_step:,}")
    print(f"  Total time:   {total_time / 60:.1f} min")
    print(f"  Final loss:   {last_loss:.4f}")
    print(f"{'=' * 70}")

    # Final evaluation
    print("\n--- Final model evaluation ---")
    eval_results = evaluate_test_queries(model, tokenizer, device)
    print_eval_results(eval_results)
    logger.log_eval(step=global_step, query_results=eval_results)

    # Save final model
    save_model_weights(model, args.output)
    print(f"\n✅ Final model saved to: {args.output}")

    # Save final checkpoint (replaces oldest if over limit)
    final_ckpt = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, scheduler, scaler, global_step, config.epochs, last_loss, config, final_ckpt)
    _cleanup_old_checkpoints(args.checkpoint_dir, keep=config.num_checkpoints)

    # Final log
    logger.log_final(total_steps=global_step, total_time=total_time, final_loss=last_loss)
    logger.close()
    print(f"📝 Training log saved to: {args.log}")


if __name__ == "__main__":
    main()
