#!/usr/bin/env python3
"""Custom SPLADE (v1) Re-Ranking Pipeline on MS MARCO Dev — DataSphere Jobs.

This script benchmarks a custom-trained SPLADE model (distilbert-base-uncased + MLM head)
loaded from a model.pt state_dict file, using the same BM25 top-1000 re-ranking protocol
as the reference SPLADE pipeline.

Usage:
    python main.py --collection collection.tar.gz --model model.pt --max-queries 100 --output metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tarfile
import tempfile
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


# ─────────────────────────── Constants ───────────────────────────

BASE_MODEL_NAME = "distilbert-base-uncased"
TOKENIZER_NAME = "distilbert-base-uncased"
IR_DATASET_NAME = "msmarco-passage/dev/small"
BATCH_SIZE_DEFAULT = 32
MAX_LENGTH = 256
TOP_K_BM25 = 1000
LOG_EVERY = 100

# Special token IDs in BERT WordPiece vocabulary
# [PAD]=0, [unused1-99]=1..99, [UNK]=100, [CLS]=101, [SEP]=102, [MASK]=103
# Punctuation tokens (999-1066): . , - ! ? ; : ' ) ( | # & / + = @ { } ~
SPECIAL_TOKEN_IDS = sorted(set(
    [0, 100, 101, 102, 103]
    + list(range(1, 100))       # [unused0] .. [unused99]
    + list(range(999, 1067))    # all punctuation/symbol tokens
))


# ─────────────────────────── Model Definition ────────────────────
# Must match the architecture used in training (v2_splade_train/main.py)

class SPLADE(nn.Module):
    """SPLADEv2: DistilBERT MLM head → ReLU → special-token mask → max pool → log1p.

    This matches the v2 training architecture with special token masking
    in the vocabulary dimension to prevent [CLS]/[SEP]/[PAD]/[unused*] leakage.
    Also backward-compatible with v1 checkpoints (special tokens will just be near-zero).
    """

    def __init__(self, model_name=BASE_MODEL_NAME):
        super().__init__()
        self.bert_mlm = AutoModelForMaskedLM.from_pretrained(model_name)
        self.vocab_size = self.bert_mlm.config.vocab_size

        # Pre-compute vocab mask: 1 for normal tokens, 0 for special tokens
        vocab_mask = torch.ones(self.vocab_size)
        vocab_mask[SPECIAL_TOKEN_IDS] = 0.0
        self.register_buffer("vocab_mask", vocab_mask)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_mlm(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits                                  # (bs, seq_len, vocab_size)
        relu_logits = F.relu(logits)
        relu_logits = relu_logits * attention_mask.unsqueeze(-1) # mask padding positions
        relu_logits = relu_logits * self.vocab_mask              # mask special tokens in vocab
        pooled, _ = torch.max(relu_logits, dim=1)               # (bs, vocab_size)
        return torch.log1p(pooled)                               # (bs, vocab_size)


# ─────────────────────────── CLI ─────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Custom SPLADE (v1) re-ranking on MS MARCO dev"
    )
    parser.add_argument(
        "--collection", required=True,
        help="Path to collection.tar.gz"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to model.pt (state_dict checkpoint)"
    )
    parser.add_argument(
        "--max-queries", type=int, required=True,
        help="Number of dev queries to process (0 = all ~6980)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE_DEFAULT,
        help=f"Batch size for encoding (default: {BATCH_SIZE_DEFAULT})"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for output metrics JSON"
    )
    return parser.parse_args()


# ─────────────────────────── Evaluation ──────────────────────────

def evaluate_run(run, qrels, metrics=("recall_10", "recall_100", "recall_1000", "recip_rank")):
    import pytrec_eval
    qrels_filtered = {qid: rels for qid, rels in qrels.items() if qid in run}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_filtered, set(metrics))
    results = evaluator.evaluate(run)
    agg = {}
    for metric in metrics:
        values = [r[metric] for r in results.values() if metric in r]
        agg[metric] = sum(values) / max(1, len(values))
    return agg


def find_relevant_rank(scored_docs, relevant_doc_ids):
    sorted_dids = sorted(scored_docs, key=scored_docs.get, reverse=True)
    for rank, did in enumerate(sorted_dids, 1):
        if did in relevant_doc_ids:
            return rank
    return None


# ─────────────────────────── Reranker ────────────────────────────

class CustomSpladeReranker:
    """Custom SPLADE reranker with sparse doc cache."""

    def __init__(self, model_path, batch_size, max_length, device, collection):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.collection = collection

        # Load model architecture + trained weights
        # strict=False allows loading v1 checkpoints (no vocab_mask buffer)
        print(f"Loading custom SPLADE model from {model_path}")
        self.model = SPLADE(BASE_MODEL_NAME).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        print(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.doc_rep_cache = {}  # sparse COO cache

    @staticmethod
    def _to_sparse(dense_vec, threshold=1e-4):
        """Convert dense to sparse, zeroing values below threshold first.
        
        The custom SPLADE's log1p(x + 1e-8) makes ALL values non-zero.
        Without thresholding, sparse COO gives zero memory savings.
        With threshold=1e-4, we keep only ~100-300 meaningful activations
        per doc, reducing from 122 KB to ~2-4 KB per doc.
        """
        dense_vec = dense_vec.clone()
        dense_vec[dense_vec.abs() < threshold] = 0.0
        return dense_vec.to_sparse_coo().coalesce()

    @staticmethod
    def _sparse_to_dense(sparse_vec):
        return sparse_vec.to_dense()

    def _encode_batch(self, texts):
        """Encode a list of texts. Returns dense (len(texts), vocab_size) tensor."""
        tokens = self.tokenizer(
            texts, return_tensors="pt",
            truncation=True, max_length=self.max_length, padding=True,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        with torch.no_grad():
            reps = self.model(input_ids, attention_mask)  # (batch, vocab_size)
        return reps

    def encode_query(self, query_text):
        """Encode a single query. Returns 1-D dense CPU tensor."""
        reps = self._encode_batch([query_text])
        return reps.squeeze(0).cpu()

    def encode_docs_cached(self, doc_ids):
        """Encode docs with sparse cache. Returns dense (n, vocab_size)."""
        uncached = [did for did in doc_ids if did not in self.doc_rep_cache]
        if uncached:
            for i in range(0, len(uncached), self.batch_size):
                batch_ids = uncached[i: i + self.batch_size]
                batch_texts = [self.collection[did] for did in batch_ids]
                reps = self._encode_batch(batch_texts)
                for j, did in enumerate(batch_ids):
                    self.doc_rep_cache[did] = self._to_sparse(reps[j].cpu())
        return torch.stack([self._sparse_to_dense(self.doc_rep_cache[did]) for did in doc_ids])

    def rerank(self, query_text, candidate_doc_ids):
        """Re-rank candidates. Returns {doc_id: score}."""
        q_vec = self.encode_query(query_text)
        d_matrix = self.encode_docs_cached(candidate_doc_ids)
        scores = torch.matmul(d_matrix, q_vec).tolist()
        del d_matrix
        return {did: score for did, score in zip(candidate_doc_ids, scores)}

    @property
    def cache_size(self):
        return len(self.doc_rep_cache)


# ─────────────────────────── Main ────────────────────────────────

def main():
    args = parse_args()
    max_queries = args.max_queries if args.max_queries > 0 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load queries + qrels
    import ir_datasets
    print(f"Loading dataset: {IR_DATASET_NAME}")
    dataset = ir_datasets.load(IR_DATASET_NAME)

    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    print(f"Total queries: {len(queries)}")

    qrels = {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    print(f"Queries with judgments: {len(qrels)}")

    # BM25 top-1000
    print("Loading BM25 top-1000...")
    bm25_run = defaultdict(dict)
    for sd in tqdm(dataset.scoreddocs_iter(), desc="BM25"):
        qid = sd.query_id
        if max_queries and len(bm25_run) >= max_queries and qid not in bm25_run:
            continue
        if len(bm25_run[qid]) < TOP_K_BM25:
            bm25_run[qid][sd.doc_id] = sd.score
    bm25_run = dict(bm25_run)
    queries = {qid: t for qid, t in queries.items() if qid in bm25_run}
    print(f"Queries: {len(queries)}")

    # BM25 baseline
    bm25_run_eval = {qid: {did: float(s) for did, s in docs.items()} for qid, docs in bm25_run.items()}
    bm25_metrics = evaluate_run(bm25_run_eval, qrels)
    print("\n=== BM25 Baseline ===")
    for m, v in bm25_metrics.items():
        print(f"  {m:20s} = {v:.4f}")

    # Extract & load collection
    print(f"\nExtracting {args.collection}...")
    extract_dir = tempfile.mkdtemp(prefix="collection_")
    with tarfile.open(args.collection, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    collection_path = os.path.join(extract_dir, "collection.tsv")

    collection = {}
    with open(collection_path, encoding="utf-8") as f:
        for row in tqdm(csv.reader(f, delimiter="\t"), desc="Loading collection"):
            collection[row[0]] = row[1]
    print(f"Collection: {len(collection):,}")

    # Init reranker
    reranker = CustomSpladeReranker(
        model_path=args.model,
        batch_size=args.batch_size,
        max_length=MAX_LENGTH,
        device=device,
        collection=collection,
    )

    # Streaming re-ranking
    splade_run = {}
    query_details = []
    processed = 0
    query_ids = list(queries.keys())

    print(f"\nRe-ranking {len(query_ids)} queries...")
    pbar = tqdm(query_ids, desc="Re-ranking")

    for idx, qid in enumerate(pbar):
        relevant_docs = {did: rel for did, rel in qrels.get(qid, {}).items() if rel > 0}
        if not relevant_docs:
            continue

        splade_run[qid] = reranker.rerank(queries[qid], list(bm25_run[qid].keys()))
        processed += 1

        best_did = max(splade_run[qid], key=splade_run[qid].get)
        best_score = splade_run[qid][best_did]
        best_text = collection.get(best_did, "N/A")
        correct_dids = list(relevant_docs.keys())
        correct_text = collection.get(correct_dids[0], "N/A")
        rank = find_relevant_rank(splade_run[qid], relevant_docs)
        rr = 1.0 / rank if rank else 0.0

        query_details.append({
            "query_idx": processed,
            "qid": qid,
            "query": queries[qid],
            "model_answer": {"doc_id": best_did, "score": round(best_score, 4), "text": best_text[:300]},
            "correct_answer": {"doc_ids": correct_dids, "text": correct_text[:300]},
            "relevant_rank": rank,
            "reciprocal_rank": round(rr, 6),
            "is_top1_hit": best_did in relevant_docs,
        })

        if processed % LOG_EVERY == 0 or (idx + 1) == len(query_ids):
            rm = evaluate_run(splade_run, qrels)
            mrr = rm.get("recip_rank", 0)
            r10 = rm.get("recall_10", 0)
            r100 = rm.get("recall_100", 0)
            r1000 = rm.get("recall_1000", 0)
            pbar.set_postfix({"MRR@10": f"{mrr:.4f}", "R@10": f"{r10:.4f}", "cache": reranker.cache_size})

            tqdm.write("")
            tqdm.write(f"  ── Query #{processed} (qid={qid}) ──")
            tqdm.write(f"  Question : {queries[qid]}")
            tqdm.write(f"  Model #1 : [did={best_did}, score={best_score:.2f}] {best_text[:150]}...")
            tqdm.write(
                f"  Correct  : [did={correct_dids[0]}"
                + (f" +{len(correct_dids)-1} more" if len(correct_dids) > 1 else "")
                + f"] {correct_text[:150]}..."
            )
            tqdm.write(f"  Rank: {rank}  (1/rank={rr:.4f})  Hit: {'✓' if best_did in relevant_docs else '✗'}")
            tqdm.write(f"  MRR@10={mrr:.4f}  R@10={r10:.4f}  R@100={r100:.4f}  R@1000={r1000:.4f}")

    print(f"\nDone. {processed} queries. Cache: {reranker.cache_size:,}.")

    # Final metrics
    splade_metrics = evaluate_run(splade_run, qrels)

    # MRR@10 from per-query details
    mrr10_vals = [1.0 / qd["relevant_rank"] if qd["relevant_rank"] and qd["relevant_rank"] <= 10 else 0.0
                  for qd in query_details]
    mrr_at_10 = sum(mrr10_vals) / max(1, len(mrr10_vals))
    splade_metrics["mrr@10"] = mrr_at_10

    print("\n=== Custom SPLADE v1 (final) ===")
    for m, v in splade_metrics.items():
        print(f"  {m:20s} = {v:.4f}")

    print(f"\n{'Metric':<22} {'BM25':>10} {'SPLADE':>10} {'Delta':>10}")
    print("-" * 55)
    for m in bm25_metrics:
        bv, sv = bm25_metrics[m], splade_metrics.get(m, 0)
        print(f"  {m:<20} {bv:>10.4f} {sv:>10.4f} {sv - bv:>+10.4f}")
    print(f"  {'mrr@10':<20} {'N/A':>10} {mrr_at_10:>10.4f}")

    # Write output
    output = {
        "bm25": bm25_metrics,
        "splade_v1": splade_metrics,
        "delta": {m: round(splade_metrics.get(m, 0) - bm25_metrics[m], 6) for m in bm25_metrics},
        "config": {
            "model_name": "custom_splade_v1 (distilbert-base-uncased)",
            "model_file": os.path.basename(args.model),
            "max_queries": max_queries,
            "batch_size": args.batch_size,
            "top_k_bm25": TOP_K_BM25,
            "queries_processed": processed,
            "doc_cache_size": reranker.cache_size,
        },
        "queries": query_details,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics written to {args.output} ({len(query_details)} queries)")


if __name__ == "__main__":
    main()
