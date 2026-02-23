#!/usr/bin/env python3
"""SPLADE Re-Ranking Pipeline on MS MARCO Dev — standalone script for DataSphere Jobs.

Usage:
    python main.py --collection /path/to/collection.tar.gz --max-queries 100 --output metrics.json

The script:
    1. Clones the SPLADE repo (if not already present) for model inference code
    2. Loads the SPLADE model from HuggingFace
    3. Loads MS MARCO dev queries + qrels via ir_datasets
    4. Loads BM25 top-1000 candidates via ir_datasets
    5. Loads the full passage collection from the provided TSV
    6. Streams query-by-query re-ranking with doc-level caching
    7. Writes final metrics + per-query details to a JSON output file
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


# ─────────────────────────── Constants ───────────────────────────

MODEL_NAME = "naver/splade-cocondenser-ensembledistil"
IR_DATASET_NAME = "msmarco-passage/dev/small"
BATCH_SIZE_DEFAULT = 32
MAX_LENGTH = 256
TOP_K_BM25 = 1000
LOG_EVERY = 100
SPLADE_REPO_URL = "https://github.com/naver/splade.git"
SPLADE_DIR = "splade"


# ─────────────────────────── Setup ───────────────────────────────

def clone_splade():
    """Clone the SPLADE repo if not already present, and add it to sys.path."""
    if not os.path.exists(SPLADE_DIR):
        print(f"Cloning SPLADE from {SPLADE_REPO_URL}...")
        subprocess.run(["git", "clone", SPLADE_REPO_URL, SPLADE_DIR], check=True)
    else:
        print(f"SPLADE repo already exists at ./{SPLADE_DIR}")
    if SPLADE_DIR not in sys.path:
        sys.path.insert(0, SPLADE_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SPLADE re-ranking on MS MARCO dev with BM25 top-1000"
    )
    parser.add_argument(
        "--collection", required=True,
        help="Path to collection.tar.gz (archive containing collection.tsv)"
    )
    parser.add_argument(
        "--max-queries", type=int, required=True,
        help="Number of dev queries to process (use 0 for all ~6980)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE_DEFAULT,
        help=f"Batch size for SPLADE encoding (default: {BATCH_SIZE_DEFAULT})"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for output metrics JSON file"
    )
    return parser.parse_args()


# ─────────────────────────── Evaluation ──────────────────────────

def evaluate_run(run, qrels, metrics=("recall_10", "recall_100", "recall_1000", "recip_rank")):
    """Evaluate a run dict against qrels using pytrec_eval."""
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
    """Find the rank (1-based) of the first relevant document.

    Args:
        scored_docs: {doc_id: score}
        relevant_doc_ids: set/dict of relevant doc_ids

    Returns:
        rank (1-based) of first relevant doc, or None if not found in the pool.
    """
    sorted_dids = sorted(scored_docs, key=scored_docs.get, reverse=True)
    for rank, did in enumerate(sorted_dids, 1):
        if did in relevant_doc_ids:
            return rank
    return None


# ─────────────────────────── SPLADE Encoding ─────────────────────

class SpladeReranker:
    """Wraps SPLADE model with doc-level caching for streaming re-ranking.

    Memory optimization: SPLADE vectors are very sparse (~100-200 non-zero values
    out of 30522). Dense float32 = 122 KB/doc → 122 GB for 1M docs.
    Sparse COO storage: ~1.8 KB/doc → ~1.8 GB for 1M docs (~68x reduction).
    """

    def __init__(self, model_name, batch_size, max_length, device, collection):
        from splade.models.transformer_rep import Splade  # noqa: from cloned repo

        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.collection = collection

        print(f"Loading SPLADE model: {model_name}")
        self.model = Splade(model_name, agg="max").to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.doc_rep_cache = {}  # {doc_id: sparse_coo_tensor}

    @staticmethod
    def _to_sparse(dense_vec):
        """Convert a 1-D dense tensor to sparse COO for compact storage."""
        return dense_vec.to_sparse_coo().coalesce()

    @staticmethod
    def _sparse_to_dense(sparse_vec):
        """Convert sparse COO back to dense for computation."""
        return sparse_vec.to_dense()

    def encode_query(self, query_text):
        """Encode a single query. Returns 1-D dense CPU tensor (vocab_size,)."""
        tokens = self.tokenizer(
            [query_text], return_tensors="pt",
            truncation=True, max_length=self.max_length, padding=True,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            q_rep = self.model(q_kwargs=tokens)["q_rep"].squeeze(0).cpu()
        return q_rep

    def encode_docs_cached(self, doc_ids):
        """Encode docs using cache (stored SPARSE). Returns dense (n, vocab_size)."""
        uncached_ids = [did for did in doc_ids if did not in self.doc_rep_cache]
        if uncached_ids:
            for i in range(0, len(uncached_ids), self.batch_size):
                batch_ids = uncached_ids[i: i + self.batch_size]
                batch_texts = [self.collection[did] for did in batch_ids]
                tokens = self.tokenizer(
                    batch_texts, return_tensors="pt",
                    truncation=True, max_length=self.max_length, padding=True,
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                with torch.no_grad():
                    reps = self.model(d_kwargs=tokens)["d_rep"]
                for j, did in enumerate(batch_ids):
                    self.doc_rep_cache[did] = self._to_sparse(reps[j].cpu())
        # Densify only the docs needed for this query (~1000 × 122KB = 122MB temporary)
        return torch.stack([self._sparse_to_dense(self.doc_rep_cache[did]) for did in doc_ids])

    def rerank(self, query_text, candidate_doc_ids):
        """Re-rank candidates for a single query. Returns {doc_id: score}."""
        q_vec = self.encode_query(query_text)
        d_matrix = self.encode_docs_cached(candidate_doc_ids)
        scores = torch.matmul(d_matrix, q_vec).tolist()
        del d_matrix  # free the temporary dense matrix immediately
        return {did: score for did, score in zip(candidate_doc_ids, scores)}

    @property
    def cache_size(self):
        return len(self.doc_rep_cache)


# ─────────────────────────── Main Pipeline ───────────────────────

def main():
    args = parse_args()
    max_queries = args.max_queries if args.max_queries > 0 else None

    # Step 0: Clone SPLADE source code
    clone_splade()

    # Step 1: Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 2: Load queries + qrels from ir_datasets
    import ir_datasets
    print(f"Loading dataset: {IR_DATASET_NAME}")
    dataset = ir_datasets.load(IR_DATASET_NAME)

    queries = {}
    for q in dataset.queries_iter():
        queries[q.query_id] = q.text
    print(f"Total queries: {len(queries)}")

    qrels = {}
    for qrel in dataset.qrels_iter():
        qid, did, rel = qrel.query_id, qrel.doc_id, qrel.relevance
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][did] = rel
    print(f"Queries with relevance judgments: {len(qrels)}")

    # Step 3: BM25 top-1000
    print("Loading BM25 top-1000 scored docs...")
    bm25_run = defaultdict(dict)
    for sd in tqdm(dataset.scoreddocs_iter(), desc="BM25 scoreddocs"):
        qid = sd.query_id
        if max_queries and len(bm25_run) >= max_queries and qid not in bm25_run:
            continue
        if len(bm25_run[qid]) < TOP_K_BM25:
            bm25_run[qid][sd.doc_id] = sd.score
    bm25_run = dict(bm25_run)
    queries = {qid: text for qid, text in queries.items() if qid in bm25_run}
    print(f"Queries in BM25 run: {len(bm25_run)}, to process: {len(queries)}")

    # Step 4: BM25 baseline evaluation
    bm25_run_eval = {qid: {did: float(s) for did, s in docs.items()} for qid, docs in bm25_run.items()}
    bm25_metrics = evaluate_run(bm25_run_eval, qrels)
    print("\n=== BM25 Baseline ===")
    for metric, value in bm25_metrics.items():
        print(f"  {metric:20s} = {value:.4f}")

    # Step 5: Extract archive and load full collection
    print(f"\nExtracting collection from {args.collection}...")
    extract_dir = tempfile.mkdtemp(prefix="collection_")
    with tarfile.open(args.collection, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    collection_tsv_path = os.path.join(extract_dir, "collection.tsv")
    print(f"Extracted to {collection_tsv_path}")

    print("Loading collection...")
    collection = {}
    with open(collection_tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader, desc="Loading collection"):
            collection[row[0]] = row[1]
    print(f"Collection size: {len(collection):,}")

    # Step 6: Initialize SPLADE reranker
    reranker = SpladeReranker(
        model_name=MODEL_NAME,
        batch_size=args.batch_size,
        max_length=MAX_LENGTH,
        device=device,
        collection=collection,
    )

    # Step 7: Streaming re-ranking with per-query detail collection
    splade_run = {}
    query_details = []
    processed = 0
    query_ids = list(queries.keys())

    print(f"\nStarting re-ranking of {len(query_ids)} queries...")
    pbar = tqdm(query_ids, desc="Re-ranking")

    for idx, qid in enumerate(pbar):
        # Skip queries with no positive relevance judgments
        relevant_docs = {did: rel for did, rel in qrels.get(qid, {}).items() if rel > 0}
        if not relevant_docs:
            continue

        candidate_dids = list(bm25_run[qid].keys())
        splade_run[qid] = reranker.rerank(queries[qid], candidate_dids)
        processed += 1

        # Model's best answer
        best_did = max(splade_run[qid], key=splade_run[qid].get)
        best_score = splade_run[qid][best_did]
        best_text = collection.get(best_did, "N/A")

        # Correct answer(s)
        correct_dids = list(relevant_docs.keys())
        correct_text = collection.get(correct_dids[0], "N/A")

        # Rank of first relevant doc (denominator in MRR)
        rank = find_relevant_rank(splade_run[qid], relevant_docs)
        reciprocal_rank = 1.0 / rank if rank else 0.0

        # Record per-query detail
        query_details.append({
            "query_idx": processed,
            "qid": qid,
            "query": queries[qid],
            "model_answer": {
                "doc_id": best_did,
                "score": round(best_score, 4),
                "text": best_text[:300],
            },
            "correct_answer": {
                "doc_ids": correct_dids,
                "text": correct_text[:300],
            },
            "relevant_rank": rank,
            "reciprocal_rank": round(reciprocal_rank, 6),
            "is_top1_hit": best_did in relevant_docs,
        })

        # Log every N queries
        if processed % LOG_EVERY == 0 or (idx + 1) == len(query_ids):
            running_metrics = evaluate_run(splade_run, qrels)
            mrr = running_metrics.get("recip_rank", 0)
            r10 = running_metrics.get("recall_10", 0)
            r100 = running_metrics.get("recall_100", 0)
            r1000 = running_metrics.get("recall_1000", 0)

            pbar.set_postfix({
                "MRR@10": f"{mrr:.4f}", "R@10": f"{r10:.4f}",
                "R@1000": f"{r1000:.4f}", "cache": reranker.cache_size,
            })

            tqdm.write("")
            tqdm.write(f"  ── Query #{processed} (idx={idx+1}/{len(query_ids)}, qid={qid}) ──")
            tqdm.write(f"  Question : {queries[qid]}")
            tqdm.write(f"  Model #1 : [did={best_did}, score={best_score:.2f}] {best_text[:150]}...")
            tqdm.write(
                f"  Correct  : [did={correct_dids[0]}"
                + (f" +{len(correct_dids)-1} more" if len(correct_dids) > 1 else "")
                + f"] {correct_text[:150]}..."
            )
            tqdm.write(f"  Rank of correct: {rank}  (1/rank = {reciprocal_rank:.4f})")
            tqdm.write(f"  Top-1 hit: {'✓ YES' if best_did in relevant_docs else '✗ NO'}")
            tqdm.write(
                f"  Metrics  : MRR@10={mrr:.4f}  R@10={r10:.4f}  "
                f"R@100={r100:.4f}  R@1000={r1000:.4f}  "
                f"doc_cache={reranker.cache_size:,}"
            )

    print(f"\nDone. Processed {processed} queries. Cache: {reranker.cache_size:,}.")

    # Step 8: Final evaluation
    splade_metrics = evaluate_run(splade_run, qrels)

    # Compute MRR@10 from per-query details: 1/rank if rank <= 10, else 0
    mrr_at_10_values = []
    for qd in query_details:
        rank = qd["relevant_rank"]
        mrr_at_10_values.append(1.0 / rank if rank is not None and rank <= 10 else 0.0)
    mrr_at_10 = sum(mrr_at_10_values) / max(1, len(mrr_at_10_values))
    splade_metrics["mrr@10"] = mrr_at_10

    print("\n=== SPLADE Re-Ranking (final) ===")
    for metric, value in splade_metrics.items():
        print(f"  {metric:20s} = {value:.4f}")

    print(f"\n{'Metric':<22} {'BM25':>10} {'SPLADE':>10} {'Delta':>10}")
    print("-" * 55)
    for metric in bm25_metrics:
        bv = bm25_metrics[metric]
        sv = splade_metrics.get(metric, 0)
        print(f"  {metric:<20} {bv:>10.4f} {sv:>10.4f} {sv - bv:>+10.4f}")
    print(f"  {'mrr@10':<20} {'N/A':>10} {mrr_at_10:>10.4f} {'':>10}")

    # Step 9: Write metrics JSON with per-query details
    output = {
        "bm25": bm25_metrics,
        "splade": splade_metrics,
        "delta": {m: round(splade_metrics[m] - bm25_metrics[m], 6) for m in bm25_metrics},
        "config": {
            "model_name": MODEL_NAME,
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
    print(f"\nMetrics + per-query details written to {args.output} ({len(query_details)} queries)")


if __name__ == "__main__":
    main()
