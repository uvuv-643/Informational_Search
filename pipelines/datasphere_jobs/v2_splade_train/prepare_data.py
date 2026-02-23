#!/usr/bin/env python3
"""Pre-package MS MARCO triplets for fast DataSphere upload.

Reads from the local ir_datasets cache (~/.ir_datasets/msmarco-passage/),
resolves triplet IDs to text, and saves as compressed JSONL for fast loading.

This avoids the 20-30 minute ir_datasets download on DataSphere.

Usage:
    python prepare_data.py                         # default: 500K triplets
    python prepare_data.py --max-samples 100000    # custom count
    python prepare_data.py --output my_data.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Pre-package MS MARCO triplets")
    parser.add_argument("--output", default="triplets.jsonl.gz", help="Output file path")
    parser.add_argument("--max-samples", type=int, default=0, help="Max triplets to export (0 = all ~39.8M)")
    args = parser.parse_args()

    try:
        import ir_datasets
    except ImportError:
        print("ERROR: ir_datasets not installed. Run: pip install ir_datasets")
        sys.exit(1)

    print("Loading MS MARCO passage triples...")
    t0 = time.time()

    dataset = ir_datasets.load("msmarco-passage/train/triples-small")

    print("Loading queries...")
    queries = {q.query_id: q.text for q in tqdm(dataset.queries_iter(), desc="Queries")}
    print(f"  -> {len(queries):,} queries")

    print("Loading documents...")
    docs = {d.doc_id: d.text for d in tqdm(dataset.docs_iter(), desc="Docs")}
    print(f"  -> {len(docs):,} documents")

    use_all = (args.max_samples <= 0)
    total_label = "ALL (~39.8M)" if use_all else f"{args.max_samples:,}"
    print(f"Exporting {total_label} triplets to {args.output}...")
    count = 0
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=6) as f:
        for item in tqdm(dataset.docpairs_iter(), desc="Triplets", total=None if use_all else args.max_samples):
            if not use_all and count >= args.max_samples:
                break
            line = json.dumps({
                "q": queries[item.query_id],
                "p": docs[item.doc_id_a],
                "n": docs[item.doc_id_b],
            }, ensure_ascii=False)
            f.write(line + "\n")
            count += 1

    elapsed = time.time() - t0
    file_size = os.path.getsize(args.output)
    print(f"\nDone! {count:,} triplets exported in {elapsed:.0f}s")
    print(f"File: {args.output} ({file_size / 1e6:.1f} MB)")
    print(f"\nTo use with training:")
    print(f"  python main.py --data {args.output} --output model.pt --log training_log.jsonl")


if __name__ == "__main__":
    main()
