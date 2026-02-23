"""Utility helpers for inspecting SPLADE sparse representations."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def sparse_to_bow(
    rep: torch.Tensor,
    reverse_voc: Dict[int, str],
    top_k: int = 0,
) -> List[Tuple[str, float]]:
    """Convert a single sparse SPLADE vector into a sorted list of (token, weight) pairs.

    Args:
        rep: 1-D tensor of shape ``(vocab_size,)`` – one document/query representation.
        reverse_voc: mapping ``{token_id: token_string}`` (inverted tokenizer vocab).
        top_k: if > 0, return only the *top_k* highest-weight terms; 0 means return all non-zero.

    Returns:
        List of ``(token, weight)`` sorted descending by weight.
    """
    indices = torch.nonzero(rep).squeeze(-1).cpu().tolist()
    if not indices:
        return []
    weights = rep[indices].cpu().tolist()
    pairs = sorted(
        [(reverse_voc[idx], round(w, 4)) for idx, w in zip(indices, weights)],
        key=lambda x: x[1],
        reverse=True,
    )
    if top_k > 0:
        pairs = pairs[:top_k]
    return pairs


def batch_sparse_to_bow(
    batch_rep: torch.Tensor,
    reverse_voc: Dict[int, str],
    top_k: int = 0,
) -> List[List[Tuple[str, float]]]:
    """Apply :func:`sparse_to_bow` to every row in a batch.

    Args:
        batch_rep: 2-D tensor of shape ``(batch_size, vocab_size)``.
        reverse_voc: inverted tokenizer vocab.
        top_k: passed to :func:`sparse_to_bow`.

    Returns:
        List (one per document) of ``(token, weight)`` lists.
    """
    return [sparse_to_bow(batch_rep[i], reverse_voc, top_k) for i in range(batch_rep.size(0))]


def pretty_bow(
    bow: List[Tuple[str, float]],
    max_terms: int = 30,
    bar_width: int = 25,
) -> str:
    """Return a human-friendly multi-line string with mini bar-chart.

    Args:
        bow: output of :func:`sparse_to_bow`.
        max_terms: how many top terms to show (0 = all).
        bar_width: character width of the bar.

    Returns:
        Formatted string ready to ``print()``.
    """
    if not bow:
        return "(empty representation)"
    shown = bow[:max_terms] if max_terms > 0 else bow
    max_w = shown[0][1] if shown else 1.0
    lines: list[str] = []
    longest_token = max(len(t) for t, _ in shown)
    for token, weight in shown:
        filled = int(bar_width * weight / max_w) if max_w > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        lines.append(f"  {token:<{longest_token}}  {bar}  {weight:.4f}")
    header = f"Top-{len(shown)} terms (of {len(bow)} non-zero dims):"
    return "\n".join([header] + lines)
