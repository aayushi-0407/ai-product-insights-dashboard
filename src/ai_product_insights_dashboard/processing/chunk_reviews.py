"""Chunk review text into retrievable units."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable


# The PRD treats a review as one retrievable unit unless it is unusually long.
# Amazon Software reviews are generally short, so this preserves the review
# boundary and its metadata for the normal case.
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 30


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _chunk_tokens(tokens: list[str], chunk_size: int, overlap: int) -> list[list[str]]:
    if not tokens:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[list[str]] = []
    start = 0
    step = chunk_size - overlap

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokens[start:end])
        if end == len(tokens):
            break
        start += step

    return chunks


def chunk_reviews(
    reviews: Iterable[Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Convert raw reviews into chunk records ready for embedding."""

    chunk_records: list[dict[str, Any]] = []

    for review_index, review in enumerate(reviews):
        review_dict = review if isinstance(review, dict) else dict(review)
        text = _normalize_text(str(review_dict.get("text") or review_dict.get("review_text") or ""))
        if not text:
            continue

        tokens = text.split(" ")
        token_groups = _chunk_tokens(tokens, chunk_size=chunk_size, overlap=overlap)
        review_id = review_dict.get("review_id") or review_dict.get("id")
        if review_id is None:
            # Amazon Reviews 2023 has no review-level ID. `asin` identifies a
            # product, not a review, so derive a stable ID from the review's
            # identifying fields instead of collapsing reviews per product.
            identity = "\x1f".join(
                str(review_dict.get(field, ""))
                for field in ("asin", "user_id", "timestamp", "title", "text")
            )
            review_id = f"amazon-{hashlib.sha1(identity.encode('utf-8')).hexdigest()}"
        review_id = str(review_id)

        for chunk_index, token_group in enumerate(token_groups):
            chunk_text = " ".join(token_group)
            chunk_records.append(
                {
                    "review_id": review_id,
                    "chunk_id": f"{review_id}::{chunk_index}",
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "token_count": len(token_group),
                    "metadata": {
                        key: value
                        for key, value in review_dict.items()
                        if key not in {"text", "review_text"}
                    },
                }
            )

    return chunk_records
