"""Pinecone vector store for embeddings."""

from __future__ import annotations

import os
from typing import Any


def _flatten_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten nested metadata for Pinecone storage."""
    if not metadata:
        return {}

    flattened: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            flattened[key] = value
        else:
            flattened[key] = str(value)
    return flattened


def _get_pinecone_client() -> Any:
    """Initialize Pinecone client from environment."""
    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise RuntimeError(
            "The `pinecone-client` package is required. Install it with: pip install pinecone-client"
        ) from exc

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY environment variable not set. "
            "Get your key from https://app.pinecone.io/"
        )

    return Pinecone(api_key=api_key)


def upsert_vectors(
    vectors: list[dict[str, Any]],
    index_name: str = "review-chunks",
    namespace: str = "reviews",
) -> None:
    """Persist chunk embeddings in Pinecone.
    
    Args:
        vectors: List of chunk records with 'embedding', 'chunk_id', and 'text'
        index_name: Pinecone index name (must already exist)
        namespace: Optional namespace for organization
    """

    if not vectors:
        return

    client = _get_pinecone_client()
    index = client.Index(index_name)

    # Prepare vectors for upsert
    vectors_to_upsert = []
    for vector in vectors:
        chunk_id = str(vector.get("chunk_id") or vector.get("review_id"))
        embedding = vector.get("embedding")
        
        if embedding is None:
            continue

        # Prepare metadata
        metadata = {
            "review_id": str(vector.get("review_id", "")),
            "chunk_index": vector.get("chunk_index", 0),
            "embedding_model": vector.get("embedding_model", ""),
            "text": str(vector.get("text", ""))[:1000],  # Limit text length
            "embedding_dimension": vector.get("embedding_dimension", 0),
            **_flatten_metadata(vector.get("metadata")),
        }
        
        # Pinecone upsert format: (id, values, metadata)
        vectors_to_upsert.append(
            (chunk_id, embedding, {k: v for k, v in metadata.items() if v is not None})
        )

    # Batch upsert (Pinecone recommends batches of 100-1000)
    batch_size = 500
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)

    print(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone index '{index_name}' namespace '{namespace}'")
