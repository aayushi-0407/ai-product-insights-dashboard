"""Create vector embeddings for chunked reviews using OpenAI or local models."""

from __future__ import annotations

import os
from typing import Any, Iterable


DEFAULT_EMBEDDING_MODEL = "local"  # Default to free local embeddings
DEFAULT_EMBEDDING_DIMENSION = 384  # sentence-transformers dimension


def _get_openai_client() -> Any:
    """Initialize OpenAI client from environment."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The `openai` package is required for OpenAI embeddings. Install it with: pip install openai"
        ) from exc
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Set it with: export OPENAI_API_KEY=sk-proj-..."
        )
    return OpenAI(api_key=api_key)


def _get_sentence_transformer() -> Any:
    """Initialize sentence-transformers for local embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "The `sentence-transformers` package is required. Install it with: pip install sentence-transformers torch"
        ) from exc
    
    print("  Loading local embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def embed_chunks(
    chunks: Iterable[dict[str, Any]],
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSION,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Attach embeddings to chunk records.
    
    Supports:
    - "local" (default): Free sentence-transformers (all-MiniLM-L6-v2, 384-dim)
    - "openai": OpenAI text-embedding-3-small (requires OPENAI_API_KEY, 1536-dim)
    
    Args:
        chunks: Iterable of chunk dictionaries
        model: Embedding model ("local" or "openai")
        dimensions: Target dimensions (auto-detected if using local)
        batch_size: Batch size for processing (32 for local, 100 for OpenAI)
    """
    
    chunk_list = list(chunks)
    if not chunk_list:
        return []
    
    # Use local embeddings by default (free, no API key needed)
    if model == "local" or model == "all-MiniLM-L6-v2":
        return _embed_with_sentence_transformers(chunk_list, batch_size)
    elif model == "openai" or model.startswith("text-embedding"):
        return _embed_with_openai(chunk_list, model, dimensions, batch_size)
    else:
        raise ValueError(f"Unknown embedding model: {model}. Use 'local' or 'openai'.")


def _embed_with_sentence_transformers(
    chunks: list[dict[str, Any]],
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Embed chunks using sentence-transformers (free, local)."""
    model = _get_sentence_transformer()
    embedded_chunks: list[dict[str, Any]] = []
    
    # Process in batches
    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        batch = chunks[batch_start:batch_end]
        
        texts = [chunk["text"] for chunk in batch]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        for chunk, embedding in zip(batch, embeddings):
            chunk_record = dict(chunk)
            chunk_record["embedding"] = embedding.tolist()
            chunk_record["embedding_model"] = "all-MiniLM-L6-v2"
            chunk_record["embedding_dimension"] = len(embedding)
            embedded_chunks.append(chunk_record)
    
    return embedded_chunks


def _embed_with_openai(
    chunks: list[dict[str, Any]],
    model: str,
    dimensions: int,
    batch_size: int = 100,
) -> list[dict[str, Any]]:
    """Embed chunks using OpenAI API."""
    client = _get_openai_client()
    embedded_chunks: list[dict[str, Any]] = []
    
    # Process in batches
    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        batch = chunks[batch_start:batch_end]
        
        texts = [chunk["text"] for chunk in batch]
        
        # Call OpenAI API
        response = client.embeddings.create(
            model=model,
            input=texts,
            dimensions=dimensions,
        )
        
        # Attach embeddings
        for i, chunk in enumerate(batch):
            chunk_record = dict(chunk)
            chunk_record["embedding"] = response.data[i].embedding
            chunk_record["embedding_model"] = model
            chunk_record["embedding_dimension"] = dimensions
            embedded_chunks.append(chunk_record)
    
    return embedded_chunks
