"""Vector store abstraction for embedded chunks."""

from __future__ import annotations

from typing import Any


def upsert_vectors(
    vectors: list[dict[str, Any]],
    collection_name: str = "review_chunks",
    use_pinecone: bool = True,
) -> None:
    """Persist chunk embeddings to vector database.
    
    Attempts to use Pinecone if available and use_pinecone=True,
    falls back to Chroma for local development.
    
    Args:
        vectors: List of embedded chunk records
        collection_name: Collection/index name
        use_pinecone: Prefer Pinecone if configured
    """

    if not vectors:
        return

    # Try Pinecone first if requested
    if use_pinecone:
        try:
            from ai_product_insights_dashboard.storage.pinecone_store import upsert_vectors as upsert_pinecone
            upsert_pinecone(vectors, index_name=collection_name)
            return
        except (ImportError, ValueError, Exception) as e:
            print(f"Pinecone not available ({e}), falling back to Chroma...")

    # Fall back to Chroma
    _upsert_chroma(vectors, collection_name)


def _upsert_chroma(
    vectors: list[dict[str, Any]],
    collection_name: str = "review_chunks",
) -> None:
    """Persist chunk embeddings and searchable metadata in Chroma.

    The collection is stored persistently under `data/vector_db/chroma`.
    """

    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The `chromadb` package is required to persist vectors locally."
        ) from exc

    if not vectors:
        return

    client = chromadb.PersistentClient(path="data/vector_db/chroma")
    collection = client.get_or_create_collection(name=collection_name)

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for vector in vectors:
        chunk_id = str(vector.get("chunk_id") or vector.get("review_id"))
        embedding = vector.get("embedding")
        if embedding is None:
            continue
        if chunk_id in seen_ids:
            # The source dataset occasionally contains an exact duplicate
            # review; chunk_id is a content hash, so the duplicate collapses
            # onto the same id. Chroma's upsert requires unique ids per call.
            continue
        seen_ids.add(chunk_id)

        ids.append(chunk_id)
        embeddings.append([float(value) for value in embedding])
        documents.append(str(vector.get("text", "")))
        metadata = {
            "review_id": vector.get("review_id"),
            "chunk_id": chunk_id,
            "chunk_index": vector.get("chunk_index"),
            "embedding_model": vector.get("embedding_model"),
            "embedding_dimension": vector.get("embedding_dimension"),
            **_flatten_metadata(vector.get("metadata")),
        }
        metadatas.append({key: value for key, value in metadata.items() if value is not None})

    if not ids:
        return

    # Chroma rejects a single upsert call above ~5,461 records (its internal
    # max batch size), which the original 800-product/1k-review corpus never
    # hit but a single-product 20k-review corpus does.
    batch_size = 4000
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


def _flatten_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}

    flattened: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            flattened[key] = value
        else:
            flattened[key] = str(value)
    return flattened
