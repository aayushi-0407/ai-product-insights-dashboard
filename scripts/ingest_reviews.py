"""Entry point for loading the review dataset from Hugging Face."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# See ai_product_insights_dashboard.app for why this runs before other imports.
import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_product_insights_dashboard.ingestion.fetch_reviews import fetch_reviews
from ai_product_insights_dashboard.processing.chunk_reviews import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_reviews,
)
from ai_product_insights_dashboard.processing.embed_reviews import embed_chunks
from ai_product_insights_dashboard.storage.document_store import save_documents
from ai_product_insights_dashboard.storage.vector_store import upsert_vectors
from ai_product_insights_dashboard.storage.postgres_store import save_documents as save_to_postgres
from ai_product_insights_dashboard.storage.postgres_store import init_schema


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    # Windows terminals may default to cp1252, which cannot render the status
    # symbols used by this CLI.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Fetch Amazon reviews from Hugging Face, chunk, embed, and store"
    )
    parser.add_argument(
        "--dataset",
        default="McAuley-Lab/Amazon-Reviews-2023",
        help="Hugging Face dataset ID"
    )
    parser.add_argument(
        "--config-name",
        default="raw_review_Software",
        help="Dataset configuration (e.g., raw_review_Software)"
    )
    parser.add_argument("--split", default=None, help="Dataset split (not used for this dataset)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of reviews to fetch (for testing, or the corpus size when --parent-asin is set)"
    )
    parser.add_argument(
        "--parent-asin",
        default=None,
        help="Fetch only this product's reviews (see scripts/find_top_product.py) — "
             "a single-product corpus produces sharp, specific themes instead of "
             "vague cross-product ones",
    )
    parser.add_argument(
        "--raw-output",
        default=str(PROJECT_ROOT / "data" / "raw" / "reviews.jsonl"),
        help="Output path for raw reviews"
    )
    parser.add_argument(
        "--chunks-output",
        default=str(PROJECT_ROOT / "data" / "processed" / "review_chunks.jsonl"),
        help="Output path for chunked reviews"
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize PostgreSQL schema"
    )
    parser.add_argument(
        "--skip-vector-db",
        action="store_true",
        help="Skip uploading to vector DB (useful for testing)"
    )
    parser.add_argument(
        "--chunk-only",
        action="store_true",
        help="Fetch and chunk the Amazon dataset without embedding or database writes",
    )
    parser.add_argument(
        "--embedding-model",
        default="local",
        help="Embedding model: 'local' (free, all-MiniLM-L6-v2) or 'openai' (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=384,
        help="Embedding dimension: 384 for local, 1536 for OpenAI text-embedding-3-small"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Token chunk size (default: 300; preserves most review boundaries)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Token overlap between chunks"
    )

    args = parser.parse_args()

    # Initialize database if requested
    if args.init_db:
        print("Initializing PostgreSQL schema...")
        try:
            init_schema()
            print("✓ PostgreSQL schema initialized")
        except Exception as e:
            print(f"✗ Failed to initialize schema: {e}")
            print("  Continuing without PostgreSQL...")

    # Fetch reviews
    print(f"Fetching reviews from {args.dataset}:{args.config_name}")
    print(f"  Limit: {args.limit or 'none'}")
    if args.parent_asin:
        print(f"  Filtering to single product: {args.parent_asin} (full file scan required)")
    reviews = fetch_reviews(
        args.dataset,
        config_name=args.config_name,
        split=args.split,
        limit=args.limit,
        parent_asin=args.parent_asin,
    )
    print(f"✓ Fetched {len(reviews)} reviews")

    # Save raw reviews
    raw_path = Path(args.raw_output)
    _write_jsonl(raw_path, reviews)
    print(f"✓ Saved raw reviews -> {raw_path}")

    # Chunk reviews
    print("Chunking reviews...")
    chunks = chunk_reviews(reviews, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    print(f"✓ Created {len(chunks)} chunks")

    if args.chunk_only:
        chunks_path = save_documents(chunks, Path(args.chunks_output))
        print(f"✓ Saved chunks -> {chunks_path}")
        print("\n✓ Amazon dataset chunking complete!")
        return

    # Embed chunks with OpenAI
    print(f"Embedding chunks with {args.embedding_model}...")
    try:
        embedded_chunks = embed_chunks(
            chunks,
            model=args.embedding_model,
            dimensions=args.embedding_dimension,
        )
        print(f"✓ Embedded {len(embedded_chunks)} chunks")
    except Exception as e:
        print(f"✗ Embedding failed: {e}")
        print("  Make sure OPENAI_API_KEY environment variable is set")
        sys.exit(1)

    # Save embedded chunks to file
    chunks_path = save_documents(embedded_chunks, Path(args.chunks_output))
    print(f"✓ Saved embedded chunks -> {chunks_path}")

    # Save to PostgreSQL
    print("Saving to PostgreSQL...")
    try:
        num_saved = save_to_postgres(embedded_chunks)
        print(f"✓ Saved {num_saved} documents to PostgreSQL")
    except Exception as e:
        print(f"✗ PostgreSQL save failed: {e}")
        print("  Make sure DATABASE_URL environment variable is set")

    # Upsert to vector database
    if not args.skip_vector_db:
        print("Upserting to vector database...")
        try:
            upsert_vectors(embedded_chunks)
            print(f"✓ Upserted {len(embedded_chunks)} vectors")
        except Exception as e:
            print(f"✗ Vector DB upsert failed: {e}")
            print("  Check your Pinecone configuration or fall back to local Chroma")
    else:
        print("⊘ Skipped vector database upload")

    print("\n✓ Ingestion pipeline complete!")
    print(f"  Reviews: {raw_path}")
    print(f"  Chunks: {chunks_path}")


if __name__ == "__main__":
    main()
