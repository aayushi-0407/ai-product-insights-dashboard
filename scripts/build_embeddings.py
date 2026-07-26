"""Entry point for chunking and embedding reviews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_product_insights_dashboard.processing.chunk_reviews import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_reviews,
)
from ai_product_insights_dashboard.processing.embed_reviews import embed_chunks


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk reviews and build embeddings")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "raw" / "reviews.jsonl"),
        help="JSONL file containing raw review records",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "review_chunks.jsonl"),
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--dimensions", type=int, default=128)
    args = parser.parse_args()

    raw_reviews = _read_jsonl(Path(args.input))
    chunks = chunk_reviews(raw_reviews, chunk_size=args.chunk_size, overlap=args.overlap)
    embedded_chunks = embed_chunks(chunks, dimensions=args.dimensions)
    _write_jsonl(Path(args.output), embedded_chunks)
    print(f"Built {len(embedded_chunks)} embedded chunks -> {args.output}")


if __name__ == "__main__":
    main()
