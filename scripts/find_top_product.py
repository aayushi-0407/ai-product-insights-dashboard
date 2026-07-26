#!/usr/bin/env python3
"""Find the single highest-review-volume product in raw_review_Software.

We want one product with 10k+ reviews so clustering produces sharp,
product-specific themes instead of vague cross-product ones (a corpus
spanning hundreds of unrelated products clusters more by writing style
than by actual recurring issues, since there's no shared problem space).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import truststore

truststore.inject_into_ssl()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from huggingface_hub import hf_hub_url
    import requests

    url = hf_hub_url(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        repo_type="dataset",
        filename="raw/review_categories/Software.jsonl",
    )
    print("Streaming Software.jsonl (~1.87GB) to count reviews per product...")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    counts: Counter[str] = Counter()
    scanned = 0
    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            scanned += 1
            if scanned % 500_000 == 0:
                print(f"  scanned {scanned:,} reviews...")
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = record.get("parent_asin")
            if asin:
                counts[asin] += 1
    finally:
        response.close()

    print(f"\n✓ Scanned {scanned:,} reviews across {len(counts):,} distinct products")

    top20 = counts.most_common(20)
    output_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "top_products.json"
    output_path.write_text(json.dumps(top20, indent=2), encoding="utf-8")
    print(f"✓ Saved top 20 -> {output_path}")

    print("\nTop 20 products by review count:")
    for asin, count in top20:
        print(f"  {asin}: {count:,} reviews")


if __name__ == "__main__":
    main()
