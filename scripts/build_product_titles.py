#!/usr/bin/env python3
"""One-time enrichment: join real product titles onto the review corpus.

raw_review_Software has no product-name field, only `asin`/`parent_asin`
(PRD §4). raw_meta_Software has `title` (+ `store`) keyed by `parent_asin`.
This streams the ~256MB meta file once, keeps only the products actually
present in our review corpus, and writes a small lookup table — the meta
file itself is not committed or baked into the deployed image, only this
lookup is.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import truststore

truststore.inject_into_ssl()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    # Windows terminals default to cp1252, which can't render arbitrary
    # product titles (or the status symbols below).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from huggingface_hub import hf_hub_url
    import requests

    reviews_path = PROJECT_ROOT / "data" / "raw" / "reviews.jsonl"
    needed_asins = set()
    with reviews_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            asin = record.get("parent_asin")
            if asin:
                needed_asins.add(asin)

    print(f"Looking up product titles for {len(needed_asins)} distinct products...")

    url = hf_hub_url(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        repo_type="dataset",
        filename="raw/meta_categories/meta_Software.jsonl",
    )
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    titles: dict[str, dict[str, str]] = {}
    scanned = 0
    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            scanned += 1
            if scanned % 200_000 == 0:
                print(f"  scanned {scanned:,} products, matched {len(titles)}/{len(needed_asins)}...")
            record = json.loads(line)
            asin = record.get("parent_asin")
            if asin in needed_asins and asin not in titles:
                titles[asin] = {
                    "title": record.get("title") or "",
                    "store": record.get("store") or "",
                }
                if len(titles) == len(needed_asins):
                    break
    finally:
        response.close()

    output_path = PROJECT_ROOT / "data" / "processed" / "product_titles.json"
    output_path.write_text(json.dumps(titles, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"✓ Matched {len(titles)}/{len(needed_asins)} products -> {output_path}")


if __name__ == "__main__":
    main()
