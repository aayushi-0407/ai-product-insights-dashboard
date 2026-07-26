"""Load review data from the source dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _review_file_from_config(config_name: str) -> str:
    if config_name.startswith("raw_review_"):
        category = config_name.removeprefix("raw_review_")
        return f"raw/review_categories/{category}.jsonl"
    if config_name.startswith("raw_meta_"):
        category = config_name.removeprefix("raw_meta_")
        return f"raw/meta_categories/meta_{category}.jsonl"
    raise ValueError(
        "config_name must start with `raw_review_` or `raw_meta_` for this dataset"
    )


def _iter_local_jsonl(
    path: Path, limit: int | None, parent_asin: str | None = None
) -> Iterable[dict[str, Any]]:
    matched = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if parent_asin and record.get("parent_asin") != parent_asin:
                continue
            if limit is not None and matched >= limit:
                break
            yield record
            matched += 1


def _iter_remote_jsonl(
    source: str,
    config_name: str,
    limit: int | None,
    parent_asin: str | None = None,
) -> Iterable[dict[str, Any]]:
    try:
        import truststore
        import requests
        from huggingface_hub import hf_hub_url
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The `requests`, `huggingface_hub`, and `truststore` packages are required to stream the dataset from Hugging Face."
        ) from exc

    # Use the operating system's trust store. This supports managed Windows
    # environments whose HTTPS proxy certificate is not in certifi.
    truststore.inject_into_ssl()
    filename = _review_file_from_config(config_name)
    url = hf_hub_url(repo_id=source, repo_type="dataset", filename=filename)
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    matched = 0
    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            record = json.loads(line)
            if parent_asin and record.get("parent_asin") != parent_asin:
                continue
            if limit is not None and matched >= limit:
                break
            yield record
            matched += 1
    finally:
        response.close()


def fetch_reviews(
    source: str,
    config_name: str = "raw_review_Software",
    split: str | None = None,
    limit: int | None = None,
    parent_asin: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch reviews from the PRD dataset source.

    The PRD points to `McAuley-Lab/Amazon-Reviews-2023` and the
    `raw_review_Software` review file. Uses the Hugging Face datasets library
    for better reliability and caching.

    `parent_asin`, when set, filters to a single product's reviews — `limit`
    then counts matched records, not raw scanned lines, so `limit=10000`
    with a `parent_asin` set returns up to 10,000 reviews of that product
    even though the source file must still be scanned in full to find them.
    """

    if split is not None:
        raise ValueError("split is not used for this JSONL-backed dataset")

    source_path = Path(source)
    if source_path.exists():
        records = list(_iter_local_jsonl(source_path, limit=limit, parent_asin=parent_asin))
    else:
        # Newer versions of `datasets` no longer execute the loading script
        # used by Amazon Reviews 2023. Fall back to the repository's JSONL
        # file so the PRD dataset remains loadable across library versions.
        try:
            records = list(_iter_hf_dataset(source, config_name=config_name, limit=limit, parent_asin=parent_asin))
        except RuntimeError as exc:
            if "scripts are no longer supported" not in str(exc):
                raise
            print("  Falling back to Hugging Face JSONL streaming...")
            records = list(_iter_remote_jsonl(source, config_name, limit, parent_asin=parent_asin))

    return records


def _iter_hf_dataset(
    source: str,
    config_name: str,
    limit: int | None,
    parent_asin: str | None = None,
) -> Iterable[dict[str, Any]]:
    """Load dataset using Hugging Face datasets library."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install it with: pip install datasets"
        ) from exc

    print(f"  Loading from Hugging Face (this may take a moment on first run)...")
    dataset = load_dataset(
        source,
        config_name,
        split=None,  # No split for this dataset
    )

    # Handle both dict and Dataset object returns
    if isinstance(dataset, dict):
        # If dict of datasets, try to get the default one
        data = next(iter(dataset.values()))
    else:
        data = dataset

    matched = 0
    for record in data:
        record = dict(record)
        if parent_asin and record.get("parent_asin") != parent_asin:
            continue
        if limit is not None and matched >= limit:
            break
        yield record
        matched += 1
