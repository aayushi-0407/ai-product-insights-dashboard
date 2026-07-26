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


def _iter_local_jsonl(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            yield json.loads(line)


def _iter_remote_jsonl(
    source: str,
    config_name: str,
    limit: int | None,
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
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    try:
        for index, line in enumerate(response.iter_lines(decode_unicode=True)):
            if limit is not None and index >= limit:
                break
            if line:
                yield json.loads(line)
    finally:
        response.close()


def fetch_reviews(
    source: str,
    config_name: str = "raw_review_Software",
    split: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch reviews from the PRD dataset source.

    The PRD points to `McAuley-Lab/Amazon-Reviews-2023` and the
    `raw_review_Software` review file. Uses the Hugging Face datasets library
    for better reliability and caching.
    """

    if split is not None:
        raise ValueError("split is not used for this JSONL-backed dataset")

    source_path = Path(source)
    if source_path.exists():
        records = list(_iter_local_jsonl(source_path, limit=limit))
    else:
        # Newer versions of `datasets` no longer execute the loading script
        # used by Amazon Reviews 2023. Fall back to the repository's JSONL
        # file so the PRD dataset remains loadable across library versions.
        try:
            records = list(_iter_hf_dataset(source, config_name=config_name, limit=limit))
        except RuntimeError as exc:
            if "scripts are no longer supported" not in str(exc):
                raise
            print("  Falling back to Hugging Face JSONL streaming...")
            records = list(_iter_remote_jsonl(source, config_name, limit))

    return records


def _iter_hf_dataset(
    source: str,
    config_name: str,
    limit: int | None,
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
    
    count = 0
    for record in data:
        if limit is not None and count >= limit:
            break
        yield dict(record)
        count += 1
