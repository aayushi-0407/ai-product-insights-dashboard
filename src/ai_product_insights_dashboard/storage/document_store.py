"""Document storage for processed review data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOCUMENT_PATH = PROJECT_ROOT / "data" / "processed" / "reviews.jsonl"


def save_documents(
    documents: list[dict[str, Any]],
    output_path: Path | str = DEFAULT_DOCUMENT_PATH,
) -> Path:
    """Persist processed review documents and metadata as JSONL."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False))
            handle.write("\n")

    return path
