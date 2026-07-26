"""Tests for chunking scaffolding."""

from ai_product_insights_dashboard.processing.chunk_reviews import chunk_reviews


def test_chunk_reviews_splits_long_text() -> None:
    reviews = [
        {
            "review_id": "r1",
            "text": " ".join(f"token{i}" for i in range(10)),
            "rating": 4,
        }
    ]

    chunks = chunk_reviews(reviews, chunk_size=4, overlap=1)

    assert len(chunks) == 3
    assert chunks[0]["chunk_id"] == "r1::0"
    assert chunks[0]["token_count"] == 4
    assert chunks[-1]["text"].startswith("token6")


def test_chunk_reviews_ignores_blank_text() -> None:
    chunks = chunk_reviews([{"review_id": "r2", "text": "   "}])

    assert chunks == []
