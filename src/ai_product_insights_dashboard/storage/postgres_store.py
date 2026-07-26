"""PostgreSQL store for review metadata and the Agent A analytics queries.

Per the PRD (§5), the vector DB answers similarity questions and Postgres
answers "top-N" / "trend over time" questions via GROUP BY — that split is
implemented here rather than by post-processing JSONL in Python.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Any


def _get_db_connection() -> Any:
    """Get a SQLAlchemy engine for DATABASE_URL."""
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise RuntimeError(
            "The `sqlalchemy` package is required. Install it with: pip install sqlalchemy"
        ) from exc

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError(
            "DATABASE_URL environment variable not set. "
            "Format: postgresql://user:password@host:5432/dbname"
        )

    return create_engine(db_url, echo=False, pool_pre_ping=True)


def init_schema() -> None:
    """Create tables if they don't exist.

    `review_id` is TEXT, not UUID: the Amazon Reviews 2023 dataset has no
    review-level ID, so ingestion derives a stable id by hashing identifying
    fields (see `chunk_reviews.py`). Using that same id as the primary key
    (instead of minting a random UUID per insert) is what makes `ON CONFLICT
    DO NOTHING` actually dedupe on re-ingestion.
    """
    from sqlalchemy import text

    engine = _get_db_connection()

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS reviews (
                review_id TEXT PRIMARY KEY,
                asin VARCHAR(20),
                parent_asin VARCHAR(20),
                review_text TEXT,
                rating SMALLINT,
                review_date DATE,
                verified_purchase BOOLEAN,
                helpful_vote INTEGER,
                user_tier VARCHAR(20),
                sentiment VARCHAR(10),
                cluster_id INTEGER,
                cluster_label TEXT,
                urgency_flag BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_reviews_cluster ON reviews(cluster_id, review_date)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews(review_date)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_reviews_tier ON reviews(user_tier)"
        ))


def _to_review_date(timestamp: Any) -> date:
    """Amazon Reviews 2023 timestamps are millisecond epoch, not seconds."""
    if isinstance(timestamp, (int, float)) and timestamp > 0:
        return datetime.fromtimestamp(timestamp / 1000).date()
    return datetime.now().date()


def _derive_fields(source: dict[str, Any]) -> dict[str, Any]:
    rating = source.get("rating")
    rating_int = int(rating) if rating not in (None, "") else None
    return {
        "asin": str(source.get("asin", ""))[:20],
        "parent_asin": str(source.get("parent_asin", ""))[:20],
        "rating": rating_int,
        "review_date": _to_review_date(source.get("timestamp")),
        "verified_purchase": bool(source.get("verified_purchase", False)),
        "helpful_vote": int(source.get("helpful_vote") or 0),
        # Synthetic per PRD §4: no `raw_meta_Software`/price join in this build,
        # so tier is bootstrapped from rating and disclosed as synthetic in the README.
        "user_tier": "premium" if rating_int and rating_int >= 4 else "free",
        "sentiment": "negative" if rating_int and rating_int <= 2 else "positive",
    }


def save_review_document(review: dict[str, Any], review_id: str | None = None) -> str:
    """Save a single review to PostgreSQL. Returns the review_id used."""
    from sqlalchemy import text

    engine = _get_db_connection()
    resolved_id = str(review_id or review.get("review_id") or review.get("id"))
    fields = _derive_fields(review)
    review_text = review.get("text") or review.get("review_text") or ""

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO reviews (
                review_id, asin, parent_asin, review_text, rating,
                review_date, verified_purchase, helpful_vote,
                user_tier, sentiment
            ) VALUES (
                :review_id, :asin, :parent_asin, :review_text, :rating,
                :review_date, :verified_purchase, :helpful_vote,
                :user_tier, :sentiment
            )
            ON CONFLICT (review_id) DO NOTHING
        """), {"review_id": resolved_id, "review_text": str(review_text), **fields})

    return resolved_id


def save_documents(documents: list[dict[str, Any]], batch_size: int = 1000) -> int:
    """Bulk save chunk/review records to PostgreSQL, idempotent on review_id.

    Batches via SQLAlchemy's executemany-style multi-params execute instead
    of one round-trip per row — at 1k reviews the per-row loop was fine, but
    it turned a 20k-review load into ~20,000 sequential network round-trips
    to a remote (Neon) database, which is the difference between seconds
    and many minutes.
    """
    from sqlalchemy import text

    engine = _get_db_connection()

    rows = []
    for doc in documents:
        metadata = doc.get("metadata", {}) or {}
        review_id = str(doc.get("review_id") or doc.get("id"))
        fields = _derive_fields(metadata)
        rows.append({"review_id": review_id, "review_text": str(doc.get("text", "")), **fields})

    if not rows:
        return 0

    stmt = text("""
        INSERT INTO reviews (
            review_id, asin, parent_asin, review_text, rating,
            review_date, verified_purchase, helpful_vote,
            user_tier, sentiment
        ) VALUES (
            :review_id, :asin, :parent_asin, :review_text, :rating,
            :review_date, :verified_purchase, :helpful_vote,
            :user_tier, :sentiment
        )
        ON CONFLICT (review_id) DO NOTHING
    """)

    with engine.begin() as conn:
        for start in range(0, len(rows), batch_size):
            conn.execute(stmt, rows[start:start + batch_size])

    # psycopg2 doesn't reliably sum rowcount across a batched executemany, so
    # this is "rows attempted" rather than "rows actually inserted" (some may
    # have hit ON CONFLICT DO NOTHING) — fine for the progress count this feeds.
    return len(rows)


def update_cluster_assignments(assignments: dict[str, tuple[int | None, str]], batch_size: int = 1000) -> int:
    """Write cluster_id/cluster_label back onto reviews, keyed by review_id."""
    from sqlalchemy import text

    engine = _get_db_connection()

    rows = [
        {"review_id": review_id, "cluster_id": cluster_id, "cluster_label": cluster_label}
        for review_id, (cluster_id, cluster_label) in assignments.items()
    ]
    if not rows:
        return 0

    stmt = text("""
        UPDATE reviews
        SET cluster_id = :cluster_id, cluster_label = :cluster_label, updated_at = CURRENT_TIMESTAMP
        WHERE review_id = :review_id
    """)

    with engine.begin() as conn:
        for start in range(0, len(rows), batch_size):
            conn.execute(stmt, rows[start:start + batch_size])

    return len(rows)


def get_dashboard_metrics(window_days: int = 30) -> dict[str, Any]:
    """Real SQL aggregation for the dashboard: per-cluster metrics + growth.

    Growth is anchored to the most recent `review_date` in the table rather
    than wall-clock `now()` — the corpus is a static historical bulk load
    (PRD §5 replays it in timestamp order to simulate a live feed), so
    "recent" has to be relative to the data's own clock, not the server's.
    """
    from sqlalchemy import text

    engine = _get_db_connection()

    with engine.connect() as conn:
        anchor = conn.execute(text("SELECT MAX(review_date) FROM reviews")).scalar()
        if anchor is None:
            return {"total_reviews": 0, "avg_rating": 0.0, "negative_review_percentage": 0.0,
                     "clusters": [], "window_days": window_days, "anchor_date": None}

        current_start = anchor - timedelta(days=window_days)
        prior_start = current_start - timedelta(days=window_days)

        cluster_rows = conn.execute(text("""
            SELECT cluster_id, cluster_label,
                   COUNT(*) AS count,
                   AVG(rating) AS avg_rating,
                   SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) AS negative_count,
                   SUM(CASE WHEN verified_purchase THEN 1 ELSE 0 END) AS verified_count,
                   COALESCE(SUM(helpful_vote), 0) AS helpful_votes_total
            FROM reviews
            WHERE cluster_id IS NOT NULL
            GROUP BY cluster_id, cluster_label
        """)).mappings().all()

        current_counts = dict(conn.execute(text("""
            SELECT cluster_id, COUNT(*) FROM reviews
            WHERE cluster_id IS NOT NULL AND review_date > :start
            GROUP BY cluster_id
        """), {"start": current_start}).all())

        prior_counts = dict(conn.execute(text("""
            SELECT cluster_id, COUNT(*) FROM reviews
            WHERE cluster_id IS NOT NULL AND review_date > :pstart AND review_date <= :start
            GROUP BY cluster_id
        """), {"pstart": prior_start, "start": current_start}).all())

        overall = conn.execute(text("""
            SELECT COUNT(*) AS total, AVG(rating) AS avg_rating,
                   SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) AS negative_count
            FROM reviews
        """)).mappings().one()

        # For the sparkline: cheap to pull all (cluster_id, review_date) pairs
        # in one shot at this corpus size rather than a per-cluster query.
        date_rows = conn.execute(text("""
            SELECT cluster_id, review_date FROM reviews WHERE cluster_id IS NOT NULL
        """)).all()

    dates_by_cluster: dict[Any, list[str]] = {}
    for cid, review_date in date_rows:
        if review_date is not None:
            dates_by_cluster.setdefault(cid, []).append(review_date.isoformat())

    clusters = []
    for row in cluster_rows:
        cid = row["cluster_id"]
        count = row["count"] or 0
        current = current_counts.get(cid, 0)
        prior = prior_counts.get(cid, 0)
        growth_rate = (current / prior - 1) if prior else (1.0 if current else 0.0)
        avg_rating = float(row["avg_rating"] or 0)
        clusters.append({
            "cluster_id": cid,
            "cluster_label": row["cluster_label"] or f"Cluster {cid}",
            "count": count,
            "avg_rating": avg_rating,
            "negative_percentage": (row["negative_count"] / count * 100) if count else 0.0,
            "verified_purchase_percentage": (row["verified_count"] / count * 100) if count else 0.0,
            "helpful_votes_total": int(row["helpful_votes_total"] or 0),
            "growth_rate": round(growth_rate, 3),
            "recent_count": current,
            "sentiment": "negative" if avg_rating <= 3 else "positive",
            "dates": dates_by_cluster.get(cid, []),
        })

    total = overall["total"] or 0
    return {
        "total_reviews": total,
        "avg_rating": float(overall["avg_rating"] or 0),
        "negative_review_percentage": (overall["negative_count"] / total * 100) if total else 0.0,
        "clusters": clusters,
        "window_days": window_days,
        "anchor_date": anchor.isoformat(),
    }


def get_cluster_reviews(cluster_id: int, limit: int = 10) -> list[dict[str, Any]]:
    """Representative "top complaints" for a theme: worst rating first, then
    most-helpful-voted, so the most impactful negative reviews surface first.
    """
    from sqlalchemy import text

    engine = _get_db_connection()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT review_id, review_text, rating, helpful_vote, verified_purchase, asin, parent_asin
            FROM reviews
            WHERE cluster_id = :cluster_id
            ORDER BY rating ASC, helpful_vote DESC
            LIMIT :limit
        """), {"cluster_id": cluster_id, "limit": limit}).mappings().all()

    return [dict(row) for row in rows]
