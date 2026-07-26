"""
Phase 2b: Analytics API Endpoints
- /api/v1/trends - Top/fastest-growing clusters
- /api/v1/complaints/top - Ranked complaints by urgency
- /api/v1/dashboard/summary - Weekly rollup
- /api/v1/alerts/urgent - Critical issue clusters

Per the PRD (§5), aggregation/trend questions are answered by Postgres
(GROUP BY), not the vector DB. When DATABASE_URL is configured, these
endpoints run real SQL aggregation via `storage.postgres_store`; otherwise
they fall back to computing the same metrics from the local JSONL corpus
so the demo still works without a database.
"""

import os
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analytics"])


# ===== DATA MODELS =====

class ClusterMetrics(BaseModel):
    cluster_id: Optional[int]
    cluster_label: str
    count: int
    avg_rating: float
    negative_percentage: float
    verified_purchase_percentage: float
    helpful_votes_total: int
    growth_rate: float = 0.0
    recent_count: int = 0
    sentiment: str = "neutral"
    pain_index: int = 0
    emerging: bool = False
    sparkline: List[int] = []


class ThemeReview(BaseModel):
    review_id: Optional[str]
    text: str
    rating: Optional[float]
    helpful_vote: int
    verified_purchase: bool
    product: str


class ThemeDetail(ClusterMetrics):
    top_reviews: List[ThemeReview]


class ComplaintItem(BaseModel):
    cluster_id: Optional[int]
    cluster_label: str
    complaint_count: int
    avg_rating: float
    severity_score: float  # 0-100
    top_issues: List[str]


class DashboardSummary(BaseModel):
    total_reviews: int
    total_clusters: int
    avg_rating: float
    negative_review_percentage: float
    top_clusters: List[ClusterMetrics]
    top_complaints: List[ComplaintItem]
    collection_period: str


class Alert(BaseModel):
    cluster_id: Optional[int]
    cluster_label: str
    urgency_level: str  # "critical", "high", "medium"
    issue_count: int
    avg_rating: float
    urgency_score: float


# ===== SHARED AGGREGATION LAYER =====

def _pain_index(count: int, avg_rating: float, negative_percentage: float) -> int:
    """volume x bad-outcome severity — same spirit as a support-ticket "pain"
    score, adapted to review fields: there's no CSAT/churn/SLA-breach here,
    so severity is negative-review share plus how far avg_rating sits below
    a neutral 3.5/5, and pain scales with how many reviews carry that
    severity. Weights/scale are tunable, not derived from anything external.
    """
    severity = (negative_percentage / 100) * 1.5 + max(0.0, (3.5 - avg_rating) / 3.5)
    return round(count * severity * 20)


def _is_emerging(growth_rate: float, recent_count: int) -> bool:
    return growth_rate >= 0.5 and recent_count >= 3


def _bucket_sparkline(dates: List[str], buckets: int = 8) -> List[int]:
    """Chronological volume shape for a theme, independent of calendar time.

    This corpus is a historical bulk load spread across two decades with
    very uneven density (see PRD §5) — fixed calendar buckets (e.g. one per
    week) would leave most buckets at zero for any theme with sparse dates.
    Splitting each theme's own reviews into N equal chronological groups
    always produces a populated shape, at the cost of the x-axis not being
    literal wall-clock time.
    """
    valid = sorted(d for d in dates if d)
    if not valid:
        return [0] * buckets
    n = len(valid)
    bucket_size = max(1, -(-n // buckets))  # ceil division
    counts = [len(valid[i:i + bucket_size]) for i in range(0, n, bucket_size)]
    counts += [0] * (buckets - len(counts))
    return counts[:buckets]


def _enrich_clusters(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach pain_index/emerging/sparkline; used by every endpoint so the
    ranking is consistent whether the caller hit /trends, /themes, or the
    dashboard summary."""
    for c in clusters:
        c["pain_index"] = _pain_index(c["count"], c["avg_rating"], c["negative_percentage"])
        c["emerging"] = _is_emerging(c["growth_rate"], c["recent_count"])
        c["sparkline"] = _bucket_sparkline(c.pop("dates", []))
    return clusters


def _parse_window_days(window: str) -> int:
    """Parse a PRD-style window string like '30d' into a day count."""
    window = (window or "30d").strip().lower()
    if window.endswith("d"):
        try:
            return max(1, int(window[:-1]))
        except ValueError:
            pass
    return 30


def load_processed_chunks(path: str = "data/processed/clustered_chunks.jsonl") -> List[Dict[str, Any]]:
    """Load review chunks for JSONL-fallback analytics (no DATABASE_URL set).

    Prefers the clustered file so the dashboard reflects real cluster labels.
    Falls back to the pre-clustering ingestion output if clustering hasn't
    been run yet, or if it's stale relative to the latest ingestion.
    """
    chunks = []
    try:
        clustered_path = Path(path)
        raw_path = Path("data/processed/review_chunks.jsonl")

        use_path = clustered_path
        if not clustered_path.exists():
            use_path = raw_path
        elif raw_path.exists() and raw_path.stat().st_mtime > clustered_path.stat().st_mtime:
            # A newer ingestion run has produced chunks the clustering step
            # hasn't seen yet; show the current corpus instead of a stale
            # clustered snapshot from a previous run.
            use_path = raw_path

        if use_path.exists():
            with open(use_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunks.append(json.loads(line))
    except Exception as e:
        logger.warning(f"Could not load {path}: {e}")

    return chunks


def _cluster_label(chunk: Dict[str, Any]) -> str:
    """Return a useful label before the optional clustering step has run."""
    cluster_id = chunk.get("cluster_id")
    return chunk.get("cluster_label") or (
        f"Cluster {cluster_id}" if cluster_id is not None else "Amazon Software reviews"
    )


def _unique_reviews(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Use one representative chunk per review for review-level metrics."""
    reviews: Dict[str, Dict[str, Any]] = {}
    for index, chunk in enumerate(chunks):
        review_id = str(chunk.get("review_id") or f"chunk-{index}")
        reviews.setdefault(review_id, chunk)
    return list(reviews.values())


def calculate_cluster_metrics(chunks: List[Dict], cluster_id: Optional[int]) -> Dict[str, Any]:
    """Calculate metrics for a specific cluster."""
    cluster_chunks = _unique_reviews(
        [c for c in chunks if c.get('cluster_id') == cluster_id]
    )

    if not cluster_chunks:
        return {}

    ratings = [c['metadata'].get('rating', 3) for c in cluster_chunks]
    helpful_votes = [c['metadata'].get('helpful_vote', 0) for c in cluster_chunks]
    verified = [c['metadata'].get('verified_purchase', False) for c in cluster_chunks]

    negative_count = sum(1 for r in ratings if r <= 2)
    verified_count = sum(1 for v in verified if v)

    return {
        'avg_rating': sum(ratings) / len(ratings),
        'negative_percentage': (negative_count / len(ratings)) * 100,
        'verified_purchase_percentage': (verified_count / len(verified)) * 100 if verified else 0,
        'helpful_votes_total': sum(helpful_votes),
        'count': len(cluster_chunks),
    }


def _jsonl_cluster_metrics(window_days: int) -> Dict[str, Any]:
    """Fallback aggregation over the local JSONL corpus (no Postgres)."""
    empty = {"total_reviews": 0, "avg_rating": 0.0, "negative_review_percentage": 0.0,
             "clusters": [], "window_days": window_days, "anchor_date": None}

    chunks = load_processed_chunks()
    if not chunks:
        return empty

    reviews = _unique_reviews(chunks)
    timestamps = [r.get('metadata', {}).get('timestamp') for r in reviews]
    timestamps = [t for t in timestamps if isinstance(t, (int, float)) and t > 0]

    # Amazon Reviews 2023 timestamps are millisecond epoch. Anchor "now" to
    # the newest review in the corpus rather than wall-clock time, since the
    # corpus is a static historical bulk load replayed in timestamp order
    # (PRD §5) rather than a live feed.
    anchor_dt = datetime.fromtimestamp(max(timestamps) / 1000) if timestamps else None
    current_start = anchor_dt - timedelta(days=window_days) if anchor_dt else None
    prior_start = current_start - timedelta(days=window_days) if current_start else None

    cluster_groups: Dict[Any, Dict[str, Any]] = {}
    for chunk in chunks:
        cid = chunk.get('cluster_id')
        label = _cluster_label(chunk)
        cluster_groups.setdefault(cid, {'label': label, 'chunks': []})['chunks'].append(chunk)

    clusters = []
    for cid, group in cluster_groups.items():
        metrics = calculate_cluster_metrics(chunks, cid)
        if not metrics:
            continue

        current = prior = 0
        dates = []
        for r in _unique_reviews(group['chunks']):
            ts = r.get('metadata', {}).get('timestamp')
            if not isinstance(ts, (int, float)) or ts <= 0:
                continue
            dt = datetime.fromtimestamp(ts / 1000)
            dates.append(dt.date().isoformat())
            if not anchor_dt:
                continue
            if dt > current_start:
                current += 1
            elif dt > prior_start:
                prior += 1

        growth_rate = (current / prior - 1) if prior else (1.0 if current else 0.0)
        clusters.append({
            'cluster_id': cid,
            'cluster_label': group['label'],
            **metrics,
            'growth_rate': round(growth_rate, 3),
            'recent_count': current,
            'sentiment': 'negative' if metrics['avg_rating'] <= 3 else 'positive',
            'dates': dates,
        })

    all_ratings = [r.get('metadata', {}).get('rating', 3) for r in reviews]
    negative = sum(1 for r in all_ratings if r <= 2)

    return {
        'total_reviews': len(reviews),
        'avg_rating': sum(all_ratings) / len(all_ratings) if all_ratings else 0.0,
        'negative_review_percentage': (negative / len(all_ratings) * 100) if all_ratings else 0.0,
        'clusters': clusters,
        'window_days': window_days,
        'anchor_date': anchor_dt.date().isoformat() if anchor_dt else None,
    }


def get_cluster_metrics(window_days: int = 30) -> Dict[str, Any]:
    """Single aggregation entry point shared by all analytics endpoints.

    Uses Postgres (real GROUP BY) when DATABASE_URL is configured; falls
    back to the JSONL corpus so the API still works in a database-less demo.
    """
    if os.getenv("DATABASE_URL"):
        try:
            from ..storage.postgres_store import get_dashboard_metrics
            data = get_dashboard_metrics(window_days)
            if data["total_reviews"] > 0:
                data["clusters"] = _enrich_clusters(data["clusters"])
                return data
            logger.info("Postgres has no reviews yet; falling back to JSONL")
        except Exception as e:
            logger.warning(f"Postgres aggregation failed ({e}); falling back to JSONL")

    data = _jsonl_cluster_metrics(window_days)
    data["clusters"] = _enrich_clusters(data["clusters"])
    return data


_product_titles_cache: Optional[Dict[str, Dict[str, str]]] = None


def _product_titles() -> Dict[str, Dict[str, str]]:
    """Same parent_asin -> {title, store} lookup used for /ask citations
    (see scripts/build_product_titles.py) — reused here so theme detail
    reviews show a real product name instead of a bare ASIN."""
    global _product_titles_cache
    if _product_titles_cache is None:
        try:
            _product_titles_cache = json.loads(
                Path("data/processed/product_titles.json").read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            _product_titles_cache = {}
    return _product_titles_cache


def _resolve_product(parent_asin: Optional[str], asin: Optional[str]) -> str:
    info = _product_titles().get(parent_asin, {})
    return info.get("title") or parent_asin or asin or "unknown product"


def _theme_reviews(cluster_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """Top complaint excerpts for a theme's detail view: worst rating first."""
    if os.getenv("DATABASE_URL"):
        try:
            from ..storage.postgres_store import get_cluster_reviews
            rows = get_cluster_reviews(cluster_id, limit=limit)
            if rows:
                return [
                    {
                        "review_id": r.get("review_id"),
                        "text": (r.get("review_text") or "")[:400],
                        "rating": r.get("rating"),
                        "helpful_vote": r.get("helpful_vote") or 0,
                        "verified_purchase": bool(r.get("verified_purchase")),
                        "product": _resolve_product(r.get("parent_asin"), r.get("asin")),
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.warning(f"Postgres cluster reviews failed ({e}); falling back to JSONL")

    chunks = load_processed_chunks()
    cluster_chunks = _unique_reviews([c for c in chunks if c.get('cluster_id') == cluster_id])
    cluster_chunks.sort(key=lambda c: (c.get('metadata', {}).get('rating', 3), -(c.get('metadata', {}).get('helpful_vote', 0) or 0)))

    reviews = []
    for c in cluster_chunks[:limit]:
        meta = c.get('metadata', {}) or {}
        reviews.append({
            "review_id": c.get("review_id"),
            "text": (c.get("text") or "")[:400],
            "rating": meta.get("rating"),
            "helpful_vote": meta.get("helpful_vote") or 0,
            "verified_purchase": bool(meta.get("verified_purchase")),
            "product": _resolve_product(meta.get("parent_asin"), meta.get("asin")),
        })
    return reviews


# ===== ENDPOINTS =====

@router.get("/trends", response_model=List[ClusterMetrics])
async def get_trends(
    limit: int = Query(10, ge=1, le=50),
    window: str = Query("30d", description="Growth window, e.g. '30d' or '90d'"),
    min_growth: float = Query(0.0, description="Only include clusters growing at least this fast"),
    sort: str = Query("count", pattern="^(count|growth)$", description="Sort by review volume or growth rate"),
) -> List[ClusterMetrics]:
    """
    Get top/fastest-growing clusters.
    **Metrics**: Cluster size, avg rating, negative %, growth rate over `window`
    """
    data = get_cluster_metrics(_parse_window_days(window))
    if not data["clusters"]:
        raise HTTPException(status_code=404, detail="No processed data available")

    clusters = [c for c in data["clusters"] if c["growth_rate"] >= min_growth]
    sort_key = "count" if sort == "count" else "growth_rate"
    clusters.sort(key=lambda c: c[sort_key], reverse=True)

    return [ClusterMetrics(**c) for c in clusters[:limit]]


@router.get("/themes", response_model=List[ClusterMetrics])
async def get_themes(
    window: str = Query("30d", description="Growth window, e.g. '30d' or '90d'"),
    sort: str = Query("pain", pattern="^(pain|count|growth)$"),
) -> List[ClusterMetrics]:
    """
    All discovered themes (HDBSCAN clusters), ranked by pain_index by default.
    Excludes the unclustered "noise" bucket — a theme is by definition a
    coherent cluster; noise isn't one.
    """
    data = get_cluster_metrics(_parse_window_days(window))
    clusters = [c for c in data["clusters"] if c["cluster_id"] is not None]
    if not clusters:
        raise HTTPException(status_code=404, detail="No clustered themes available yet")

    sort_key = {"pain": "pain_index", "count": "count", "growth": "growth_rate"}[sort]
    clusters.sort(key=lambda c: c[sort_key], reverse=True)
    return [ClusterMetrics(**c) for c in clusters]


@router.get("/themes/{cluster_id}", response_model=ThemeDetail)
async def get_theme_detail(
    cluster_id: int,
    window: str = Query("30d", description="Growth window, e.g. '30d' or '90d'"),
) -> ThemeDetail:
    """A single theme's full scorecard plus its top complaint excerpts."""
    data = get_cluster_metrics(_parse_window_days(window))
    match = next((c for c in data["clusters"] if c["cluster_id"] == cluster_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Theme not found")

    reviews = _theme_reviews(cluster_id, limit=10)
    return ThemeDetail(**match, top_reviews=[ThemeReview(**r) for r in reviews])


@router.get("/complaints/top", response_model=List[ComplaintItem])
async def get_top_complaints(limit: int = Query(10, ge=1, le=50)) -> List[ComplaintItem]:
    """
    Get top complaint clusters ranked by severity.
    **Severity Formula**: (negative_percentage × 2) + (1 - avg_rating/5 × 100)
    """
    data = get_cluster_metrics()
    if not data["clusters"]:
        raise HTTPException(status_code=404, detail="No processed data available")

    complaints = []
    for c in data["clusters"]:
        if c["negative_percentage"] > 20:  # At least 20% negative
            severity = (c["negative_percentage"] * 2) + ((5 - c["avg_rating"]) * 20)
            complaints.append(ComplaintItem(
                cluster_id=c["cluster_id"],
                cluster_label=c["cluster_label"],
                complaint_count=c["count"],
                avg_rating=c["avg_rating"],
                severity_score=min(100, severity),
                top_issues=[],
            ))

    complaints.sort(key=lambda x: x.severity_score, reverse=True)
    return complaints[:limit]


@router.get("/dashboard/summary", response_model=DashboardSummary)
async def get_dashboard_summary(
    window: str = Query("30d", description="Growth window, e.g. '30d' or '90d'"),
) -> DashboardSummary:
    """
    Get dashboard summary: total reviews, cluster breakdown, top issues.
    **Returns**: Overall metrics + top 5 clusters (with growth over `window`) + top 5 complaints
    """
    data = get_cluster_metrics(_parse_window_days(window))
    if not data["clusters"] and data["total_reviews"] == 0:
        raise HTTPException(status_code=404, detail="No processed data available")

    top_clusters = sorted(data["clusters"], key=lambda c: c["count"], reverse=True)[:5]

    complaints = [c for c in data["clusters"] if c["negative_percentage"] > 20]
    complaints.sort(
        key=lambda c: (c["negative_percentage"] * 2) + ((5 - c["avg_rating"]) * 20),
        reverse=True,
    )
    top_complaints = [
        ComplaintItem(
            cluster_id=c["cluster_id"],
            cluster_label=c["cluster_label"],
            complaint_count=c["count"],
            avg_rating=c["avg_rating"],
            severity_score=min(100, (c["negative_percentage"] * 2) + ((5 - c["avg_rating"]) * 20)),
            top_issues=[],
        )
        for c in complaints[:5]
    ]

    period = f"Last {data['window_days']}d as of {data['anchor_date']}" if data["anchor_date"] else "No data yet"
    return DashboardSummary(
        total_reviews=data["total_reviews"],
        total_clusters=len(data["clusters"]) or (1 if data["total_reviews"] else 0),
        avg_rating=data["avg_rating"],
        negative_review_percentage=data["negative_review_percentage"],
        top_clusters=[ClusterMetrics(**c) for c in top_clusters],
        top_complaints=top_complaints,
        collection_period=period,
    )


@router.get("/alerts/urgent", response_model=List[Alert])
async def get_urgent_alerts(threshold: float = Query(60, ge=0, le=100)) -> List[Alert]:
    """
    Get critical alert clusters (high negative %, low rating).
    **Urgency Score**: (negative_percentage × 2) + (helpfulness_ratio × 30)
    """
    data = get_cluster_metrics()
    if not data["clusters"]:
        raise HTTPException(status_code=404, detail="No processed data available")

    alerts = []
    for c in data["clusters"]:
        helpful_ratio = c["helpful_votes_total"] / max(c["count"], 1)
        urgency = (c["negative_percentage"] * 2) + (helpful_ratio * 30)

        if urgency >= threshold and c["negative_percentage"] > 30:
            level = "critical" if urgency > 80 else "high" if urgency > 60 else "medium"
            alerts.append(Alert(
                cluster_id=c["cluster_id"],
                cluster_label=c["cluster_label"],
                urgency_level=level,
                issue_count=c["count"],
                avg_rating=c["avg_rating"],
                urgency_score=min(100, urgency),
            ))

    alerts.sort(key=lambda x: x.urgency_score, reverse=True)
    return alerts


# ===== PHASE 3: RAG & MULTI-AGENT =====

class QuestionRequest(BaseModel):
    query: str
    top_k: int = 5


class QuestionResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict]
    confidence: float
    metadata: Dict


class PRDRequest(BaseModel):
    template: str = "standard"  # standard, lean, executive


class PRDResponse(BaseModel):
    status: str
    template: str
    prd: str
    metadata: Dict


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    """
    Ask a question about product reviews using RAG.

    **Features**:
    - Query rewriting for better retrieval
    - Self-correcting retrieval
    - LLM-powered answers with source attribution

    **Example Query**: "What are the main complaints about this product?"
    """
    from ..rag.answer_question import answer_question

    result = answer_question(request.query)

    return QuestionResponse(
        query=request.query,
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
        metadata={
            "total_chunks_evaluated": result.get("total_chunks_evaluated", 0),
            "relevant_chunks": result.get("relevant_chunks_used", 0),
            "iterations": result.get("iterations", 1),
            "self_correction_applied": result.get("self_correction_applied", False),
            "correction_reason": result.get("correction_reason"),
        }
    )


@router.post("/ask/stream")
async def ask_question_stream(request: QuestionRequest) -> StreamingResponse:
    """
    Same question-answering as /ask, but streams the answer as newline-delimited
    JSON events (`{"type": "sources"|"token"|"error", ...}`) so the chat UI can
    render tokens as they're generated instead of waiting for the full answer.

    Trades the self-correction retry loop for immediacy — see `stream_answer`'s
    docstring. Use /ask instead when the retry guarantee matters more than
    latency (e.g. programmatic callers).
    """
    from ..rag.crag_graph import stream_answer

    return StreamingResponse(stream_answer(request.query), media_type="application/x-ndjson")


@router.post("/prd/generate", response_model=PRDResponse)
async def generate_prd(request: PRDRequest) -> PRDResponse:
    """
    Generate Product Requirements Document from review insights.

    **Templates**:
    - `standard`: Full PRD with all sections
    - `lean`: Executive summary + top 3 issues
    - `executive`: High-level overview

    **Returns**: Structured PRD with priority ranking and timeline
    """
    from ..agents.prd_agent import PRDAgent

    agent = PRDAgent()
    result = await agent.process(request.template)

    return PRDResponse(
        status=result.get("status", "error"),
        template=request.template,
        prd=result.get("prd", "Error generating PRD"),
        metadata=result.get("metadata", {})
    )


@router.post("/route", response_model=Dict)
async def route_request(query: str, request_type: Optional[str] = None) -> Dict:
    """
    Route query to appropriate agent (Q&A or PRD).
    Orchestrator automatically detects type if not specified.

    **Auto-Detection**:
    - "What...", "How...", "Tell me..." → Q&A Agent
    - "PRD", "Generate", "Requirements" → PRD Agent
    """
    from ..agents.orchestrator import route_request as orchestrator_route

    result = await orchestrator_route(query, request_type)
    return result


# ===== APP INTEGRATION =====

def get_routes():
    """Return the configured API router."""
    return router
