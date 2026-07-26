---
title: AI Product Insights Dashboard
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AI Product Insights Dashboard

A production-shaped RAG system that turns product review streams into actionable insights. Processes Amazon reviews with AI-powered clustering, trend detection, and conversational Q&A.

**Status**: Phase 1 (data pipeline) ✅ | Phase 2 (clustering & analytics) ✅ | Phase 3 (RAG agents, self-correcting CRAG, multi-agent orchestration) ✅

**Synthetic field disclosure**: `user_tier` (premium/free) is bootstrapped from `rating` at ingestion, not real subscription data — the Amazon Reviews 2023 dataset has no price/tier field for the `raw_review_Software` split used here. See PRD §4 for the real-`price`-via-`raw_meta_Software` alternative left as future work.

---

## 🚀 Quick Start

### 1. Install

```bash
cd "c:\Users\Arsh\Desktop\AI Dashboard"
pip install -e .
```

### 2. Configure

Copy `.env.example` → `.env` and fill in API keys:
```bash
OPENAI_API_KEY=sk-proj-...
PINECONE_API_KEY=pcsk_...
DATABASE_URL=postgresql://...
```

Get keys:
- **OpenAI**: https://platform.openai.com/api-keys
- **Pinecone**: https://app.pinecone.io/ (create index: `review-chunks`, dim=1536, metric=cosine)
- **PostgreSQL**: Optional (use `.env.example` URL pattern)

### 3. Verify Setup

```bash
python verify_setup.py
```

### 4. Ingest Reviews

```bash
# Test with 100 reviews
python scripts/ingest_reviews.py --limit 100 --init-db

# Full ingestion (all Amazon Software reviews)
python scripts/ingest_reviews.py
```

**What it does:**
1. Fetches reviews from Hugging Face (`McAuley-Lab/Amazon-Reviews-2023` / `raw_review_Software`)
2. Chunks text (120 tokens, 20-token overlap)
3. Embeds with OpenAI `text-embedding-3-small` (1536 dimensions)
4. Stores vectors in Pinecone
5. Stores metadata in PostgreSQL
6. Archives JSONL locally

---

## 📊 Dataset

**Amazon Reviews 2023 — Software**
- Source: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- Fields: `asin`, `rating` (1–5), `text`, `timestamp`, `verified_purchase`, `helpful_vote`, `user_id`
- Real timestamp + verified purchase → basis for product tier classification
- Synthetic fields: `user_tier` (free/premium by rating), `sentiment` (≤2 stars = negative)

---

## 🏗️ Architecture

### Data Pipeline
```
Hugging Face Dataset
         ↓
  [fetch_reviews] → JSONL stream from HF
         ↓
  [chunk_reviews] → 120-token chunks, 20-token overlap
         ↓
 [embed_reviews] → OpenAI text-embedding-3-small (1536-dim vectors)
         ↓
      ┌──────────────┬──────────────┬──────────────┐
      ↓              ↓              ↓              ↓
   Pinecone    PostgreSQL      Local JSONL    (Archive)
  (vectors)    (metadata)      (cache)
```

### Storage Layer

| Component | Type | Purpose |
|-----------|------|---------|
| **Pinecone** | Vector DB | Semantic search over embeddings (kNN retrieval) |
| **PostgreSQL** | Structured DB | Metadata: rating, timestamp, user_tier, sentiment, cluster_id |
| **Local JSONL** | File archive | Backup; replay ingestion pipeline |

### Embedding Model

- **Model**: OpenAI `text-embedding-3-small`
- **Dimension**: 1536
- **Batching**: 100 chunks per API call (efficient)
- **Cost**: ~$0.02 per 1M input tokens (10K reviews ≈ $0.02)

---

## 📁 Project Structure

```
src/ai_product_insights_dashboard/
├── ingestion/
│   └── fetch_reviews.py          # Stream from HF Dataset or local JSONL
├── processing/
│   ├── chunk_reviews.py          # Chunk text by token count
│   └── embed_reviews.py          # OpenAI embeddings (batched)
├── storage/
│   ├── vector_store.py           # Pinecone (primary) + Chroma fallback
│   ├── pinecone_store.py         # Pinecone client & upsert logic
│   ├── postgres_store.py         # PostgreSQL schema & document persistence
│   └── document_store.py         # Local JSONL archive
├── config/
│   └── settings.py               # Environment-driven config
└── ...

data/
├── raw/                          # Downloaded reviews.jsonl
├── interim/                      # Intermediate processing
├── processed/                    # Chunked + embedded review_chunks.jsonl
└── vector_db/chroma/             # Local vector DB (fallback)

scripts/
├── ingest_reviews.py             # Main pipeline (fetch→chunk→embed→store)
└── ...

tests/
├── test_ingestion.py
├── test_chunking.py
└── test_rag.py
```

---

## 🔧 Pipeline Options

Run ingestion with `--help` to see all options:

```bash
python scripts/ingest_reviews.py --help
```

Common scenarios:

**Test with 100 reviews** (quick validation):
```bash
python scripts/ingest_reviews.py --limit 100 --init-db
```

**Production ingestion** (all reviews, initialize DB):
```bash
python scripts/ingest_reviews.py --init-db
```

**Skip PostgreSQL** (Pinecone only):
```bash
python scripts/ingest_reviews.py --limit 1000
```

**Use local Chroma** (no Pinecone, for local dev):
```bash
python scripts/ingest_reviews.py --skip-vector-db --limit 100
```

---

## 📈 Next Phases

### Phase 2: Clustering & Analytics (in progress)
- [ ] HDBSCAN clustering over embeddings
- [ ] LLM-based cluster labeling ("Login bugs", "Pricing concerns", ...)
- [ ] Urgency scoring (density + negative sentiment)
- [ ] Agent A: Analytics endpoints (`/api/v1/trends`, `/api/v1/complaints/top`)

### Phase 3: RAG & Multi-Agent Orchestration (planned)
- [ ] CRAG self-correcting retrieval (relevance grading + query rewrite)
- [ ] Agent B: Q&A tool (`/api/v1/ask`)
- [ ] Agent B: PRD generation tool (`/api/v1/prd/generate`)
- [ ] Router: Question intent classification
- [ ] Notification service for urgent clusters

---

## 🛠️ Configuration Reference

### Environment Variables (`.env`)

```bash
# Required
OPENAI_API_KEY=sk-proj-...
PINECONE_API_KEY=pcsk_...

# Optional (PostgreSQL)
DATABASE_URL=postgresql://user:pass@localhost:5432/reviews_db

# Optional (advanced)
PINECONE_INDEX_NAME=review-chunks     # default
PINECONE_NAMESPACE=reviews            # default
```

### API Reference

See [SETUP.md](SETUP.md) for full API specs and examples.

---

## 📝 Implementation Notes

### Embedding Choice: OpenAI vs. Alternatives

We chose **OpenAI text-embedding-3-small** because:
- State-of-the-art quality (beats legacy models)
- 1536 dimensions well-suited for semantic search
- Cheap: $0.02 per 1M tokens
- Built-in rate limiting handling
- No local GPU needed (cloud-based)

Alternative: `voyage-2` (comparable quality, different pricing model)

### Data Model: Real vs. Synthetic Fields

**Real fields** from Amazon dataset:
- `rating`, `timestamp`, `verified_purchase`, `helpful_vote`, `asin`, `user_id`

**Synthetic fields** (TODO: upgrade with `raw_meta_Software`):
- `user_tier` = "premium" if rating ≥ 4, else "free"
  - **TODO v1.1**: Load `raw_meta_Software` to use real `price` field

**Derived fields**:
- `sentiment` = "negative" if rating ≤ 2, else "positive"
- `review_date` = parsed from Unix `timestamp`

### Why Pinecone + PostgreSQL?

**Common mistake**: Using only a vector DB for all queries.

- **Vector DB (Pinecone)**: Semantic similarity search → "What are people saying about crashes?"
- **Structured DB (PostgreSQL)**: Aggregations → "How many negative reviews in the last 30 days?"

Both are needed for a complete dashboard.

---

## ⚠️ Known Limitations (v1)

1. **Batch ingestion only** (no real-time streaming)
2. **Synthetic `user_tier`** (use rating as proxy; upgrade with metadata later)
3. **No clustering/labeling yet** (Phase 2)
4. **No RAG endpoints** (Phase 3)

---

## 🤝 Contributing

PRs welcome. See `tests/` for test patterns.

---

## 📚 References

- PRD: [AI_Product_Insights_Dashboard_PRD.md](AI_Product_Insights_Dashboard_PRD.md)
- Setup Guide: [SETUP.md](SETUP.md)
- Dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Pinecone: https://docs.pinecone.io/
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
