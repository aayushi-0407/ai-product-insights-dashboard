# AI Product Insights Dashboard - TODO Checklist

## Phase 1: Data Pipeline ✅ COMPLETE

### Core Ingestion
- [x] Load reviews from Hugging Face (McAuley-Lab/Amazon-Reviews-2023)
- [x] Chunk review text (120 tokens, 20-token overlap)
- [x] Embed with local model (sentence-transformers, all-MiniLM-L6-v2, 384-dim) — FREE, NO API KEY
- [x] Store vectors in Pinecone (with Chroma fallback)
- [x] Store metadata in PostgreSQL
- [x] Archive JSONL locally
- [x] Idempotent ingestion (safe to re-run)

### Configuration & Setup
- [x] Environment variables (.env.example)
- [x] PyProject.toml with all dependencies
- [x] Settings module (settings.py)
- [x] Main ingestion script (scripts/ingest_reviews.py)
- [x] Verification script (verify_setup.py)

### Documentation
- [x] README.md (architecture, quick start, dataset)
- [x] SETUP.md (8-step setup guide)
- [x] IMPLEMENTATION.md (technical details)
- [x] CHECKLIST.md (verification tests)
- [x] .env.example (configuration template)

### Testing
- [x] Python syntax validation
- [x] Sample JSONL test data
- [x] End-to-end pipeline test (10 reviews)
- [x] Local embedding model working ✅
- [x] Fallback handling (no Pinecone → Chroma)
- [x] No API key required ✅

---

## Phase 2: Clustering & Analytics ⏳ NOT STARTED

### Clustering Job
- [ ] HDBSCAN clustering over embeddings
  - [ ] Implement `scripts/cluster_embeddings.py`
  - [ ] Load embeddings from Pinecone
  - [ ] Run HDBSCAN (min_cluster_size=5, min_samples=3)
  - [ ] Generate cluster_id for each chunk
  - [ ] Write cluster_id back to PostgreSQL

### Cluster Labeling
- [ ] LLM-based cluster labeling
  - [ ] Implement `scripts/label_clusters.py`
  - [ ] Sample 3-5 reviews per cluster
  - [ ] Call LLM (free Groq API recommended!)
  - [ ] Generate human-readable labels ("Login bugs", "Pricing concerns", ...)
  - [ ] Save cluster_label to PostgreSQL

### Urgency Scoring
- [ ] Calculate urgency per cluster
  - [ ] Density score (cluster size / total reviews)
  - [ ] Sentiment score (% negative reviews)
  - [ ] Growth rate (velocity over time)
  - [ ] Flag urgent clusters (urgency_flag in PostgreSQL)

### Agent A: Analytics Endpoints
- [ ] FastAPI routes in `src/api/routes.py`
  - [ ] `GET /api/v1/trends?window=30d&min_growth=0.2`
    - Query: Top trending clusters by growth rate
    - Response: cluster_id, label, count, growth_rate, sentiment
  - [ ] `GET /api/v1/complaints/top?limit=10`
    - Query: Top complaint categories by count
    - Response: ranked list with severity scores
  - [ ] `GET /api/v1/dashboard/summary?window=7d`
    - Query: Weekly rollup for dashboard
    - Response: summary stats, urgent clusters, trends
  - [ ] `GET /api/v1/alerts/urgent`
    - Query: Clusters above urgency threshold
    - Response: urgent issue list with recommendations

### PostgreSQL Schema Extension
- [ ] Add cluster-related columns (already in schema)
- [ ] Create indexes on cluster queries
- [ ] Verify data types and constraints

### Testing
- [ ] Unit tests for clustering logic
- [ ] Unit tests for labeling logic
- [ ] API endpoint integration tests
- [ ] Query performance tests

---

## Phase 3: RAG & Multi-Agent Orchestration ⏳ NOT STARTED

### Retriever & Vector Search
- [ ] Implement Pinecone retriever
  - [ ] `src/rag/retriever.py`
  - [ ] Semantic search with kNN
  - [ ] Metadata filtering (user_tier, sentiment, date range)
  - [ ] Hybrid search (vector + metadata)

### Self-Correcting RAG (CRAG)
- [ ] Relevance grading
  - [ ] Implement cheap LLM judge (Groq)
  - [ ] Score retrieved chunks: relevant / ambiguous / irrelevant
  - [ ] Fallback to local classifier if needed
- [ ] Query rewriting
  - [ ] Detect ambiguous retrieval
  - [ ] Generate alternative queries
  - [ ] Re-retrieve with new queries
- [ ] Answer verification
  - [ ] Check generated answer against sources
  - [ ] Flag unsupported claims
  - [ ] Return answer only if confident

### Agent B: Q&A Tool
- [ ] Implement `src/rag/answer_question.py`
  - [ ] `POST /api/v1/ask`
  - [ ] Input: free-text question + optional filters
  - [ ] Process: retrieve → grade → [rewrite] → generate → verify
  - [ ] Output: answer + source citations (review_id) + confidence
  - [ ] Example: "What are premium users complaining about most?"

### Agent B: PRD Generation Tool
- [ ] Implement `src/rag/generate_prd.py`
  - [ ] `POST /api/v1/prd/generate`
  - [ ] Input: cluster_id or top_k clusters
  - [ ] Gather trend data from Agent A
  - [ ] Retrieve supporting reviews via RAG
  - [ ] Generate structured PRD:
    - [ ] Problem statement
    - [ ] Severity / impact analysis
    - [ ] RICE scoring (Reach, Impact, Confidence, Effort)
    - [ ] Proposed scope
    - [ ] Supporting evidence (citations)
  - [ ] Output: formatted PRD document
  - [ ] Example: "Generate PRD for Dark Mode feature"

### Suggestions Tool
- [ ] Implement feature/complaint recommendations
  - [ ] `POST /api/v1/suggestions`
  - [ ] Ranked by RICE score
  - [ ] Include supporting evidence
  - [ ] Filter by user tier / sentiment / date

### Router / Orchestrator
- [ ] Implement LangGraph orchestrator
  - [ ] `src/api/orchestrator.py`
  - [ ] Classify question intent:
    - [ ] Aggregation (→ Agent A)
    - [ ] Semantic search (→ Agent B RAG)
    - [ ] Generation (→ Agent B PRD tool)
  - [ ] Route to appropriate agent
  - [ ] Combine results if needed
- [ ] AgentA ↔ AgentB coordination
  - [ ] Agent B can call `/api/v1/trends` to fetch data
  - [ ] Avoid circular dependencies

### Testing
- [ ] Unit tests for retriever
- [ ] Unit tests for CRAG pipeline
- [ ] API endpoint tests for /ask
- [ ] API endpoint tests for /prd/generate
- [ ] Multi-agent orchestration tests

---

## Integration & Polish ⏳ NOT STARTED

### API Framework
- [ ] Set up FastAPI app
  - [ ] `src/api/main.py`
  - [ ] Route registration
  - [ ] Error handling
  - [ ] Logging
- [ ] API documentation (Swagger)
  - [ ] Request/response schemas
  - [ ] Examples

### Dashboard (Optional for v1)
- [ ] Frontend skeleton (React or Streamlit)
  - [ ] Display top complaints
  - [ ] Trend charts
  - [ ] Q&A chat interface
  - [ ] PRD viewer

### Notification Service (Optional for v1)
- [ ] Detect urgent clusters
- [ ] Alert PM via email/webhook
- [ ] Configurable thresholds

### Monitoring & Logging
- [ ] Log all queries and responses
- [ ] Track performance metrics
  - [ ] Retrieval latency
  - [ ] Generation latency
  - [ ] Query success rate
- [ ] Error tracking

---

## Testing & Validation ⏳ NOT STARTED

### Unit Tests
- [ ] `tests/test_ingestion.py` — Data loading ✅ (DONE in Phase 1)
- [ ] `tests/test_chunking.py` — Chunking logic ✅ (DONE in Phase 1)
- [ ] `tests/test_clustering.py` — HDBSCAN logic
- [ ] `tests/test_rag.py` — RAG pipeline
- [ ] `tests/test_orchestrator.py` — Agent routing

### Integration Tests
- [ ] End-to-end: ingest → cluster → query → answer
- [ ] Multi-agent flows
- [ ] Error handling & fallbacks

### Performance Tests
- [ ] Ingestion speed (reviews/sec)
- [ ] Clustering time (100 embeddings)
- [ ] Query latency (<4s for /ask)
- [ ] Throughput (concurrent requests)

### Data Quality Tests
- [ ] Vector dimension consistency
- [ ] Cluster size distribution
- [ ] Answer citation accuracy
- [ ] Label quality (manual review)

---

## Documentation & Deployment ⏳ NOT STARTED

### Code Documentation
- [ ] Docstrings for all functions
- [ ] Architecture diagrams (updated)
- [ ] API specification (OpenAPI/Swagger)
- [ ] Data model documentation

### Deployment
- [ ] Docker containerization
- [ ] Environment configuration for production
- [ ] Database migrations
- [ ] Secrets management

### README Updates
- [ ] Usage examples for each endpoint
- [ ] Troubleshooting guide
- [ ] Performance tuning tips

---

## Summary

### Completed ✅
- [x] Phase 1: Data Pipeline (100%)
- [x] Free local embeddings working (no API key needed!)
- [x] Sample test data working
- [x] End-to-end ingestion validated

### In Progress ⏳
- [ ] Phase 2: Clustering & Analytics (0%)
- [ ] Phase 3: RAG & Agents (0%)
- [ ] Testing & Validation (0%)
- [ ] Deployment (0%)

### Estimated Timeline
- **Phase 2**: 2-3 days
- **Phase 3**: 3-4 days
- **Testing & Polish**: 2-3 days
- **Total remaining**: 7-10 days

### Quick Win (Next 1-2 Days)
Implement **Phase 2a + 2b** (Clustering + Labeling):
- Add HDBSCAN clustering
- Label clusters with LLM
- Build `/api/v1/trends` and `/api/v1/complaints/top`
- This gives you a working analytics dashboard!

---

## How to Track Progress

Update this file as you go:
```bash
- [x] Task name  # Done
- [ ] Task name  # Not done
- [ ] Task name  # In progress (optional: add ⏳)
```

Commit after each phase to mark milestones! 🎯
