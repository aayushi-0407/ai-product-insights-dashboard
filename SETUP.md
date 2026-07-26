# Setup Guide: Amazon Reviews Pipeline with OpenAI + Pinecone

This guide walks you through setting up the complete ingestion pipeline to load Amazon product reviews from Hugging Face, embed them with OpenAI's `text-embedding-3-small`, and store them in Pinecone.

## Prerequisites

1. **Python 3.10+** installed
2. **API Keys:**
   - OpenAI API key (for embeddings)
   - Pinecone API key (for vector storage)
   - PostgreSQL database (optional, for metadata)

## Step 1: Install Dependencies

```bash
cd "c:\Users\Arsh\Desktop\AI Dashboard"
pip install -e .
```

This installs the project in editable mode with all dependencies:
- `pinecone-client` (vector database)
- `openai` (embeddings)
- `psycopg2-binary` (PostgreSQL driver)
- `sqlalchemy` (ORM)
- `huggingface-hub` (dataset access)
- And more...

## Step 2: Configure Environment Variables

Create a `.env` file in the project root with your API keys:

```bash
# .env
OPENAI_API_KEY=sk-proj-xxxxx...
PINECONE_API_KEY=pcsk_xxxxx...
DATABASE_URL=postgresql://user:password@localhost:5432/reviews_db
```

Or set them in your shell:

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY = "sk-proj-xxxxx..."
$env:PINECONE_API_KEY = "pcsk_xxxxx..."
$env:DATABASE_URL = "postgresql://user:password@localhost:5432/reviews_db"
```

**Windows CMD:**
```cmd
set OPENAI_API_KEY=sk-proj-xxxxx...
set PINECONE_API_KEY=pcsk_xxxxx...
set DATABASE_URL=postgresql://user:password@localhost:5432/reviews_db
```

### Getting API Keys

**OpenAI:**
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy it to `.env`

**Pinecone:**
1. Go to https://app.pinecone.io/
2. Create a free account (1 index included)
3. Create an index named `review-chunks` with:
   - Dimension: `1536` (for text-embedding-3-small)
   - Metric: `cosine`
4. Copy your API key to `.env`

**PostgreSQL (Optional):**
- Local: `postgresql://postgres:password@localhost:5432/reviews_db`
- Cloud: Use a managed service like Render, Railway, or AWS RDS

## Step 3: (Optional) Set Up PostgreSQL

If you want to store metadata in PostgreSQL:

### Local PostgreSQL
```bash
# On Windows, install PostgreSQL or use WSL2
# Create database and user:
createdb reviews_db
psql reviews_db -c "CREATE USER reviews_user WITH PASSWORD 'your_password';"
psql reviews_db -c "ALTER USER reviews_user CREATEDB;"
```

Or use Docker:
```bash
docker run --name reviews_postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=reviews_db \
  -p 5432:5432 \
  -d postgres:15
```

Then set `DATABASE_URL`:
```
DATABASE_URL=postgresql://postgres:password@localhost:5432/reviews_db
```

## Step 4: Run the Ingestion Pipeline

### Full Pipeline (Hugging Face → OpenAI → Pinecone + PostgreSQL)

```bash
python scripts/ingest_reviews.py \
  --dataset McAuley-Lab/Amazon-Reviews-2023 \
  --config-name raw_review_Software \
  --limit 1000 \
  --init-db \
  --embedding-model text-embedding-3-small \
  --embedding-dimension 1536
```

**Arguments:**
- `--dataset`: Hugging Face dataset ID (default: `McAuley-Lab/Amazon-Reviews-2023`)
- `--config-name`: Dataset split (default: `raw_review_Software`)
- `--limit`: Number of reviews to fetch (default: all; use for testing)
- `--init-db`: Initialize PostgreSQL schema (run once)
- `--embedding-model`: OpenAI model (default: `text-embedding-3-small`)
- `--embedding-dimension`: Embedding dimensions (default: `1536`)
- `--skip-vector-db`: Skip Pinecone upload (useful for testing)

### Example: Test with 100 Reviews

```bash
python scripts/ingest_reviews.py --limit 100 --init-db
```

**Output:**
```
Fetching reviews from McAuley-Lab/Amazon-Reviews-2023:raw_review_Software
  Limit: 100
✓ Fetched 100 reviews
✓ Saved raw reviews -> data/raw/reviews.jsonl
Chunking reviews...
✓ Created 105 chunks
Embedding chunks with text-embedding-3-small...
✓ Embedded 105 chunks
✓ Saved embedded chunks -> data/processed/review_chunks.jsonl
Saving to PostgreSQL...
✓ Saved 105 documents to PostgreSQL
Upserting to vector database...
✓ Upserted 105 vectors
✓ Ingestion pipeline complete!
```

## Step 5: Verify Setup

### Check Pinecone

```python
from pinecone import Pinecone

pc = Pinecone()
index = pc.Index("review-chunks")
stats = index.describe_index_stats()
print(f"Vectors in Pinecone: {stats['namespaces']['reviews']['vector_count']}")
```

### Check PostgreSQL

```bash
psql reviews_db -c "SELECT COUNT(*) FROM reviews;"
```

## Dataset Info

**Amazon Reviews 2023 — Software category**

- **Source**: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- **Fields per review**:
  - `asin`: Product ID
  - `rating`: 1–5 stars
  - `text`: Review text
  - `timestamp`: Unix timestamp (seconds)
  - `verified_purchase`: Boolean
  - `helpful_vote`: Count
  - `user_id`: Reviewer ID

## Pipeline Architecture

```
Hugging Face Dataset (raw_review_Software)
           ↓
[fetch_reviews.py] → Stream JSONL from HF
           ↓
[chunk_reviews.py] → Split into ~120-token chunks with overlap
           ↓
[embed_reviews.py] → OpenAI text-embedding-3-small (1536-dim vectors)
           ↓
        ┌──────────────┬──────────────┐
        ↓              ↓              ↓
   [Pinecone]   [PostgreSQL]   [Local JSONL]
 (vector search) (metadata)    (archive)
```

## Troubleshooting

### OpenAI API Error: "No API key provided"
```bash
echo $env:OPENAI_API_KEY  # Check it's set
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

### Pinecone: "Index not found"
1. Go to https://app.pinecone.io/
2. Create an index named `review-chunks` (dimension: 1536, metric: cosine)
3. Verify `PINECONE_API_KEY` is correct

### PostgreSQL: "connection refused"
- Verify PostgreSQL is running
- Check `DATABASE_URL` format
- Ensure database user has permissions

### Rate limiting on OpenAI
- The pipeline batches requests (default: 100 chunks/batch)
- For large datasets, use `--limit` to test incrementally
- Check your OpenAI quota at https://platform.openai.com/account/rate-limits

## Next Steps

1. ✅ Run ingestion with `--limit 100` to test
2. ✅ Verify vectors in Pinecone
3. ✅ Build clustering job (HDBSCAN on embeddings)
4. ✅ Implement Agent A (analytics queries on PostgreSQL)
5. ✅ Implement Agent B (RAG with self-correction)

## Cost Estimate

- **OpenAI embeddings**: ~$0.02 per 1M tokens (text-embedding-3-small)
  - For 10K reviews (~1M tokens): ~$0.02
- **Pinecone**: Free tier (1 index, up to 100K vectors)
- **PostgreSQL**: Free tier or $5-15/month on managed services
