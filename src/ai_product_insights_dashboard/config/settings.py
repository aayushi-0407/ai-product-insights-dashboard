"""Application settings for data ingestion and RAG services."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    project_name: str = "ai_product_insights_dashboard"
    raw_data_dir: str = "data/raw"
    interim_data_dir: str = "data/interim"
    processed_data_dir: str = "data/processed"
    
    # Embeddings (default: free local model)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "local")  # "local" (free) or "openai"
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))  # 384 for local, 1536 for OpenAI
    
    # OpenAI (optional, for OpenAI embeddings)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = "review-chunks"
    pinecone_namespace: str = "reviews"
    
    # PostgreSQL
    database_url: str = os.getenv("DATABASE_URL", "")
    
    # Chunking
    chunk_size: int = 300
    chunk_overlap: int = 30


settings = Settings()
