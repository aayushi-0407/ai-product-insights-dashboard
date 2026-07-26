"""Storage adapters for documents and vectors."""

from ai_product_insights_dashboard.storage.document_store import save_documents
from ai_product_insights_dashboard.storage.vector_store import upsert_vectors
from ai_product_insights_dashboard.storage.postgres_store import (
    init_schema,
    save_review_document,
    save_documents as save_to_postgres,
)
from ai_product_insights_dashboard.storage.pinecone_store import (
    upsert_vectors as upsert_to_pinecone,
)

__all__ = [
    "save_documents",
    "upsert_vectors",
    "init_schema",
    "save_review_document",
    "save_to_postgres",
    "upsert_to_pinecone",
]
