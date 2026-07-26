"""
Retrieve relevant chunks for user questions.
Supports both Pinecone (production) and Chroma (local) backends.
"""

import os
import json
import logging
import numpy as np
from typing import Any, List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except Exception:
    # The legacy `pinecone-client` package raises at import time. Pinecone is
    # optional; local Chroma/JSONL retrieval must still be available.
    PINECONE_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class Retriever:
    """Retrieve relevant chunks from vector database."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_pinecone: bool = True):
        """
        Initialize retriever.
        
        Args:
            model_name: Embedding model name
            use_pinecone: Prefer Pinecone over Chroma
        """
        self.model_name = model_name
        self.use_pinecone = use_pinecone

        # Initialize embeddings
        if EMBEDDINGS_AVAILABLE:
            try:
                # Keep the local demo responsive when offline. A missing model
                # falls back to JSONL retrieval below instead of attempting a
                # long network download during a chat request.
                self.embedder = SentenceTransformer(model_name, local_files_only=True)
                logger.info(f"✓ Loaded embedder: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedder: {e}")
                self.embedder = None
        else:
            logger.warning("SentenceTransformers not available")
            self.embedder = None
        
        # Initialize vector store
        self.pinecone_index = None
        self.chroma_client = None
        self._init_vector_store()
    
    def _init_vector_store(self):
        """Initialize connection to vector database."""
        if self.use_pinecone and PINECONE_AVAILABLE:
            api_key = os.getenv("PINECONE_API_KEY")
            if api_key:
                try:
                    pc = Pinecone(api_key=api_key)
                    self.pinecone_index = pc.Index("review-chunks")
                    logger.info("✓ Connected to Pinecone")
                except Exception as e:
                    logger.warning(f"Pinecone connection failed: {e}, falling back to Chroma")
                    self._init_chroma()
            else:
                self._init_chroma()
        else:
            self._init_chroma()
    
    def _init_chroma(self):
        """Initialize local Chroma database."""
        if CHROMA_AVAILABLE:
            try:
                self.chroma_client = chromadb.PersistentClient(path="data/vector_db/chroma")
                logger.info("✓ Connected to Chroma")
            except Exception as e:
                logger.warning(f"Chroma connection failed: {e}")
    
    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks for query.
        
        Args:
            query: User query
            top_k: Number of results to return
            filters: Optional metadata filters (not yet implemented)
        
        Returns:
            List of relevant chunks with scores
        """
        if not self.embedder:
            # The UI must work offline as well. If the embedding model is not
            # cached locally (or Hugging Face is unavailable), use the
            # lightweight JSONL keyword retriever instead of returning no data.
            logger.warning("Embedder unavailable; using local JSONL retrieval")
            return self._retrieve_from_jsonl(query, top_k)
        
        # Embed query
        try:
            query_embedding = self.embedder.encode(query, convert_to_numpy=True).tolist()
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return []
        
        # Retrieve from vector store
        if self.pinecone_index:
            return self._retrieve_from_pinecone(query_embedding, top_k)
        elif self.chroma_client:
            return self._retrieve_from_chroma(query_embedding, top_k)
        else:
            # Fallback: Search in local JSONL
            return self._retrieve_from_jsonl(query, top_k)
    
    def _retrieve_from_pinecone(self, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Retrieve from Pinecone."""
        try:
            results = self.pinecone_index.query(
                vector=embedding,
                top_k=top_k,
                namespace="reviews",
                include_metadata=True
            )
            
            chunks = []
            for match in results['matches']:
                chunk_data = {
                    'chunk_id': match['id'],
                    'score': match['score'],
                    'metadata': match.get('metadata', {})
                }
                chunks.append(chunk_data)
            
            logger.info(f"Retrieved {len(chunks)} chunks from Pinecone")
            return chunks
        
        except Exception as e:
            logger.warning(f"Pinecone retrieval failed: {e}")
            return self._retrieve_from_jsonl("query", top_k)
    
    def _retrieve_from_chroma(self, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Retrieve from Chroma."""
        try:
            collection = self.chroma_client.get_collection("review_chunks")
            
            results = collection.query(
                query_embeddings=[embedding],
                n_results=top_k
            )
            
            chunks = []
            for i, chunk_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to similarity (1 - distance for L2)
                similarity = 1 - (distance / 2)  # Normalize
                
                chunk_data = {
                    'chunk_id': chunk_id,
                    'text': document,
                    'score': similarity,
                    'metadata': metadata
                }
                chunks.append(chunk_data)
            
            logger.info(f"Retrieved {len(chunks)} chunks from Chroma")
            return chunks
        
        except Exception as e:
            logger.warning(f"Chroma retrieval failed: {e}")
            return self._retrieve_from_jsonl("query", top_k)
    
    def _retrieve_from_jsonl(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback: Retrieve from local JSONL file using keyword search."""
        chunks = []
        jsonl_path = Path("data/processed/review_chunks.jsonl")
        
        if not jsonl_path.exists():
            logger.warning(f"JSONL file not found: {jsonl_path}")
            return []
        
        try:
            query_words = set(query.lower().split())
            scored_chunks = []
            
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for row_number, line in enumerate(f):
                    chunk = json.loads(line)
                    text = chunk.get('text', '').lower()
                    
                    # Simple keyword overlap scoring
                    score = len(set(text.split()) & query_words) / len(query_words)
                    
                    # Boost for verified, helpful reviews
                    metadata = chunk.get('metadata', {})
                    if metadata.get('verified_purchase'):
                        score += 0.1
                    if metadata.get('helpful_vote', 0) > 5:
                        score += 0.1
                    
                    scored_chunks.append({
                        # Some demo rows share a source review id. Make the
                        # retrieval identity unique so RAG keeps every review.
                        'chunk_id': f"{chunk.get('chunk_id', 'chunk')}::{row_number}",
                        # The Amazon dataset has no review-level id, so the
                        # original review dict (used as `metadata` here) never
                        # carries one — it's only synthesized onto the chunk
                        # record itself during chunking. Surface it separately
                        # so citations can reference it.
                        'review_id': chunk.get('review_id'),
                        'text': chunk.get('text'),
                        'score': score,
                        'metadata': metadata
                    })
            
            # Sort by score and return top-k
            scored_chunks.sort(key=lambda x: x['score'], reverse=True)
            logger.info(f"Retrieved {min(top_k, len(scored_chunks))} chunks from JSONL")
            return scored_chunks[:top_k]
        
        except Exception as e:
            logger.error(f"JSONL retrieval failed: {e}")
            return []


# Module-level functions for backward compatibility
_retriever = None


def get_retriever():
    """Get or create global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def retrieve(question: str, filters: Dict[str, Any] | None = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks for a question.
    
    Args:
        question: User question
        filters: Optional metadata filters (not yet implemented)
        top_k: Number of results to return
    
    Returns:
        List of relevant chunks with metadata
    """
    retriever = get_retriever()
    return retriever.retrieve(question, top_k=top_k, filters=filters)
