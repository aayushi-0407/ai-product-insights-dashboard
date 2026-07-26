"""Q&A Agent: Answer questions about product reviews using RAG."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..rag.answer_question import answer_question

logger = logging.getLogger(__name__)


class QAAgent:
    """Agent for answering questions about product reviews."""
    
    def __init__(self):
        self.name = "Q&A Agent"
        self.version = "1.0"
    
    def process(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a question about product reviews.
        
        Args:
            query: User question
            metadata: Optional context (user_id, session_id, etc.)
        
        Returns:
            Agent response dict
        """
        logger.info(f"Q&A Agent processing: {query}")
        
        try:
            # Use RAG pipeline to answer
            result = answer_question(query)
            
            return {
                "status": "success",
                "agent": self.name,
                "query": query,
                "response": result.get("answer"),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "metadata": {
                    "total_chunks_evaluated": result.get("total_chunks_evaluated", 0),
                    "relevant_chunks": result.get("relevant_chunks_used", 0),
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        except Exception as e:
            logger.error(f"Q&A Agent error: {e}")
            return {
                "status": "error",
                "agent": self.name,
                "query": query,
                "response": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def validate_query(self, query: str) -> bool:
        """Check if query is valid and non-empty."""
        return query and len(query.strip()) > 0


def qa_agent():
    """Factory function for Q&A agent."""
    return QAAgent()
