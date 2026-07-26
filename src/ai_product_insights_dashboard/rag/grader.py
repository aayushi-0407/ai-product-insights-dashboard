"""
Relevance Grader: Scores retrieved chunks for relevance to query
Uses Groq (free LLM) for semantic relevance grading
"""

import os
import logging
import json
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class RelevanceGrader:
    """Grade relevance of retrieved chunks to a query."""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and GROQ_AVAILABLE
        if self.use_llm:
            self.groq_key = os.getenv("GROQ_API_KEY")
            self.use_llm = bool(self.groq_key)
    
    def grade_chunks(self, query: str, chunks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Grade chunks for relevance to query.
        
        Args:
            query: User's question
            chunks: Retrieved chunks with metadata
        
        Returns:
            (relevant_chunks, irrelevant_chunks)
        """
        if not chunks:
            return [], []
        
        relevant = []
        irrelevant = []
        
        if self.use_llm:
            relevant, irrelevant = self._grade_with_groq(query, chunks)
        else:
            relevant, irrelevant = self._grade_with_heuristics(query, chunks)
        
        logger.info(f"Graded {len(relevant)} relevant, {len(irrelevant)} irrelevant")
        return relevant, irrelevant
    
    def _grade_with_groq(self, query: str, chunks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Use Groq LLM to grade relevance.

        Grades all chunks in a single call instead of one call per chunk —
        with top_k=5 that was 5 sequential network round-trips just for
        grading, the single largest contributor to /ask latency. A small,
        fast model is enough for this (grading is closer to a classification
        task than an open-ended one); the larger model is reserved for the
        final answer generation, where quality actually matters most.
        """
        try:
            client = Groq(api_key=self.groq_key)

            numbered = "\n\n".join(
                f"[{i}] (rating {c.get('metadata', {}).get('rating', 'N/A')}): {c.get('text', '')[:200]}"
                for i, c in enumerate(chunks)
            )
            prompt = f"""Question: {query}

Review chunks:
{numbered}

Which chunks are relevant to answering the question? Reply with ONLY a JSON array of the chunk numbers, e.g. [0,2,3]. If none are relevant, reply []."""

            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = completion.choices[0].message.content.strip()
            raw = raw[raw.find("["):raw.rfind("]") + 1] if "[" in raw and "]" in raw else "[]"
            relevant_indices = set(json.loads(raw))

            relevant = [c for i, c in enumerate(chunks) if i in relevant_indices]
            irrelevant = [c for i, c in enumerate(chunks) if i not in relevant_indices]
            return relevant, irrelevant

        except Exception as e:
            logger.warning(f"LLM grading failed: {e}, using heuristics")
            return self._grade_with_heuristics(query, chunks)
    
    def _grade_with_heuristics(self, query: str, chunks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Fallback: Grade using keyword overlap and rating."""
        relevant = []
        irrelevant = []
        
        query_words = set(query.lower().split())
        
        for chunk in chunks:
            text = chunk.get('text', '').lower()
            metadata = chunk.get('metadata', {})
            
            # Score based on keyword overlap
            word_overlap = len(set(text.split()) & query_words) / (len(query_words) + 0.1)
            
            # Bonus for negative reviews if query contains complaint keywords
            complaint_keywords = {
                'problem', 'bug', 'issue', 'crash', 'fail', 'error', 'slow',
                'broken', 'negative', 'complaint', 'complaints', 'feedback',
                'bad', 'worst', 'unhappy', 'dissatisfied'
            }
            rating = metadata.get('rating', 3)
            has_complaint_words = any(kw in query.lower() for kw in complaint_keywords)
            
            if has_complaint_words and rating <= 2:
                word_overlap += 0.3
            elif not has_complaint_words and rating >= 4:
                word_overlap += 0.3
            
            # Threshold: > 0.2 overlap is relevant
            if word_overlap > 0.2:
                relevant.append(chunk)
            else:
                irrelevant.append(chunk)
        
        return relevant, irrelevant
    
    def get_relevance_score(self, query: str, chunk: Dict) -> float:
        """Get a relevance score (0-1) for a single chunk."""
        query_words = set(query.lower().split())
        text = chunk.get('text', '').lower()
        
        # Basic word overlap
        word_overlap = len(set(text.split()) & query_words) / (len(query_words) + 0.1)
        
        # Adjust for chunk quality metrics
        metadata = chunk.get('metadata', {})
        helpful = metadata.get('helpful_vote', 0)
        verified = metadata.get('verified_purchase', False)
        
        # Higher score for verified, helpful reviews
        score = word_overlap
        if verified:
            score += 0.1
        if helpful > 0:
            score += min(helpful / 100, 0.2)  # Max +0.2 for helpfulness
        
        return min(score, 1.0)
