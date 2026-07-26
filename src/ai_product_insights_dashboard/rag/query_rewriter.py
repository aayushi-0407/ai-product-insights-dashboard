"""
Query Rewriter: Expands and rewrites queries for better retrieval
Uses Groq (free LLM) to generate alternative queries and keywords
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


def rewrite_query(query: str, use_groq: bool = True) -> List[str]:
    """
    Generate alternative query formulations for better retrieval.
    
    Args:
        query: Original user query
        use_groq: Whether to use Groq LLM for rewriting (fallback to keyword expansion)
    
    Returns:
        List of query variations (original + rewrites)
    """
    queries = [query]  # Always include original
    
    if use_groq and GROQ_AVAILABLE:
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            queries.extend(_rewrite_with_groq(query, groq_key))
        else:
            queries.extend(_expand_keywords(query))
    else:
        queries.extend(_expand_keywords(query))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        if q.lower() not in seen:
            unique_queries.append(q)
            seen.add(q.lower())
    
    return unique_queries


def _rewrite_with_groq(query: str, api_key: str) -> List[str]:
    """Use Groq LLM to generate query variations."""
    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""Generate 2-3 alternative ways to phrase this customer review question for better search results. Return ONLY the alternative queries, one per line, no numbering or explanation.

Original query: {query}

Alternative queries:"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )

        rewrites = response.choices[0].message.content.strip().split('\n')
        rewrites = [r.strip() for r in rewrites if r.strip()]
        
        logger.info(f"Generated {len(rewrites)} query rewrites")
        return rewrites[:3]  # Limit to 3 rewrites
        
    except Exception as e:
        logger.warning(f"Groq rewriting failed: {e}, falling back to keyword expansion")
        return _expand_keywords(query)


def _expand_keywords(query: str) -> List[str]:
    """Simple keyword-based query expansion (fallback)."""
    rewrites = []
    
    # Synonym mappings for common review terms
    synonyms = {
        'bug': ['issue', 'problem', 'crash'],
        'crash': ['freeze', 'hang', 'fail'],
        'slow': ['lag', 'performance', 'speed'],
        'fast': ['quick', 'responsive', 'efficient'],
        'interface': ['UI', 'design', 'usability'],
        'feature': ['functionality', 'capability', 'option'],
        'install': ['setup', 'installation', 'deploy'],
    }
    
    # Generate variations by replacing synonyms
    query_lower = query.lower()
    for original, replacement_list in synonyms.items():
        if original in query_lower:
            for replacement in replacement_list[:2]:  # Limit to 2 per synonym
                rewrite = query_lower.replace(original, replacement)
                if rewrite not in [q.lower() for q in rewrites]:
                    rewrites.append(rewrite.capitalize())
    
    logger.info(f"Generated {len(rewrites)} keyword-based query rewrites")
    return rewrites[:3]
