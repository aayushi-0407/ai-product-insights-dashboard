"""Public entry point for the self-correcting RAG pipeline.

The actual retrieve/grade/rewrite/generate/verify logic lives in
`crag_graph.py` as an explicit LangGraph graph (PRD §9). This module just
keeps the function signature `answer_question(question, filters)` that
`qa_agent.py` and the `/api/v1/ask` route already depend on.
"""

from typing import Any, Dict, Optional

from .crag_graph import run_crag


def answer_question(question: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate a grounded, self-correcting answer from retrieved review chunks."""
    return run_crag(question, filters=filters)
