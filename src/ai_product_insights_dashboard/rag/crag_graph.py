"""Corrective RAG (CRAG) as an explicit LangGraph graph.

PRD §9: retrieve -> grade -> [rewrite -> retrieve] -> generate -> verify,
with the grade/verify steps able to loop back through a query rewrite
instead of generating over irrelevant chunks or returning an unverified
answer. This module is the graph; `answer_question.py` is a thin wrapper
that keeps the existing module-level function signature callers depend on.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .grader import RelevanceGrader
from .query_rewriter import rewrite_query
from .retriever import retrieve as retrieve_chunks

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

DEFAULT_TOP_K = 5
MAX_ITERATIONS = 2
GENERATION_MODEL = "llama-3.3-70b-versatile"


class CRAGState(TypedDict, total=False):
    question: str
    filters: Optional[Dict[str, Any]]
    query: str
    top_k: int
    iteration: int
    max_iterations: int
    chunks: List[Dict[str, Any]]
    relevant_chunks: List[Dict[str, Any]]
    irrelevant_chunks: List[Dict[str, Any]]
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    is_valid: bool
    issues: str
    self_correction_applied: bool
    correction_reason: Optional[str]
    total_chunks_evaluated: int


def _node_retrieve(state: CRAGState) -> Dict[str, Any]:
    query = state.get("query") or state["question"]
    top_k = state.get("top_k", DEFAULT_TOP_K)
    new_chunks = retrieve_chunks(query, filters=state.get("filters"), top_k=top_k)

    existing = state.get("chunks", [])
    seen = {c.get("chunk_id") for c in existing}
    merged = existing + [c for c in new_chunks if c.get("chunk_id") not in seen]

    logger.info(f"[retrieve] query={query!r} top_k={top_k} -> {len(new_chunks)} new, {len(merged)} total")
    return {"chunks": merged, "total_chunks_evaluated": len(merged)}


def _node_grade(state: CRAGState) -> Dict[str, Any]:
    grader = RelevanceGrader(use_llm=GROQ_AVAILABLE and bool(os.getenv("GROQ_API_KEY")))
    relevant, irrelevant = grader.grade_chunks(state["question"], state.get("chunks", []))
    logger.info(f"[grade] {len(relevant)} relevant, {len(irrelevant)} irrelevant")
    return {"relevant_chunks": relevant, "irrelevant_chunks": irrelevant}


def _edge_after_grade(state: CRAGState) -> str:
    if state.get("relevant_chunks"):
        return "generate"
    if state.get("iteration", 0) < state.get("max_iterations", MAX_ITERATIONS):
        return "rewrite"
    return "insufficient"


def _node_rewrite(state: CRAGState) -> Dict[str, Any]:
    """Widen the search: try a rewritten query and pull more chunks per retry."""
    rewrites = rewrite_query(state["question"])
    iteration = state.get("iteration", 0) + 1
    next_query = rewrites[iteration] if iteration < len(rewrites) else state["question"]

    logger.info(f"[rewrite] attempt {iteration}: {next_query!r}")
    return {
        "query": next_query,
        "iteration": iteration,
        "top_k": state.get("top_k", DEFAULT_TOP_K) + 5,
        "self_correction_applied": True,
        "correction_reason": (
            f"Initial retrieval had low relevance; retried with a rewritten "
            f"query and a wider top_k (attempt {iteration})"
        ),
    }


def _fallback_answer(question: str, chunks: List[Dict[str, Any]]) -> tuple[str, float]:
    """Heuristic answer when no Groq key is configured."""
    avg_rating = sum(c.get("metadata", {}).get("rating", 3) for c in chunks) / len(chunks)
    review_texts = [c.get("text", "")[:100] for c in chunks[:3]]

    question_lower = question.lower()
    parts = []
    if any(kw in question_lower for kw in ("problem", "bug", "issue")):
        low_rating_count = sum(1 for c in chunks if c.get("metadata", {}).get("rating", 3) <= 2)
        parts.append(
            "Yes, customers report significant issues with this product."
            if low_rating_count > len(chunks) * 0.5
            else "There are some reported issues, but they're not widespread."
        )
    elif any(kw in question_lower for kw in ("good", "like", "best")):
        high_rating_count = sum(1 for c in chunks if c.get("metadata", {}).get("rating", 3) >= 4)
        parts.append(
            "Yes, customers generally have positive feedback."
            if high_rating_count > len(chunks) * 0.5
            else "Opinions are mixed among customers."
        )

    if parts:
        parts.append(f"Average rating: {avg_rating:.1f}/5. Key mentions: {'; '.join(review_texts[:2])}")
    else:
        parts.append(f"Based on {len(chunks)} reviews with average rating {avg_rating:.1f}/5: {review_texts[0]}")

    return " ".join(parts), 0.6


def _node_generate(state: CRAGState) -> Dict[str, Any]:
    chunks = state.get("relevant_chunks", [])[:10]
    if not chunks:
        return {"answer": "I couldn't find specific information in the reviews to answer that question.",
                 "confidence": 0.0, "sources": []}

    groq_key = os.getenv("GROQ_API_KEY")
    if GROQ_AVAILABLE and groq_key:
        try:
            client = Groq(api_key=groq_key)
            context = "\n\n".join(
                f"[{c.get('chunk_id')}] (rating {c.get('metadata', {}).get('rating', 'N/A')}): {c.get('text', '')}"
                for c in chunks
            )
            prompt = f"""Answer the question using ONLY the customer review excerpts below. Cite the bracketed id of every review you rely on. If the reviews don't support an answer, say so plainly instead of guessing.

Reviews:
{context}

Question: {state['question']}

Answer (2-4 sentences, with [chunk_id] citations):"""
            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content.strip()
            confidence = 0.85
        except Exception as e:
            logger.warning(f"LLM answer generation failed: {e}")
            answer, confidence = _fallback_answer(state["question"], chunks)
    else:
        answer, confidence = _fallback_answer(state["question"], chunks)

    grader = RelevanceGrader(use_llm=False)
    sources = [
        {
            "review_id": (c.get("metadata", {}) or {}).get("review_id") or c.get("review_id"),
            "chunk_id": c.get("chunk_id"),
            "text": (c.get("text") or "")[:200],
            "relevance": grader.get_relevance_score(state["question"], c),
        }
        for c in chunks[:3]
    ]
    return {"answer": answer, "confidence": confidence, "sources": sources}


def _node_verify(state: CRAGState) -> Dict[str, Any]:
    """Check the generated answer is grounded in retrieved evidence before returning it."""
    answer = state.get("answer", "")
    chunks = state.get("relevant_chunks", [])
    issues = []

    if len(answer.split()) < 5:
        issues.append("answer too short")

    if chunks:
        cited = any((c.get("chunk_id") or "") in answer for c in chunks)
        mentions_reviews = any(w in answer.lower() for w in ("review", "rating", "customer", "feedback"))
        if not cited and not mentions_reviews:
            issues.append("answer does not reference retrieved review content")

    is_valid = not issues
    logger.info(f"[verify] valid={is_valid} issues={issues or 'none'}")
    return {"is_valid": is_valid, "issues": "; ".join(issues) or "OK"}


def _edge_after_verify(state: CRAGState) -> str:
    if state.get("is_valid", True) or state.get("iteration", 0) >= state.get("max_iterations", MAX_ITERATIONS):
        return "done"
    return "retry"


def _node_insufficient(state: CRAGState) -> Dict[str, Any]:
    return {
        "answer": "I couldn't find enough relevant reviews to answer that confidently. Try rephrasing the question.",
        "confidence": 0.0,
        "sources": [],
        "self_correction_applied": True,
        "correction_reason": "No relevant chunks found after query rewriting and widened retrieval.",
    }


def _build_graph():
    graph = StateGraph(CRAGState)
    graph.add_node("retrieve", _node_retrieve)
    graph.add_node("grade", _node_grade)
    graph.add_node("rewrite", _node_rewrite)
    graph.add_node("generate", _node_generate)
    graph.add_node("verify", _node_verify)
    graph.add_node("insufficient", _node_insufficient)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", _edge_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "insufficient": "insufficient"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", _edge_after_verify, {"done": END, "retry": "rewrite"})
    graph.add_edge("insufficient", END)

    return graph.compile()


_compiled_graph = _build_graph()


def run_crag(
    question: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = DEFAULT_TOP_K,
    max_iterations: int = MAX_ITERATIONS,
) -> Dict[str, Any]:
    """Run the self-correcting RAG graph end to end for one question."""
    initial_state: CRAGState = {
        "question": question,
        "filters": filters,
        "query": question,
        "top_k": top_k,
        "iteration": 0,
        "max_iterations": max_iterations,
        "chunks": [],
        "relevant_chunks": [],
        "irrelevant_chunks": [],
        "answer": "",
        "confidence": 0.0,
        "sources": [],
        "is_valid": True,
        "issues": "OK",
        "self_correction_applied": False,
        "correction_reason": None,
        "total_chunks_evaluated": 0,
    }

    final_state = _compiled_graph.invoke(initial_state)

    return {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "confidence": final_state.get("confidence", 0.0),
        "total_chunks_evaluated": final_state.get("total_chunks_evaluated", 0),
        "relevant_chunks_used": len(final_state.get("relevant_chunks", [])),
        "iterations": final_state.get("iteration", 0) + 1,
        "self_correction_applied": final_state.get("self_correction_applied", False),
        "correction_reason": final_state.get("correction_reason"),
    }
