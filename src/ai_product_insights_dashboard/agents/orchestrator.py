"""
Multi-Agent Orchestrator using LangGraph
Routes user requests to appropriate agent (Q&A or PRD)
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not installed. Using simple router instead.")

from .qa_agent import QAAgent
from .prd_agent import PRDAgent


class AgentType(str, Enum):
    """Available agent types."""
    QA = "qa"
    PRD = "prd"
    UNKNOWN = "unknown"


class Orchestrator:
    """Route requests to appropriate agent via an explicit LangGraph graph."""

    def __init__(self):
        self.qa_agent = QAAgent()
        self.prd_agent = PRDAgent()
        self.name = "Agent Orchestrator"
        self.version = "1.0"
        self.use_langgraph = LANGGRAPH_AVAILABLE

        if self.use_langgraph:
            self._setup_langgraph()

    def _setup_langgraph(self):
        """Build the router -> {qa_handler, prd_handler} -> final graph."""
        logger.info("Setting up LangGraph orchestrator...")

        workflow = StateGraph(dict)

        workflow.add_node("router", self._route_request)
        workflow.add_node("qa_handler", self._handle_qa)
        workflow.add_node("prd_handler", self._handle_prd)
        workflow.add_node("final", self._finalize)

        workflow.add_edge(START, "router")
        workflow.add_conditional_edges(
            "router",
            self._choose_agent,
            {
                "qa": "qa_handler",
                "prd": "prd_handler",
                "unknown": "final"
            }
        )
        workflow.add_edge("qa_handler", "final")
        workflow.add_edge("prd_handler", "final")
        workflow.add_edge("final", END)

        self.graph = workflow.compile()
        logger.info("✓ LangGraph orchestrator ready")

    async def process(self, user_input: str, request_type: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process request by routing it through the compiled orchestrator graph.

        Args:
            user_input: User query or request
            request_type: Optional hint about request type ("qa" or "prd")
            metadata: Optional context

        Returns:
            Response from the routed agent
        """
        logger.info(f"Orchestrator processing: {user_input[:50]}...")

        if self.use_langgraph:
            initial_state = {
                "query": user_input,
                "request_type": request_type,
                "metadata": metadata or {},
            }
            final_state = await self.graph.ainvoke(initial_state)
            return final_state.get("result", self._unknown_result())

        # Fallback when langgraph isn't installed: same routing, no graph.
        agent_type = self._detect_agent_type(user_input) if not request_type else self._parse_agent_type(request_type)
        logger.info(f"Routed to: {agent_type.value.upper()}")

        if agent_type == AgentType.QA:
            return self.qa_agent.process(user_input, metadata)
        elif agent_type == AgentType.PRD:
            template = self._extract_prd_template(user_input)
            return await self.prd_agent.process(template, metadata)
        else:
            return self._unknown_result()

    def _unknown_result(self) -> Dict[str, Any]:
        return {
            "status": "error",
            "error": "Could not determine request type",
            "hint": "Ask a question about reviews (Q&A) or request a PRD generation"
        }

    def _detect_agent_type(self, user_input: str) -> AgentType:
        """Detect request type from input."""
        input_lower = user_input.lower()

        prd_keywords = ['prd', 'requirements', 'document', 'product requirements', 'generate doc']
        if any(kw in input_lower for kw in prd_keywords):
            return AgentType.PRD

        qa_keywords = ['?', 'what', 'how', 'why', 'which', 'tell', 'explain', 'show', 'find', 'list']
        if any(kw in input_lower for kw in qa_keywords):
            return AgentType.QA

        # Default to Q&A for ambiguous queries
        return AgentType.QA

    def _parse_agent_type(self, request_type: str) -> AgentType:
        """Parse agent type from string."""
        type_lower = request_type.lower()

        if 'qa' in type_lower or 'question' in type_lower:
            return AgentType.QA
        elif 'prd' in type_lower or 'requirements' in type_lower:
            return AgentType.PRD
        else:
            return AgentType.UNKNOWN

    def _extract_prd_template(self, user_input: str) -> str:
        """Extract PRD template preference from input."""
        input_lower = user_input.lower()

        templates = ["standard", "lean", "executive"]
        for template in templates:
            if template in input_lower:
                return template

        return "standard"  # Default

    def _route_request(self, state: Dict) -> Dict:
        """LangGraph router node: classify intent and stash it in state."""
        request_type = state.get("request_type")
        query = state.get("query", "")
        agent_type = (
            self._detect_agent_type(query) if not request_type else self._parse_agent_type(request_type)
        )
        logger.info(f"Router: {query[:50]!r} -> {agent_type.value.upper()}")
        return {**state, "agent_type": agent_type.value}

    def _choose_agent(self, state: Dict) -> str:
        """LangGraph conditional edge."""
        return state.get("agent_type", "unknown")

    def _handle_qa(self, state: Dict) -> Dict:
        """LangGraph Q&A handler node."""
        result = self.qa_agent.process(state.get("query", ""), state.get("metadata"))
        return {**state, "result": result}

    async def _handle_prd(self, state: Dict) -> Dict:
        """LangGraph PRD handler node."""
        template = self._extract_prd_template(state.get("query", ""))
        result = await self.prd_agent.process(template, state.get("metadata"))
        return {**state, "result": result}

    def _finalize(self, state: Dict) -> Dict:
        """LangGraph finalize node."""
        if "result" not in state:
            return {**state, "result": self._unknown_result()}
        return state

    def health_check(self) -> Dict[str, Any]:
        """Check orchestrator health."""
        return {
            "status": "healthy",
            "name": self.name,
            "version": self.version,
            "agents": ["qa", "prd"],
            "langgraph_enabled": self.use_langgraph,
            "timestamp": datetime.now().isoformat()
        }


# Global orchestrator instance
_orchestrator = None


def get_orchestrator() -> Orchestrator:
    """Get or create global orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


async def route_request(user_input: str, request_type: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Route user request to appropriate agent.

    Args:
        user_input: User query
        request_type: Optional type hint ("qa" or "prd")
        metadata: Optional context

    Returns:
        Agent response
    """
    orchestrator = get_orchestrator()
    return await orchestrator.process(user_input, request_type, metadata)
