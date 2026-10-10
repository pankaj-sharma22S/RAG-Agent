"""Agents package for RAG-Agent and GitHub Analyzer Agent."""

from .schemas import (
    CodeStructureAnalysis,
    DiagnosticHypothesis,
    DocumentationComparison,
    Finding,
    InvestigationPlan,
    StructuredAnalysisReport,
)
from .github_analyzer_agent import analyze_github_repository, create_github_analyzer_graph

try:
    from app.agent.graph import rag_agent_app, AgentState, workflow
except ImportError:
    try:
        from agent.graph import rag_agent_app, AgentState, workflow
    except ImportError:
        rag_agent_app = None
        AgentState = None
        workflow = None

__all__ = [
    "CodeStructureAnalysis",
    "DiagnosticHypothesis",
    "DocumentationComparison",
    "Finding",
    "InvestigationPlan",
    "StructuredAnalysisReport",
    "analyze_github_repository",
    "create_github_analyzer_graph",
    "rag_agent_app",
    "AgentState",
    "workflow",
]


