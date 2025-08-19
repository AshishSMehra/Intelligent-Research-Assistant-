"""
Multi-Agent Orchestration for the Intelligent Research Assistant.

This module implements a modular agent system with:
- Planner Agent: Task decomposition and tool selection
- Research Agent: Information retrieval and API calls
- Reasoner Agent: Validation and follow-up requests
- Executor Agent: Side effects and external operations
"""

from .agent_orchestrator import AgentOrchestrator
from .base_agent import BaseAgent
from .executor_agent import ExecutorAgent
from .planner_agent import PlannerAgent
from .reasoner_agent import ReasonerAgent
from .research_agent import ResearchAgent

__all__ = [
    "BaseAgent",
    "PlannerAgent",
    "ResearchAgent",
    "ReasonerAgent",
    "ExecutorAgent",
    "AgentOrchestrator",
]
