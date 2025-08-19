"""
Agent Orchestrator for coordinating multi-agent operations.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from .base_agent import AgentResult, AgentTask, BaseAgent
from .executor_agent import ExecutorAgent
from .planner_agent import PlannerAgent
from .reasoner_agent import ReasonerAgent
from .research_agent import ResearchAgent


class AgentOrchestrator:
    """Orchestrates the execution of tasks across multiple agents."""

    def __init__(self):
        """Initialize the agent orchestrator."""
        self.agents = {}
        self.workflows = {}
        self.execution_history = []

        # Initialize agents
        self._initialize_agents()

        logger.info("Agent Orchestrator initialized with all agents")

    def _initialize_agents(self):
        """Initialize all agents."""
        self.agents["planner"] = PlannerAgent("planner_001")
        self.agents["research"] = ResearchAgent("research_001")
        self.agents["reasoner"] = ReasonerAgent("reasoner_001")
        self.agents["executor"] = ExecutorAgent("executor_001")

        logger.info(f"Initialized {len(self.agents)} agents")

    async def execute_workflow(
        self, query: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a complete workflow for a user query.

        Args:
            query: User's query
            context: Additional context

        Returns:
            Dict containing workflow results
        """
        workflow_id = f"workflow_{int(time.time())}"
        start_time = time.time()

        logger.info(f"Starting workflow {workflow_id} for query: {query[:50]}...")

        try:
            # Step 1: Planning - Decompose task
            planning_result = await self._execute_planning(query, context)

            # Step 2: Research - Gather information
            research_result = await self._execute_research(planning_result)

            # Step 3: Reasoning - Analyze and generate
            reasoning_result = await self._execute_reasoning(research_result)

            # Step 4: Execution - Perform side effects
            execution_result = await self._execute_side_effects(reasoning_result)

            # Compile final result
            execution_time = time.time() - start_time
            final_result = {
                "workflow_id": workflow_id,
                "query": query,
                "execution_time": execution_time,
                "stages": {
                    "planning": planning_result,
                    "research": research_result,
                    "reasoning": reasoning_result,
                    "execution": execution_result,
                },
                "final_response": (
                    reasoning_result.data.get("generated_content", "")
                    if reasoning_result.data
                    else ""
                ),
                "sources": (
                    research_result.data.get("sources", [])
                    if research_result.data
                    else []
                ),
                "metadata": {
                    "total_agents_used": len(self.agents),
                    "workflow_success": True,
                },
            }

            # Store workflow
            self.workflows[workflow_id] = final_result
            self.execution_history.append(final_result)

            logger.info(
                f"Workflow {workflow_id} completed successfully in {execution_time:.3f}s"
            )

            return final_result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Workflow {workflow_id} failed: {e}")

            return {
                "workflow_id": workflow_id,
                "query": query,
                "execution_time": execution_time,
                "error": str(e),
                "metadata": {"workflow_success": False},
            }

    async def _execute_planning(
        self, query: str, context: Dict[str, Any] = None
    ) -> AgentResult:
        """Execute planning stage."""
        planner = self.agents["planner"]

        task = AgentTask(
            task_id=f"plan_{int(time.time())}",
            task_type="task_decomposition",
            description=f"Decompose task for query: {query}",
            parameters={
                "query": query,
                "context": context or {},
                "complexity": "medium",
            },
        )

        return await planner.execute_task(task)

    async def _execute_research(self, planning_result: AgentResult) -> AgentResult:
        """Execute research stage."""
        research_agent = self.agents["research"]

        # Extract subtasks from planning result
        subtasks = planning_result.data.get("subtasks", [])
        research_subtasks = [st for st in subtasks if st["task_type"] == "research"]

        if not research_subtasks:
            # Create default research task
            research_task = AgentTask(
                task_id=f"research_{int(time.time())}",
                task_type="research",
                description="Conduct research",
                parameters={
                    "query": planning_result.data.get("original_task", ""),
                    "search_type": "vector_search",
                    "max_results": 5,
                },
            )
        else:
            # Use first research subtask
            rst = research_subtasks[0]
            research_task = AgentTask(
                task_id=rst["task_id"],
                task_type=rst["task_type"],
                description=rst["description"],
                parameters=rst["parameters"],
            )

        return await research_agent.execute_task(research_task)

    async def _execute_reasoning(self, research_result: AgentResult) -> AgentResult:
        """Execute reasoning stage."""
        reasoner = self.agents["reasoner"]

        # Create generation task based on research results
        generation_task = AgentTask(
            task_id=f"generate_{int(time.time())}",
            task_type="generation",
            description="Generate response based on research",
            parameters={
                "research_results": research_result.data,
                "response_type": "comprehensive",
                "include_citations": True,
                "max_length": 500,
            },
        )

        return await reasoner.execute_task(generation_task)

    async def _execute_side_effects(self, reasoning_result: AgentResult) -> AgentResult:
        """Execute side effects stage."""
        executor = self.agents["executor"]

        # Create execution task for logging and metrics
        execution_task = AgentTask(
            task_id=f"execute_{int(time.time())}",
            task_type="execution",
            description="Execute side effects",
            parameters={
                "operations": [
                    {
                        "type": "logging",
                        "parameters": {
                            "message": "Workflow completed successfully",
                            "log_level": "info",
                            "context": {"workflow_id": f"workflow_{int(time.time())}"},
                        },
                    },
                    {
                        "type": "metrics_collection",
                        "parameters": {
                            "metrics": {
                                "workflow_duration": reasoning_result.execution_time,
                                "agents_used": [
                                    "planner",
                                    "research",
                                    "reasoner",
                                    "executor",
                                ],
                            },
                            "storage_type": "memory",
                        },
                    },
                ]
            },
        )

        return await executor.execute_task(execution_task)

    async def execute_single_agent_task(
        self, agent_type: str, task: AgentTask
    ) -> AgentResult:
        """
        Execute a single agent task.

        Args:
            agent_type: Type of agent to use
            task: Task to execute

        Returns:
            AgentResult: Result of task execution
        """
        if agent_type not in self.agents:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent = self.agents[agent_type]

        if not agent.can_handle_task(task):
            raise ValueError(
                f"Agent {agent_type} cannot handle task type: {task.task_type}"
            )

        return await agent.execute_task(task)

    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get metrics from all agents."""
        metrics = {}

        for agent_type, agent in self.agents.items():
            metrics[agent_type] = agent.get_metrics()

        return {
            "agents": metrics,
            "total_agents": len(self.agents),
            "active_agents": sum(
                1 for agent in self.agents.values() if agent.is_active
            ),
            "total_workflows": len(self.workflows),
        }

    def get_workflow_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent workflow history."""
        return self.execution_history[-limit:] if self.execution_history else []

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get specific workflow by ID."""
        return self.workflows.get(workflow_id)

    def get_available_agents(self) -> List[str]:
        """Get list of available agent types."""
        return list(self.agents.keys())

    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all agents."""
        capabilities = {}

        for agent_type, agent in self.agents.items():
            capabilities[agent_type] = agent.capabilities

        return capabilities

    def deactivate_agent(self, agent_type: str):
        """Deactivate a specific agent."""
        if agent_type in self.agents:
            self.agents[agent_type].deactivate()
            logger.info(f"Agent {agent_type} deactivated")
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def activate_agent(self, agent_type: str):
        """Activate a specific agent."""
        if agent_type in self.agents:
            self.agents[agent_type].activate()
            logger.info(f"Agent {agent_type} activated")
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def reset_agents(self):
        """Reset all agents to initial state."""
        self._initialize_agents()
        logger.info("All agents reset to initial state")
