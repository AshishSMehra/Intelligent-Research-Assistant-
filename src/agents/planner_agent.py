"""
Planner Agent for task decomposition and tool selection.
"""

import time
from typing import Dict, Any, List
from loguru import logger

from .base_agent import BaseAgent, AgentTask, AgentResult


class PlannerAgent(BaseAgent):
    """Agent responsible for task decomposition and tool selection."""
    
    def __init__(self, agent_id: str = "planner_001"):
        """
        Initialize the Planner Agent.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        capabilities = [
            "task_decomposition",
            "tool_selection", 
            "workflow_planning",
            "priority_assignment"
        ]
        
        super().__init__(agent_id, "planner", capabilities)
        
        # Available tools for different task types
        self.tool_registry = {
            "search": ["vector_search", "web_search", "api_search"],
            "analysis": ["text_analysis", "data_analysis", "sentiment_analysis"],
            "generation": ["text_generation", "code_generation", "summary_generation"],
            "validation": ["fact_checking", "consistency_checking", "quality_assessment"],
            "execution": ["logging", "external_api_call", "file_operation"]
        }
        
        logger.info(f"Planner Agent {agent_id} initialized with {len(self.tool_registry)} tool categories")
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """
        Execute a planning task.
        
        Args:
            task: The task to execute
            
        Returns:
            AgentResult: Planning result with decomposed tasks
        """
        start_time = time.time()
        self._log_task_start(task)
        
        try:
            if task.task_type == "task_decomposition":
                result_data = await self._decompose_task(task)
            elif task.task_type == "tool_selection":
                result_data = await self._select_tools(task)
            elif task.task_type == "workflow_planning":
                result_data = await self._create_workflow(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            execution_time = time.time() - start_time
            result = AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                data=result_data,
                execution_time=execution_time,
                metadata={"task_type": task.task_type}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                data=None,
                error_message=str(e),
                execution_time=execution_time,
                metadata={"task_type": task.task_type}
            )
            logger.error(f"Planner Agent error: {e}")
        
        self._log_task_complete(task, result)
        self._update_metrics(result)
        self.task_history.append(result)
        
        return result
    
    async def _decompose_task(self, task: AgentTask) -> Dict[str, Any]:
        """
        Decompose a complex task into simpler subtasks.
        
        Args:
            task: The task to decompose
            
        Returns:
            Dict containing decomposed subtasks
        """
        query = task.parameters.get("query", "")
        context = task.parameters.get("context", {})
        
        # Simple task decomposition logic
        subtasks = []
        
        # Always start with research
        subtasks.append({
            "task_id": self._create_task_id(),
            "task_type": "research",
            "description": f"Research information for: {query}",
            "priority": 1,
            "parameters": {
                "query": query,
                "search_type": "vector_search",
                "max_results": 5
            }
        })
        
        # Add analysis if needed
        if len(query) > 50 or "analyze" in query.lower():
            subtasks.append({
                "task_id": self._create_task_id(),
                "task_type": "analysis",
                "description": f"Analyze research results for: {query}",
                "priority": 2,
                "dependencies": [subtasks[0]["task_id"]],
                "parameters": {
                    "analysis_type": "comprehensive",
                    "include_sentiment": True
                }
            })
        
        # Add generation
        subtasks.append({
            "task_id": self._create_task_id(),
            "task_type": "generation",
            "description": f"Generate response for: {query}",
            "priority": 3,
            "dependencies": [subtasks[0]["task_id"]],
            "parameters": {
                "response_type": "comprehensive",
                "include_citations": True,
                "max_length": 500
            }
        })
        
        # Add validation
        subtasks.append({
            "task_id": self._create_task_id(),
            "task_type": "validation",
            "description": f"Validate response for: {query}",
            "priority": 4,
            "dependencies": [subtasks[-1]["task_id"]],
            "parameters": {
                "validation_type": "fact_checking",
                "include_consistency_check": True
            }
        })
        
        return {
            "original_task": task.task_id,
            "subtasks": subtasks,
            "workflow": self._create_workflow_sequence(subtasks),
            "estimated_duration": len(subtasks) * 2.0  # Rough estimate
        }
    
    async def _select_tools(self, task: AgentTask) -> Dict[str, Any]:
        """
        Select appropriate tools for a task.
        
        Args:
            task: The task for tool selection
            
        Returns:
            Dict containing selected tools
        """
        task_type = task.parameters.get("task_type", "general")
        complexity = task.parameters.get("complexity", "medium")
        
        selected_tools = {}
        
        for category, tools in self.tool_registry.items():
            if category in task_type.lower():
                selected_tools[category] = tools
            elif complexity == "high" and category in ["analysis", "validation"]:
                selected_tools[category] = tools[:2]  # Limit for high complexity
            elif complexity == "low":
                selected_tools[category] = tools[:1]  # Minimal tools for low complexity
        
        return {
            "task_type": task_type,
            "complexity": complexity,
            "selected_tools": selected_tools,
            "tool_count": sum(len(tools) for tools in selected_tools.values())
        }
    
    async def _create_workflow(self, task: AgentTask) -> Dict[str, Any]:
        """
        Create a workflow plan for task execution.
        
        Args:
            task: The task for workflow planning
            
        Returns:
            Dict containing workflow plan
        """
        subtasks = task.parameters.get("subtasks", [])
        
        workflow = {
            "stages": [],
            "dependencies": {},
            "parallel_tasks": [],
            "critical_path": []
        }
        
        for subtask in subtasks:
            stage = {
                "stage_id": subtask["task_id"],
                "task_type": subtask["task_type"],
                "priority": subtask["priority"],
                "dependencies": subtask.get("dependencies", []),
                "estimated_duration": 2.0,
                "agent_assignment": self._assign_agent(subtask["task_type"])
            }
            
            workflow["stages"].append(stage)
            
            if stage["dependencies"]:
                workflow["dependencies"][stage["stage_id"]] = stage["dependencies"]
            else:
                workflow["parallel_tasks"].append(stage["stage_id"])
            
            if stage["priority"] == 1:
                workflow["critical_path"].append(stage["stage_id"])
        
        return workflow
    
    def _create_workflow_sequence(self, subtasks: List[Dict]) -> List[str]:
        """Create a sequence of task IDs based on dependencies."""
        sequence = []
        completed = set()
        
        while len(sequence) < len(subtasks):
            for subtask in subtasks:
                if subtask["task_id"] not in sequence:
                    dependencies = subtask.get("dependencies", [])
                    if all(dep in completed for dep in dependencies):
                        sequence.append(subtask["task_id"])
                        completed.add(subtask["task_id"])
        
        return sequence
    
    def _assign_agent(self, task_type: str) -> str:
        """Assign an agent type to a task."""
        agent_mapping = {
            "research": "research_agent",
            "analysis": "reasoner_agent", 
            "generation": "reasoner_agent",
            "validation": "reasoner_agent",
            "execution": "executor_agent"
        }
        
        return agent_mapping.get(task_type, "research_agent") 