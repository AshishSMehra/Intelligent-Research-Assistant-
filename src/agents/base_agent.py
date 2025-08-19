"""
Base Agent class for the Multi-Agent Orchestration system.
"""

import hashlib
import json
import os
import random
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests
from botocore.exceptions import NoCredentialsError
from datasets import Dataset, DatasetDict
from flask import request
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, pipeline


@dataclass
class AgentTask:
    """Represents a task for an agent to execute."""

    task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any]
    priority: int = 1
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentResult:
    """Represents the result of an agent's task execution."""

    task_id: str
    agent_id: str
    success: bool
    data: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str] = None):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (planner, research, reasoner, executor)
            capabilities: List of capabilities this agent has
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities or []
        self.is_active = True
        self.task_history = []
        self.metrics = {
            "tasks_processed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
        }

        logger.info("Initialized {self.agent_type} agent: {agent_id}")

    @abstractmethod
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """
        Execute a task. Must be implemented by subclasses.

        Args:
            task: The task to execute

        Returns:
            AgentResult: Result of task execution
        """
        pass

    def can_handle_task(self, task: AgentTask) -> bool:
        """
        Check if this agent can handle the given task.

        Args:
            task: The task to check

        Returns:
            bool: True if agent can handle the task
        """
        return task.task_type in self.capabilities

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent metrics.

        Returns:
            Dict containing agent metrics
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "is_active": self.is_active,
            "capabilities": self.capabilities,
            "metrics": self.metrics.copy(),
            "task_history_count": len(self.task_history),
        }

    def _log_task_start(self, task: AgentTask):
        """Log task start."""
        logger.info(
            "Agent {self.agent_id} starting task: {task.task_id} ({task.task_type})"
        )

    def _log_task_complete(self, task: AgentTask, result: AgentResult):
        """Log task completion."""
        status = "SUCCESS" if result.success else "FAILED"
        logger.info(
            "Agent {self.agent_id} completed task {task.task_id}: {status} ({result.execution_time:.3f}s)"
        )

    def _update_metrics(self, result: AgentResult):
        """Update agent metrics."""
        self.metrics["tasks_processed"] += 1
        if result.success:
            self.metrics["tasks_succeeded"] += 1
        else:
            self.metrics["tasks_failed"] += 1
        self.metrics["total_execution_time"] += result.execution_time

    def _create_task_id(self) -> str:
        """Create a unique task ID."""
        return "{self.agent_id}_{uuid.uuid4().hex[:8]}"

    def deactivate(self):
        """Deactivate the agent."""
        self.is_active = False
        logger.info("Agent {self.agent_id} deactivated")

    def activate(self):
        """Activate the agent."""
        self.is_active = True
        logger.info("Agent {self.agent_id} activated")
