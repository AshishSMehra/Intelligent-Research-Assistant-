"""
Admin API endpoints for monitoring and debugging the multi-agent system.
"""

import hashlib
import json
import os
import random
import re
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests
from botocore.exceptions import NoCredentialsError
from datasets import Dataset, DatasetDict
from fastapi import APIRouter, Body, HTTPException, Query
from flask import request
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, pipeline

from ..agents.agent_orchestrator import AgentOrchestrator

# Create router
admin_router = APIRouter(prefix="/admin", tags=["Admin"])

# Global orchestrator instance
orchestrator = AgentOrchestrator()


@admin_router.get("/agents")
async def get_agents_status():
    """
    Get status of all agents.

    Returns:
        Dict containing agent status and metrics
    """
    try:
        metrics = orchestrator.get_agent_metrics()

        return {"status": "success", "agents": metrics}

    except Exception as e:
        logger.error("Error getting agents status: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get agents status: {str(e)}"
        )


@admin_router.get("/agents/{agent_type}")
async def get_agent_details(agent_type: str):
    """
    Get detailed information about a specific agent.

    Args:
        agent_type: Type of agent (planner, research, reasoner, executor)

    Returns:
        Dict containing agent details
    """
    try:
        if agent_type not in orchestrator.agents:
            raise HTTPException(
                status_code=404, detail="Agent type '{agent_type}' not found"
            )

        agent = orchestrator.agents[agent_type]
        details = agent.get_metrics()

        return {"status": "success", "agent_type": agent_type, "details": details}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting agent details: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get agent details: {str(e)}"
        )


@admin_router.post("/agents/{agent_type}/activate")
async def activate_agent(agent_type: str):
    """
    Activate a specific agent.

    Args:
        agent_type: Type of agent to activate

    Returns:
        Success status
    """
    try:
        orchestrator.activate_agent(agent_type)

        return {
            "status": "success",
            "message": "Agent {agent_type} activated successfully",
        }

    except Exception as e:
        logger.error("Error activating agent: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to activate agent: {str(e)}"
        )


@admin_router.post("/agents/{agent_type}/deactivate")
async def deactivate_agent(agent_type: str):
    """
    Deactivate a specific agent.

    Args:
        agent_type: Type of agent to deactivate

    Returns:
        Success status
    """
    try:
        orchestrator.deactivate_agent(agent_type)

        return {
            "status": "success",
            "message": "Agent {agent_type} deactivated successfully",
        }

    except Exception as e:
        logger.error("Error deactivating agent: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to deactivate agent: {str(e)}"
        )


@admin_router.post("/agents/reset")
async def reset_all_agents():
    """
    Reset all agents to initial state.

    Returns:
        Success status
    """
    try:
        orchestrator.reset_agents()

        return {"status": "success", "message": "All agents reset successfully"}

    except Exception as e:
        logger.error("Error resetting agents: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset agents: {str(e)}")


@admin_router.get("/workflows")
async def get_workflow_history(limit: int = Query(10, ge=1, le=100)):
    """
    Get workflow execution history.

    Args:
        limit: Maximum number of workflows to return

    Returns:
        List of recent workflows
    """
    try:
        history = orchestrator.get_workflow_history(limit)

        return {
            "status": "success",
            "workflows": history,
            "total_workflows": len(history),
        }

    except Exception as e:
        logger.error("Error getting workflow history: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get workflow history: {str(e)}"
        )


@admin_router.get("/workflows/{workflow_id}")
async def get_workflow_details(workflow_id: str):
    """
    Get detailed information about a specific workflow.

    Args:
        workflow_id: ID of the workflow

    Returns:
        Dict containing workflow details
    """
    try:
        workflow = orchestrator.get_workflow(workflow_id)

        if not workflow:
            raise HTTPException(
                status_code=404, detail="Workflow '{workflow_id}' not found"
            )

        return {"status": "success", "workflow": workflow}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting workflow details: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get workflow details: {str(e)}"
        )


@admin_router.get("/capabilities")
async def get_agent_capabilities():
    """
    Get capabilities of all agents.

    Returns:
        Dict containing agent capabilities
    """
    try:
        capabilities = orchestrator.get_agent_capabilities()

        return {"status": "success", "capabilities": capabilities}

    except Exception as e:
        logger.error("Error getting agent capabilities: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get agent capabilities: {str(e)}"
        )


@admin_router.post("/test/workflow")
async def test_workflow(query: str = Query(..., description="Test query")):
    """
    Test a complete workflow execution.

    Args:
        query: Test query to execute

    Returns:
        Workflow execution result
    """
    try:
        logger.info("Testing workflow with query: {query}")

        result = await orchestrator.execute_workflow(query)

        return {"status": "success", "test_result": result}

    except Exception as e:
        logger.error("Error testing workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to test workflow: {str(e)}")


@admin_router.post("/test/agent/{agent_type}")
async def test_single_agent(
    agent_type: str,
    task_type: str = Query(..., description="Task type to test"),
    parameters: Dict[str, Any] = Body(..., description="Task parameters"),
):
    """
    Test a single agent with a specific task.

    Args:
        agent_type: Type of agent to test
        task_type: Type of task to execute
        parameters: Task parameters

    Returns:
        Agent execution result
    """
    try:
        from ..agents.base_agent import AgentTask

        task = AgentTask(
            task_id="test_{int(time.time())}",
            task_type=task_type,
            description="Test task for {agent_type}",
            parameters=parameters,
        )

        result = await orchestrator.execute_single_agent_task(agent_type, task)

        return {
            "status": "success",
            "agent_type": agent_type,
            "task_type": task_type,
            "result": {
                "success": result.success,
                "data": result.data,
                "execution_time": result.execution_time,
                "error_message": result.error_message,
            },
        }

    except Exception as e:
        logger.error("Error testing agent: {e}")
        raise HTTPException(status_code=500, detail="Failed to test agent: {str(e)}")


@admin_router.get("/health")
async def admin_health_check():
    """
    Health check for admin endpoints.

    Returns:
        Health status
    """
    try:
        # Check if all agents are available
        available_agents = orchestrator.get_available_agents()
        agent_metrics = orchestrator.get_agent_metrics()

        health_status = {
            "status": "healthy",
            "available_agents": available_agents,
            "total_agents": len(available_agents),
            "active_agents": agent_metrics.get("active_agents", 0),
            "total_workflows": agent_metrics.get("total_workflows", 0),
        }

        return health_status

    except Exception as e:
        logger.error("Admin health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


@admin_router.get("/logs")
async def get_recent_logs(limit: int = Query(50, ge=1, le=200)):
    """
    Get recent system logs (placeholder).

    Args:
        limit: Maximum number of log entries to return

    Returns:
        Recent log entries
    """
    try:
        # Placeholder for log retrieval
        # In production, this would read from actual log files or database

        logs = [
            {
                "timestamp": "2025-08-16T12:00:00Z",
                "level": "INFO",
                "message": "Admin endpoint accessed",
                "source": "admin_api",
            }
        ]

        return {"status": "success", "logs": logs[:limit], "total_logs": len(logs)}

    except Exception as e:
        logger.error("Error getting logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get logs: {str(e)}")
