"""
Executor Agent for side effects and external operations.
"""

import time
import json
import asyncio
from typing import Dict, Any, List
from loguru import logger

from .base_agent import BaseAgent, AgentTask, AgentResult


class ExecutorAgent(BaseAgent):
    """Agent responsible for side effects and external operations."""
    
    def __init__(self, agent_id: str = "executor_001"):
        """
        Initialize the Executor Agent.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        capabilities = [
            "logging",
            "external_api_call",
            "file_operation",
            "database_operation",
            "notification_sending",
            "metrics_collection"
        ]
        
        super().__init__(agent_id, "executor", capabilities)
        
        # Execution methods
        self.execution_methods = {
            "logging": self._perform_logging,
            "external_api_call": self._call_external_api,
            "file_operation": self._perform_file_operation,
            "database_operation": self._perform_database_operation,
            "notification_sending": self._send_notification,
            "metrics_collection": self._collect_metrics
        }
        
        # External API configurations
        self.api_configs = {
            "weather": {"url": "https://api.weatherapi.com/v1/current.json", "method": "GET"},
            "news": {"url": "https://newsapi.org/v2/top-headlines", "method": "GET"},
            "translation": {"url": "https://translation-api.example.com/translate", "method": "POST"}
        }
        
        logger.info(f"Executor Agent {agent_id} initialized with {len(self.execution_methods)} execution methods")
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """
        Execute a task with side effects.
        
        Args:
            task: The task to execute
            
        Returns:
            AgentResult: Execution result
        """
        start_time = time.time()
        self._log_task_start(task)
        
        try:
            task_type = task.task_type
            
            if task_type in self.execution_methods:
                method = self.execution_methods[task_type]
                result_data = await method(task)
            elif task_type == "execution":
                result_data = await self._execute_side_effects(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
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
            logger.error(f"Executor Agent error: {e}")
        
        self._log_task_complete(task, result)
        self._update_metrics(result)
        self.task_history.append(result)
        
        return result
    
    async def _execute_side_effects(self, task: AgentTask) -> Dict[str, Any]:
        """
        Execute multiple side effects.
        
        Args:
            task: The execution task
            
        Returns:
            Dict containing execution results
        """
        operations = task.parameters.get("operations", [])
        
        results = {
            "operations_executed": [],
            "success_count": 0,
            "failure_count": 0,
            "total_operations": len(operations)
        }
        
        for operation in operations:
            try:
                op_type = operation.get("type")
                op_params = operation.get("parameters", {})
                
                if op_type in self.execution_methods:
                    method = self.execution_methods[op_type]
                    op_task = AgentTask(
                        task_id=self._create_task_id(),
                        task_type=op_type,
                        description=f"Execute {op_type} operation",
                        parameters=op_params
                    )
                    
                    op_result = await method(op_task)
                    results["operations_executed"].append({
                        "type": op_type,
                        "success": True,
                        "result": op_result
                    })
                    results["success_count"] += 1
                else:
                    results["operations_executed"].append({
                        "type": op_type,
                        "success": False,
                        "error": f"Unknown operation type: {op_type}"
                    })
                    results["failure_count"] += 1
                    
            except Exception as e:
                results["operations_executed"].append({
                    "type": operation.get("type", "unknown"),
                    "success": False,
                    "error": str(e)
                })
                results["failure_count"] += 1
        
        return results
    
    async def _perform_logging(self, task: AgentTask) -> Dict[str, Any]:
        """
        Perform logging operations.
        
        Args:
            task: The logging task
            
        Returns:
            Dict containing logging results
        """
        message = task.parameters.get("message", "")
        log_level = task.parameters.get("log_level", "info")
        context = task.parameters.get("context", {})
        
        # Perform logging based on level
        if log_level == "debug":
            logger.debug(f"Executor Log: {message}", extra=context)
        elif log_level == "info":
            logger.info(f"Executor Log: {message}", extra=context)
        elif log_level == "warning":
            logger.warning(f"Executor Log: {message}", extra=context)
        elif log_level == "error":
            logger.error(f"Executor Log: {message}", extra=context)
        else:
            logger.info(f"Executor Log: {message}", extra=context)
        
        return {
            "logged_message": message,
            "log_level": log_level,
            "timestamp": time.time(),
            "context": context
        }
    
    async def _call_external_api(self, task: AgentTask) -> Dict[str, Any]:
        """
        Call external APIs.
        
        Args:
            task: The API call task
            
        Returns:
            Dict containing API call results
        """
        api_type = task.parameters.get("api_type", "general")
        endpoint = task.parameters.get("endpoint", "")
        method = task.parameters.get("method", "GET")
        data = task.parameters.get("data", {})
        headers = task.parameters.get("headers", {})
        
        # Simulate API call (placeholder implementation)
        # In production, use proper HTTP client like aiohttp or httpx
        
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Mock response based on API type
        if api_type == "weather":
            response_data = {
                "location": {"name": "New York", "country": "USA"},
                "current": {"temp_c": 22, "condition": {"text": "Sunny"}}
            }
        elif api_type == "news":
            response_data = {
                "status": "ok",
                "totalResults": 1,
                "articles": [{"title": "Sample News", "description": "Sample description"}]
            }
        elif api_type == "translation":
            response_data = {
                "translated_text": f"Translated: {data.get('text', '')}",
                "source_language": "en",
                "target_language": "es"
            }
        else:
            response_data = {"status": "success", "data": "API call completed"}
        
        return {
            "api_type": api_type,
            "endpoint": endpoint,
            "method": method,
            "request_data": data,
            "response_data": response_data,
            "status_code": 200,
            "response_time": 0.1
        }
    
    async def _perform_file_operation(self, task: AgentTask) -> Dict[str, Any]:
        """
        Perform file operations.
        
        Args:
            task: The file operation task
            
        Returns:
            Dict containing file operation results
        """
        operation = task.parameters.get("operation", "read")
        file_path = task.parameters.get("file_path", "")
        content = task.parameters.get("content", "")
        
        result = {
            "operation": operation,
            "file_path": file_path,
            "success": True
        }
        
        try:
            if operation == "read":
                # Simulate file read
                result["content"] = f"File content from {file_path}"
                result["file_size"] = len(result["content"])
                
            elif operation == "write":
                # Simulate file write
                result["bytes_written"] = len(content)
                result["message"] = f"Successfully wrote {len(content)} bytes to {file_path}"
                
            elif operation == "append":
                # Simulate file append
                result["bytes_appended"] = len(content)
                result["message"] = f"Successfully appended {len(content)} bytes to {file_path}"
                
            elif operation == "delete":
                # Simulate file delete
                result["message"] = f"Successfully deleted {file_path}"
                
            else:
                result["success"] = False
                result["error"] = f"Unknown file operation: {operation}"
                
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    async def _perform_database_operation(self, task: AgentTask) -> Dict[str, Any]:
        """
        Perform database operations.
        
        Args:
            task: The database operation task
            
        Returns:
            Dict containing database operation results
        """
        operation = task.parameters.get("operation", "query")
        table = task.parameters.get("table", "")
        query = task.parameters.get("query", "")
        data = task.parameters.get("data", {})
        
        result = {
            "operation": operation,
            "table": table,
            "success": True
        }
        
        try:
            if operation == "insert":
                result["rows_affected"] = 1
                result["inserted_id"] = "mock_id_123"
                result["message"] = f"Successfully inserted data into {table}"
                
            elif operation == "update":
                result["rows_affected"] = 1
                result["message"] = f"Successfully updated {table}"
                
            elif operation == "delete":
                result["rows_affected"] = 1
                result["message"] = f"Successfully deleted from {table}"
                
            elif operation == "query":
                result["rows_returned"] = 5
                result["data"] = [
                    {"id": 1, "name": "Sample 1"},
                    {"id": 2, "name": "Sample 2"},
                    {"id": 3, "name": "Sample 3"}
                ]
                result["message"] = f"Successfully queried {table}"
                
            else:
                result["success"] = False
                result["error"] = f"Unknown database operation: {operation}"
                
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    async def _send_notification(self, task: AgentTask) -> Dict[str, Any]:
        """
        Send notifications.
        
        Args:
            task: The notification task
            
        Returns:
            Dict containing notification results
        """
        notification_type = task.parameters.get("type", "email")
        recipient = task.parameters.get("recipient", "")
        subject = task.parameters.get("subject", "")
        message = task.parameters.get("message", "")
        
        # Simulate notification sending
        await asyncio.sleep(0.05)  # Simulate network delay
        
        result = {
            "notification_type": notification_type,
            "recipient": recipient,
            "subject": subject,
            "message": message,
            "sent_at": time.time(),
            "status": "sent",
            "notification_id": f"notif_{int(time.time())}"
        }
        
        # Log the notification
        logger.info(f"Notification sent: {notification_type} to {recipient}")
        
        return result
    
    async def _collect_metrics(self, task: AgentTask) -> Dict[str, Any]:
        """
        Collect and store metrics.
        
        Args:
            task: The metrics collection task
            
        Returns:
            Dict containing metrics collection results
        """
        metrics_data = task.parameters.get("metrics", {})
        storage_type = task.parameters.get("storage_type", "memory")
        
        # Simulate metrics storage
        stored_metrics = {
            "timestamp": time.time(),
            "metrics": metrics_data,
            "storage_type": storage_type,
            "metrics_count": len(metrics_data)
        }
        
        # Log metrics collection
        logger.info(f"Metrics collected: {len(metrics_data)} metrics stored in {storage_type}")
        
        return {
            "stored_metrics": stored_metrics,
            "storage_success": True,
            "metrics_processed": len(metrics_data)
        } 