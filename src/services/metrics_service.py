"""
Metrics Service for the Intelligent Research Assistant.

This service handles comprehensive logging, metrics collection, and monitoring.
"""

import time
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict
from loguru import logger


class MetricsService:
    """Service for collecting and managing application metrics."""
    
    def __init__(self):
        """Initialize the metrics service."""
        self.metrics = {
            "requests": defaultdict(int),
            "response_times": defaultdict(list),
            "errors": defaultdict(int),
            "token_usage": defaultdict(int),
            "embedding_generations": defaultdict(int),
            "search_operations": defaultdict(int),
            "llm_calls": defaultdict(int)
        }
        
        self.session_metrics = defaultdict(lambda: {
            "queries": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "errors": 0
        })
        
        logger.info("Metrics Service initialized")
    
    def log_request(self, endpoint: str, method: str, user_id: Optional[str] = None) -> str:
        """
        Log an incoming request.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            user_id: Optional user identifier
            
        Returns:
            Request ID for tracking
        """
        request_id = f"req_{int(time.time() * 1000)}"
        
        self.metrics["requests"][f"{method}_{endpoint}"] += 1
        
        logger.info(f"Request {request_id}: {method} {endpoint} (User: {user_id or 'anonymous'})")
        
        return request_id
    
    def log_response_time(self, endpoint: str, method: str, duration: float) -> None:
        """
        Log response time for an endpoint.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            duration: Response time in seconds
        """
        key = f"{method}_{endpoint}"
        self.metrics["response_times"][key].append(duration)
        
        # Keep only last 100 measurements
        if len(self.metrics["response_times"][key]) > 100:
            self.metrics["response_times"][key] = self.metrics["response_times"][key][-100:]
        
        logger.debug(f"Response time for {key}: {duration:.3f}s")
    
    def log_error(self, endpoint: str, error_type: str, error_message: str) -> None:
        """
        Log an error occurrence.
        
        Args:
            endpoint: API endpoint where error occurred
            error_type: Type of error
            error_message: Error message
        """
        self.metrics["errors"][f"{endpoint}_{error_type}"] += 1
        
        logger.error(f"Error in {endpoint}: {error_type} - {error_message}")
    
    def log_token_usage(self, model: str, tokens_used: int, cost_estimate: float = 0.0) -> None:
        """
        Log token usage for LLM calls.
        
        Args:
            model: Model name
            tokens_used: Number of tokens used
            cost_estimate: Estimated cost in USD
        """
        self.metrics["token_usage"][model] += tokens_used
        
        logger.info(f"Token usage for {model}: {tokens_used} tokens (est. cost: ${cost_estimate:.4f})")
    
    def log_embedding_generation(self, model: str, count: int, duration: float) -> None:
        """
        Log embedding generation metrics.
        
        Args:
            model: Embedding model name
            count: Number of embeddings generated
            duration: Time taken in seconds
        """
        self.metrics["embedding_generations"][model] += count
        
        logger.info(f"Embedding generation: {count} embeddings using {model} in {duration:.3f}s")
    
    def log_search_operation(self, query_length: int, results_count: int, duration: float) -> None:
        """
        Log search operation metrics.
        
        Args:
            query_length: Length of search query
            results_count: Number of results returned
            duration: Search duration in seconds
        """
        self.metrics["search_operations"]["total_searches"] += 1
        self.metrics["search_operations"]["total_results"] += results_count
        
        logger.info(f"Search operation: {results_count} results for query ({query_length} chars) in {duration:.3f}s")
    
    def log_llm_call(self, model: str, duration: float, success: bool) -> None:
        """
        Log LLM call metrics.
        
        Args:
            model: LLM model name
            duration: Call duration in seconds
            success: Whether the call was successful
        """
        self.metrics["llm_calls"][f"{model}_total"] += 1
        
        if success:
            self.metrics["llm_calls"][f"{model}_success"] += 1
        else:
            self.metrics["llm_calls"][f"{model}_failed"] += 1
        
        logger.info(f"LLM call to {model}: {'success' if success else 'failed'} in {duration:.3f}s")
    
    def log_session_activity(self, session_id: str, query_length: int, tokens_used: int, duration: float) -> None:
        """
        Log session activity metrics.
        
        Args:
            session_id: Session identifier
            query_length: Length of user query
            tokens_used: Tokens used in response
            duration: Processing duration
        """
        self.session_metrics[session_id]["queries"] += 1
        self.session_metrics[session_id]["total_tokens"] += tokens_used
        self.session_metrics[session_id]["total_time"] += duration
        
        logger.debug(f"Session {session_id}: Query {query_length} chars, {tokens_used} tokens, {duration:.3f}s")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary containing metrics summary
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "requests": dict(self.metrics["requests"]),
            "errors": dict(self.metrics["errors"]),
            "token_usage": dict(self.metrics["token_usage"]),
            "embedding_generations": dict(self.metrics["embedding_generations"]),
            "search_operations": dict(self.metrics["search_operations"]),
            "llm_calls": dict(self.metrics["llm_calls"]),
            "response_times": {}
        }
        
        # Calculate average response times
        for endpoint, times in self.metrics["response_times"].items():
            if times:
                summary["response_times"][endpoint] = {
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times)
                }
        
        return summary
    
    def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session metrics
        """
        return dict(self.session_metrics.get(session_id, {}))
    
    def get_all_session_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all sessions.
        
        Returns:
            Dictionary of session metrics
        """
        return dict(self.session_metrics)
    
    def export_metrics(self, filepath: str) -> bool:
        """
        Export metrics to a JSON file.
        
        Args:
            filepath: Path to export file
            
        Returns:
            True if export was successful
        """
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "metrics": self.get_metrics_summary(),
                "session_metrics": self.get_all_session_metrics()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False
    
    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        self.metrics = {
            "requests": defaultdict(int),
            "response_times": defaultdict(list),
            "errors": defaultdict(int),
            "token_usage": defaultdict(int),
            "embedding_generations": defaultdict(int),
            "search_operations": defaultdict(int),
            "llm_calls": defaultdict(int)
        }
        
        self.session_metrics.clear()
        
        logger.info("All metrics have been reset") 