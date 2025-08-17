"""
Research Agent for information retrieval and API calls.
"""

import time
import asyncio
from typing import Dict, Any, List
from loguru import logger

from .base_agent import BaseAgent, AgentTask, AgentResult
from ..services.search_service import SearchService


class ResearchAgent(BaseAgent):
    """Agent responsible for information retrieval and API calls."""
    
    def __init__(self, agent_id: str = "research_001"):
        """
        Initialize the Research Agent.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        capabilities = [
            "research",
            "vector_search",
            "web_search", 
            "api_search",
            "document_retrieval",
            "information_gathering"
        ]
        
        super().__init__(agent_id, "research", capabilities)
        
        # Initialize search service
        self.search_service = SearchService()
        
        # Research sources and methods
        self.research_methods = {
            "vector_search": self._perform_vector_search,
            "web_search": self._perform_web_search,
            "api_search": self._perform_api_search,
            "document_retrieval": self._retrieve_documents
        }
        
        logger.info(f"Research Agent {agent_id} initialized with {len(self.research_methods)} research methods")
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """
        Execute a research task.
        
        Args:
            task: The task to execute
            
        Returns:
            AgentResult: Research result with gathered information
        """
        start_time = time.time()
        self._log_task_start(task)
        
        try:
            if task.task_type == "research":
                result_data = await self._conduct_research(task)
            elif task.task_type == "vector_search":
                result_data = await self._perform_vector_search(task)
            elif task.task_type == "web_search":
                result_data = await self._perform_web_search(task)
            elif task.task_type == "api_search":
                result_data = await self._perform_api_search(task)
            elif task.task_type == "document_retrieval":
                result_data = await self._retrieve_documents(task)
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
            logger.error(f"Research Agent error: {e}")
        
        self._log_task_complete(task, result)
        self._update_metrics(result)
        self.task_history.append(result)
        
        return result
    
    async def _conduct_research(self, task: AgentTask) -> Dict[str, Any]:
        """
        Conduct comprehensive research using multiple methods.
        
        Args:
            task: The research task
            
        Returns:
            Dict containing research results
        """
        query = task.parameters.get("query", "")
        search_type = task.parameters.get("search_type", "vector_search")
        max_results = task.parameters.get("max_results", 5)
        
        research_results = {
            "query": query,
            "search_type": search_type,
            "results": [],
            "sources": [],
            "metadata": {}
        }
        
        # Perform the specified search type
        if search_type in self.research_methods:
            method = self.research_methods[search_type]
            results = await method(task)
            research_results["results"] = results.get("results", [])
            research_results["sources"] = results.get("sources", [])
            research_results["metadata"] = results.get("metadata", {})
        else:
            # Fallback to vector search
            results = await self._perform_vector_search(task)
            research_results["results"] = results.get("results", [])
            research_results["sources"] = results.get("sources", [])
            research_results["metadata"] = results.get("metadata", {})
        
        # Add research summary
        research_results["summary"] = self._create_research_summary(research_results)
        
        return research_results
    
    async def _perform_vector_search(self, task: AgentTask) -> Dict[str, Any]:
        """
        Perform vector search using the existing search service.
        
        Args:
            task: The search task
            
        Returns:
            Dict containing search results
        """
        query = task.parameters.get("query", "")
        limit = task.parameters.get("max_results", 5)
        score_threshold = task.parameters.get("score_threshold", 0.7)
        
        try:
            # Use the existing search service
            search_results = await self.search_service.semantic_search(
                query=query,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Convert to research format
            results = []
            sources = []
            
            for result in search_results:
                results.append({
                    "content": result.text,
                    "score": result.score,
                    "document_id": result.document_id,
                    "chunk_id": result.chunk_id,
                    "pages": result.source_pages
                })
                
                sources.append({
                    "type": "vector_database",
                    "document_id": result.document_id,
                    "pages": result.source_pages,
                    "score": result.score
                })
            
            return {
                "results": results,
                "sources": sources,
                "metadata": {
                    "search_type": "vector_search",
                    "query": query,
                    "results_count": len(results),
                    "average_score": sum(r["score"] for r in results) / len(results) if results else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return {
                "results": [],
                "sources": [],
                "metadata": {"error": str(e)}
            }
    
    async def _perform_web_search(self, task: AgentTask) -> Dict[str, Any]:
        """
        Perform web search (placeholder implementation).
        
        Args:
            task: The web search task
            
        Returns:
            Dict containing web search results
        """
        query = task.parameters.get("query", "")
        
        # Placeholder web search implementation
        # In production, this would integrate with search APIs like Google, Bing, etc.
        
        return {
            "results": [
                {
                    "content": f"Web search result for: {query}",
                    "score": 0.8,
                    "source": "web_search",
                    "url": "https://example.com/search-result"
                }
            ],
            "sources": [
                {
                    "type": "web_search",
                    "query": query,
                    "results_count": 1
                }
            ],
            "metadata": {
                "search_type": "web_search",
                "query": query,
                "results_count": 1
            }
        }
    
    async def _perform_api_search(self, task: AgentTask) -> Dict[str, Any]:
        """
        Perform API search (placeholder implementation).
        
        Args:
            task: The API search task
            
        Returns:
            Dict containing API search results
        """
        query = task.parameters.get("query", "")
        api_type = task.parameters.get("api_type", "general")
        
        # Placeholder API search implementation
        # In production, this would integrate with various APIs
        
        return {
            "results": [
                {
                    "content": f"API search result for: {query}",
                    "score": 0.9,
                    "source": f"{api_type}_api",
                    "api_response": {"status": "success"}
                }
            ],
            "sources": [
                {
                    "type": "api_search",
                    "api_type": api_type,
                    "query": query,
                    "results_count": 1
                }
            ],
            "metadata": {
                "search_type": "api_search",
                "api_type": api_type,
                "query": query,
                "results_count": 1
            }
        }
    
    async def _retrieve_documents(self, task: AgentTask) -> Dict[str, Any]:
        """
        Retrieve specific documents by ID.
        
        Args:
            task: The document retrieval task
            
        Returns:
            Dict containing document content
        """
        document_ids = task.parameters.get("document_ids", [])
        
        results = []
        sources = []
        
        for doc_id in document_ids:
            try:
                # Use search service to get document chunks
                doc_results = await self.search_service.search_by_document(doc_id)
                
                for chunk in doc_results:
                    results.append({
                        "content": chunk.get("text", ""),
                        "document_id": doc_id,
                        "chunk_id": chunk.get("chunk_id", ""),
                        "pages": chunk.get("source_pages", [])
                    })
                
                sources.append({
                    "type": "document_retrieval",
                    "document_id": doc_id,
                    "chunks_count": len(doc_results)
                })
                
            except Exception as e:
                logger.error(f"Document retrieval error for {doc_id}: {e}")
        
        return {
            "results": results,
            "sources": sources,
            "metadata": {
                "search_type": "document_retrieval",
                "documents_requested": len(document_ids),
                "documents_found": len(sources),
                "total_chunks": len(results)
            }
        }
    
    def _create_research_summary(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of research results.
        
        Args:
            research_results: The research results
            
        Returns:
            Dict containing research summary
        """
        results = research_results.get("results", [])
        sources = research_results.get("sources", [])
        
        summary = {
            "total_results": len(results),
            "total_sources": len(sources),
            "source_types": list(set(s.get("type", "unknown") for s in sources)),
            "average_score": sum(r.get("score", 0) for r in results) / len(results) if results else 0,
            "top_result": results[0] if results else None,
            "research_quality": "high" if len(results) > 3 else "medium" if len(results) > 1 else "low"
        }
        
        return summary 