"""
Chat service for handling user queries and generating responses.
"""

import time
from typing import List, Dict, Any, Optional
from loguru import logger

from ..models.chat import ChatQuery, ChatResponse, ChatMetadata
from ..models.search import SearchResult
from .search_service import SearchService


class ChatService:
    """Service for handling chat functionality."""
    
    def __init__(self):
        """Initialize the chat service."""
        self.search_service = SearchService()
        logger.info("ChatService initialized")
    
    async def process_query(self, chat_query: ChatQuery) -> ChatResponse:
        """
        Process a user query and generate a response.
        
        Args:
            chat_query: The user's query with metadata
            
        Returns:
            ChatResponse: Generated response with sources
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing chat query: {chat_query.query[:50]}...")
            
            # Step 1: Search for relevant documents
            search_results = await self._search_relevant_documents(chat_query)
            
            # Step 2: Generate response (placeholder for now)
            response = await self._generate_response(chat_query.query, search_results)
            
            # Step 3: Prepare sources
            sources = self._prepare_sources(search_results, chat_query.include_sources)
            
            # Step 4: Prepare metadata
            metadata = self._prepare_metadata(chat_query, search_results, start_time)
            
            processing_time = time.time() - start_time
            
            return ChatResponse(
                query=chat_query.query,
                response=response,
                sources=sources,
                metadata=metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing chat query: {e}")
            processing_time = time.time() - start_time
            
            return ChatResponse(
                query=chat_query.query,
                response="I apologize, but I encountered an error while processing your query. Please try again.",
                sources=[],
                metadata={"error": str(e)},
                processing_time=processing_time
            )
    
    async def _search_relevant_documents(self, chat_query: ChatQuery) -> List[SearchResult]:
        """
        Search for documents relevant to the user query.
        
        Args:
            chat_query: The user's query
            
        Returns:
            List of relevant search results
        """
        try:
            # Convert chat metadata to search filters
            filters = self._build_search_filters(chat_query.metadata)
            
            # Perform semantic search
            search_results = await self.search_service.semantic_search(
                query=chat_query.query,
                limit=chat_query.max_results,
                filters=filters
            )
            
            logger.info(f"Found {len(search_results)} relevant documents")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def _generate_response(self, query: str, search_results: List[SearchResult]) -> str:
        """
        Generate a response based on the query and search results.
        
        Args:
            query: User's query
            search_results: Relevant search results
            
        Returns:
            Generated response text
        """
        # TODO: Implement actual response generation with LLM
        # For now, return a placeholder response
        
        if not search_results:
            return "I couldn't find any relevant information to answer your question. Please try rephrasing your query or check if the relevant documents have been uploaded."
        
        # Create a simple response based on search results
        top_result = search_results[0]
        response = f"Based on the available documents, I found some relevant information: {top_result.text[:200]}..."
        
        if len(search_results) > 1:
            response += f"\n\nI found {len(search_results)} relevant sources that might help answer your question."
        
        return response
    
    def _prepare_sources(self, search_results: List[SearchResult], include_sources: bool) -> List[Dict[str, Any]]:
        """
        Prepare source documents for the response.
        
        Args:
            search_results: Search results
            include_sources: Whether to include sources
            
        Returns:
            List of source documents
        """
        if not include_sources:
            return []
        
        sources = []
        for result in search_results:
            source = {
                "document_id": result.document_id,
                "chunk_id": result.chunk_id,
                "text": result.text[:300] + "..." if len(result.text) > 300 else result.text,
                "score": result.score,
                "pages": result.source_pages
            }
            sources.append(source)
        
        return sources
    
    def _prepare_metadata(self, chat_query: ChatQuery, search_results: List[SearchResult], start_time: float) -> Dict[str, Any]:
        """
        Prepare response metadata.
        
        Args:
            chat_query: Original query
            search_results: Search results
            start_time: Processing start time
            
        Returns:
            Response metadata
        """
        return {
            "total_sources": len(search_results),
            "max_results_requested": chat_query.max_results,
            "query_length": len(chat_query.query),
            "has_metadata": chat_query.metadata is not None,
            "processing_steps": ["search", "response_generation"],
            "model_used": "placeholder"  # TODO: Add actual model info
        }
    
    def _build_search_filters(self, metadata: Optional[ChatMetadata]) -> Dict[str, Any]:
        """
        Build search filters from chat metadata.
        
        Args:
            metadata: Chat metadata
            
        Returns:
            Search filters dictionary
        """
        if not metadata:
            return {}
        
        # For now, return empty filters to avoid validation errors
        # TODO: Implement proper filter mapping to Qdrant filter format
        return {} 