"""
Chat service for handling user queries and generating responses.
"""

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from ..models.chat import ChatMetadata, ChatQuery, ChatResponse
from ..models.search import SearchResult
from .llm_service import LLMService
from .memory_service import MemoryService
from .metrics_service import MetricsService
from .search_service import SearchService


class ChatService:
    """Service for handling chat functionality."""

    def __init__(self):
        """Initialize the chat service."""
        self.search_service = SearchService()
        self.llm_service = LLMService()
        self.memory_service = MemoryService()
        self.metrics_service = MetricsService()
        logger.info("ChatService initialized with all components")

    async def process_query(self, chat_query: ChatQuery) -> ChatResponse:
        """
        Process a user query and generate a response.

        This implements the complete RAG pipeline:
        1. Receive and validate query ✅
        2. Embed query ✅
        3. Retrieve relevant chunks ✅
        4. Construct prompt ✅
        5. Generate answer with LLM ✅
        6. Parse and format answer ✅
        7. Add conversation memory ✅
        8. Return response ✅
        9. Log metrics ✅

        Args:
            chat_query: The user's query with metadata

        Returns:
            ChatResponse: Generated response with sources and citations
        """
        start_time = time.time()
        request_id = None

        try:
            # Step 1: Log request and get session info
            session_id = (
                chat_query.metadata.session_id
                if chat_query.metadata
                else "default_session"
            )
            user_id = chat_query.metadata.user_id if chat_query.metadata else None

            request_id = self.metrics_service.log_request(
                endpoint="/chat", method="POST", user_id=user_id
            )

            logger.info(f"Processing chat query: {chat_query.query[:50]}...")

            # Step 2: Get conversation context (Step 7: Short-term memory)
            conversation_context = ""
            if session_id and session_id != "default_session":
                conversation_context = self.memory_service.get_context_for_query(
                    session_id, chat_query.query
                )

            # Step 3: Search for relevant documents (Step 3: Retrieve chunks)
            search_start = time.time()
            search_results = await self._search_relevant_documents(chat_query)
            search_duration = time.time() - search_start

            # Log search metrics
            self.metrics_service.log_search_operation(
                query_length=len(chat_query.query),
                results_count=len(search_results),
                duration=search_duration,
            )

            # Step 4: Generate response with LLM (Steps 4, 5, 6: Prompt, LLM, Parse)
            llm_start = time.time()
            llm_response = await self._generate_response_with_llm(
                chat_query.query, search_results, conversation_context
            )
            llm_duration = time.time() - llm_start

            # Log LLM metrics
            self.metrics_service.log_llm_call(
                model=llm_response["metadata"]["model_used"],
                duration=llm_duration,
                success=True,
            )

            # Step 5: Prepare sources and citations
            sources = self._prepare_sources(search_results, chat_query.include_sources)
            citations = llm_response.get("citations", [])

            # Step 6: Add to conversation memory (Step 7: Memory)
            if session_id and session_id != "default_session":
                self.memory_service.add_conversation_turn(
                    session_id=session_id,
                    user_query=chat_query.query,
                    assistant_response=llm_response["answer"],
                    context_chunks=search_results,
                    metadata=llm_response["metadata"],
                )

            # Step 7: Prepare final metadata
            processing_time = time.time() - start_time
            metadata = self._prepare_metadata(
                chat_query,
                search_results,
                llm_response,
                processing_time,
                search_duration,
                llm_duration,
            )

            # Step 8: Log session activity
            if session_id:
                self.metrics_service.log_session_activity(
                    session_id=session_id,
                    query_length=len(chat_query.query),
                    tokens_used=llm_response["metadata"].get("tokens_used", 0),
                    duration=processing_time,
                )

            # Step 9: Log response time
            self.metrics_service.log_response_time(
                endpoint="/chat", method="POST", duration=processing_time
            )

            return ChatResponse(
                query=chat_query.query,
                response=llm_response["answer"],
                sources=sources,
                metadata=metadata,
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time

            # Log error
            self.metrics_service.log_error(
                endpoint="/chat", error_type="processing_error", error_message=str(e)
            )

            logger.error(f"Error processing chat query: {e}")

            return ChatResponse(
                query=chat_query.query,
                response="I apologize, but I encountered an error while processing your query. Please try again.",
                sources=[],
                metadata={"error": str(e), "processing_time": processing_time},
                processing_time=processing_time,
            )

    async def _search_relevant_documents(
        self, chat_query: ChatQuery
    ) -> List[SearchResult]:
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
                query=chat_query.query, limit=chat_query.max_results, filters=filters
            )

            logger.info(f"Found {len(search_results)} relevant documents")
            return search_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    async def _generate_response_with_llm(
        self,
        query: str,
        search_results: List[SearchResult],
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """
        Generate a response using the LLM with proper prompt construction.

        Args:
            query: User's query
            search_results: Relevant search results
            conversation_context: Previous conversation context

        Returns:
            LLM response with answer and citations
        """
        try:
            # Convert search results to context chunks
            context_chunks = []
            for result in search_results:
                chunk = {
                    "text": result.text,
                    "document_id": result.document_id,
                    "source_pages": result.source_pages,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                context_chunks.append(chunk)

            # Generate answer using LLM service
            llm_response = await self.llm_service.generate_answer(
                query=query,
                context_chunks=context_chunks,
                max_tokens=500,
                temperature=0.3,
            )

            # Log token usage if available
            if "tokens_used" in llm_response["metadata"]:
                self.metrics_service.log_token_usage(
                    model=llm_response["metadata"]["model_used"],
                    tokens_used=llm_response["metadata"]["tokens_used"],
                )

            return llm_response

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return {
                "answer": "I couldn't generate a proper response. Please try again.",
                "citations": [],
                "metadata": {"model_used": "fallback", "error": str(e)},
            }

    def _prepare_sources(
        self, search_results: List[SearchResult], include_sources: bool
    ) -> List[Dict[str, Any]]:
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
                "text": (
                    result.text[:300] + "..." if len(result.text) > 300 else result.text
                ),
                "score": result.score,
                "pages": result.source_pages,
            }
            sources.append(source)

        return sources

    def _prepare_metadata(
        self,
        chat_query: ChatQuery,
        search_results: List[SearchResult],
        llm_response: Dict[str, Any],
        processing_time: float,
        search_duration: float,
        llm_duration: float,
    ) -> Dict[str, Any]:
        """
        Prepare response metadata.

        Args:
            chat_query: Original query
            search_results: Search results
            llm_response: LLM response
            processing_time: Total processing time
            search_duration: Search duration
            llm_duration: LLM duration

        Returns:
            Response metadata
        """
        return {
            "total_sources": len(search_results),
            "max_results_requested": chat_query.max_results,
            "query_length": len(chat_query.query),
            "has_metadata": chat_query.metadata is not None,
            "processing_steps": ["search", "llm_generation", "memory", "formatting"],
            "model_used": llm_response["metadata"]["model_used"],
            "tokens_used": llm_response["metadata"].get("tokens_used", 0),
            "search_duration": search_duration,
            "llm_duration": llm_duration,
            "citations_count": len(llm_response.get("citations", [])),
            "context_chunks_used": llm_response["metadata"].get(
                "context_chunks_used", 0
            ),
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

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            Conversation history
        """
        return self.memory_service.get_conversation_history(session_id)

    def clear_conversation(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if cleared successfully
        """
        return self.memory_service.clear_session(session_id)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get application metrics.

        Returns:
            Metrics summary
        """
        return self.metrics_service.get_metrics_summary()
