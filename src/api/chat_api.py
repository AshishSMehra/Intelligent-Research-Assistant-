"""
Chat API endpoints for the Intelligent Research Assistant.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from loguru import logger

from ..models.chat import ChatQuery, ChatResponse
from ..services.chat_service import ChatService

# Create router
chat_router = APIRouter(prefix="/chat", tags=["Chat"])

# Service instance
chat_service = ChatService()


@chat_router.post("/", response_model=ChatResponse)
async def chat_endpoint(chat_query: ChatQuery):
    """
    Chat endpoint for processing user queries.
    
    This endpoint receives user queries and returns intelligent responses
    based on the available document knowledge base.
    
    Args:
        chat_query: User query with optional metadata
        
    Returns:
        ChatResponse: Generated response with sources and metadata
        
    Raises:
        HTTPException: If query processing fails
    """
    try:
        logger.info(f"Received chat query: {chat_query.query[:50]}...")
        
        # Process the query
        response = await chat_service.process_query(chat_query)
        
        logger.info(f"Successfully processed chat query in {response.processing_time:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@chat_router.get("/health")
async def chat_health():
    """
    Health check endpoint for chat service.
    
    Returns:
        dict: Health status
    """
    return {
        "status": "healthy",
        "service": "chat",
        "message": "Chat service is operational"
    } 