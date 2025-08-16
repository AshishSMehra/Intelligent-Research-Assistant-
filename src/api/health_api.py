"""
Health check API endpoints for the Intelligent Research Assistant.
"""

from fastapi import APIRouter
from loguru import logger

# Create router
health_router = APIRouter(prefix="/health", tags=["Health"])


@health_router.get("/")
async def health_check():
    """
    Main health check endpoint.
    
    Returns:
        dict: Health status
    """
    return {
        "status": "healthy",
        "service": "intelligent-research-assistant",
        "version": "1.0.0"
    }


@health_router.get("/detailed")
async def detailed_health_check():
    """
    Detailed health check endpoint.
    
    Returns:
        dict: Detailed health status
    """
    try:
        # Check vector database connection
        from ..pipeline.pipeline import get_collection_info
        collection_info = get_collection_info()
        
        # Check embedding model
        from ..pipeline.pipeline import get_model_info
        model_info = get_model_info()
        
        return {
            "status": "healthy",
            "service": "intelligent-research-assistant",
            "version": "1.0.0",
            "components": {
                "vector_database": {
                    "status": "healthy" if collection_info.get("status") == "success" else "unhealthy",
                    "details": collection_info
                },
                "embedding_model": {
                    "status": "healthy" if "model_name" in model_info else "unhealthy",
                    "details": model_info
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "intelligent-research-assistant",
            "version": "1.0.0",
            "error": str(e)
        } 