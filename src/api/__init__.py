"""
API module for the Intelligent Research Assistant.

This module contains FastAPI endpoints and API-related functionality.
"""

from .chat_api import chat_router
from .search_api import search_router
from .health_api import health_router

__all__ = [
    "chat_router",
    "search_router", 
    "health_router"
] 