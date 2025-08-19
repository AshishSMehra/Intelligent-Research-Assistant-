"""
API module for the Intelligent Research Assistant.

This module contains FastAPI endpoints and API-related functionality.
"""

from .admin_api import admin_router
from .chat_api import chat_router
from .health_api import health_router
from .search_api import search_router

__all__ = ["chat_router", "search_router", "health_router", "admin_router"]
