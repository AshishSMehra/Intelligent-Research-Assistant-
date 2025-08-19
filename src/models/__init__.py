"""
Data models for the Intelligent Research Assistant.
"""

from .chat import ChatMetadata, ChatQuery, ChatResponse
from .search import SearchQuery, SearchResult

__all__ = ["ChatQuery", "ChatResponse", "ChatMetadata", "SearchQuery", "SearchResult"]
