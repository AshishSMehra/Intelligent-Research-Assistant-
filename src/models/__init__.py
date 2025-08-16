"""
Data models for the Intelligent Research Assistant.
"""

from .chat import ChatQuery, ChatResponse, ChatMetadata
from .search import SearchQuery, SearchResult

__all__ = [
    "ChatQuery",
    "ChatResponse", 
    "ChatMetadata",
    "SearchQuery",
    "SearchResult"
] 