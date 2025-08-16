"""
Search models for the Intelligent Research Assistant.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """Search query model."""
    
    query: str = Field(..., description="Search query text")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    score_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    include_metadata: bool = Field(True, description="Include metadata in results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "machine learning algorithms",
                "limit": 5,
                "score_threshold": 0.8,
                "include_metadata": True
            }
        }


class SearchResult(BaseModel):
    """Search result model."""
    
    id: str = Field(..., description="Result ID")
    score: float = Field(..., description="Similarity score")
    text: str = Field(..., description="Result text")
    document_id: str = Field(..., description="Source document ID")
    chunk_id: int = Field(..., description="Chunk ID")
    source_pages: List[int] = Field(default_factory=list, description="Source pages")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata") 