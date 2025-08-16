"""
Chat models for the Intelligent Research Assistant.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class ChatMetadata(BaseModel):
    """Metadata for chat queries."""
    
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    document_type: Optional[str] = Field(None, description="Filter by document type")
    date_from: Optional[datetime] = Field(None, description="Filter documents from date")
    date_to: Optional[datetime] = Field(None, description="Filter documents to date")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs to search")
    tags: Optional[List[str]] = Field(None, description="Filter by document tags")
    quality_threshold: Optional[float] = Field(0.7, description="Minimum quality score for results")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "session_id": "session456",
                "document_type": "research_paper",
                "tags": ["machine-learning", "ai"]
            }
        }


class ChatQuery(BaseModel):
    """User query model for chat endpoint."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="User's question or query")
    metadata: Optional[ChatMetadata] = Field(None, description="Optional query metadata")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results to return")
    include_sources: bool = Field(True, description="Include source documents in response")
    include_metadata: bool = Field(True, description="Include metadata in response")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty and contains meaningful content."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v.strip()) < 3:
            raise ValueError("Query must be at least 3 characters long")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the main concepts of machine learning?",
                "metadata": {
                    "user_id": "user123",
                    "session_id": "session456",
                    "document_type": "research_paper",
                    "tags": ["machine-learning", "ai"]
                },
                "max_results": 5,
                "include_sources": True,
                "include_metadata": True
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    
    query: str = Field(..., description="Original user query")
    response: str = Field(..., description="Generated response")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    processing_time: float = Field(..., description="Time taken to process query (seconds)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "query": "What are the main concepts of machine learning?",
                "response": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed...",
                "sources": [
                    {
                        "document_id": "doc123",
                        "chunk_id": 0,
                        "text": "Machine learning is a subset of artificial intelligence...",
                        "score": 0.95,
                        "page": 1
                    }
                ],
                "metadata": {
                    "total_sources": 1,
                    "confidence_score": 0.95,
                    "model_used": "gpt-3.5-turbo"
                },
                "processing_time": 1.23,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        } 