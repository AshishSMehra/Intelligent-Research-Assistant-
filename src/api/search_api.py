"""
Search API endpoints for the Intelligent Research Assistant.
"""

import hashlib
import json
import os
import random
import re
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests
from botocore.exceptions import NoCredentialsError
from datasets import Dataset, DatasetDict
from fastapi import APIRouter, HTTPException, Query
from flask import request
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, pipeline

from ..models.search import SearchQuery, SearchResult
from ..services.search_service import SearchService

# Create router
search_router = APIRouter(prefix="/search", tags=["Search"])

# Service instance
search_service = SearchService()


@search_router.post("/", response_model=List[SearchResult])
async def search_endpoint(search_query: SearchQuery):
    """
    Semantic search endpoint.

    Args:
        search_query: Search query with parameters

    Returns:
        List of search results
    """
    try:
        logger.info("Received search query: {search_query.query[:50]}...")

        results = await search_service.semantic_search(
            query=search_query.query,
            limit=search_query.limit,
            score_threshold=search_query.score_threshold,
        )

        logger.info("Found {len(results)} search results")
        return results

    except Exception as e:
        logger.error("Error in search: {e}")
        raise HTTPException(status_code=500, detail="Search failed: {str(e)}")


@search_router.get("/documents/{document_id}")
async def get_document_chunks(
    document_id: str,
    include_text: bool = Query(True, description="Include text in results"),
):
    """
    Get all chunks for a specific document.

    Args:
        document_id: Document ID
        include_text: Whether to include text

    Returns:
        Document chunks
    """
    try:
        results = await search_service.search_by_document(
            document_id=document_id, include_text=include_text
        )

        return {
            "document_id": document_id,
            "chunks_count": len(results),
            "chunks": results,
        }

    except Exception as e:
        logger.error("Error getting document chunks: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get document chunks: {str(e)}"
        )


@search_router.get("/stats")
async def get_search_stats():
    """
    Get search statistics.

    Returns:
        Search statistics
    """
    try:
        stats = await search_service.get_collection_stats()
        return stats

    except Exception as e:
        logger.error("Error getting search stats: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get search stats: {str(e)}"
        )
