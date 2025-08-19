"""
Search service for semantic search functionality.
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
from flask import request
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, pipeline

from ..models.search import SearchQuery, SearchResult
from ..pipeline.pipeline import search_similar_chunks


class SearchService:
    """Service for handling search operations."""

    def __init__(self):
        """Initialize the search service."""
        logger.info("SearchService initialized")

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Perform semantic search for documents.

        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filters: Optional search filters

        Returns:
            List of search results
        """
        try:
            logger.info("Performing semantic search: {query[:50]}...")

            # Use the existing pipeline function
            search_results = search_similar_chunks(
                query_text=query,
                limit=limit,
                score_threshold=score_threshold,
                filter_conditions=filters,
                include_metadata=True,
            )

            # Convert to SearchResult objects
            results = []
            for result in search_results:
                search_result = SearchResult(
                    id=result["id"],
                    score=result["score"],
                    text=result["text"],
                    document_id=result["document_id"],
                    chunk_id=result["chunk_id"],
                    source_pages=result["source_pages"],
                    metadata=result.get("metadata"),
                )
                results.append(search_result)

            logger.info("Found {len(results)} search results")
            return results

        except Exception as e:
            logger.error("Error in semantic search: {e}")
            return []

    async def search_by_document(
        self, document_id: str, include_text: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for all chunks in a specific document.

        Args:
            document_id: Document ID to search
            include_text: Whether to include text in results

        Returns:
            List of document chunks
        """
        try:
            from ..pipeline.pipeline import search_by_document

            logger.info("Searching document: {document_id}")

            results = search_by_document(
                document_id=document_id, include_text=include_text
            )

            logger.info("Found {len(results)} chunks for document {document_id}")
            return results

        except Exception as e:
            logger.error("Error searching document {document_id}: {e}")
            return []

    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get vector database collection statistics.

        Returns:
            Collection statistics
        """
        try:
            from ..pipeline.pipeline import get_collection_stats

            stats = get_collection_stats()
            logger.info("Retrieved collection statistics")
            return stats

        except Exception as e:
            logger.error("Error getting collection stats: {e}")
            return {"error": str(e)}
