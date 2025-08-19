"""
Search models for the Intelligent Research Assistant.
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
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, pipeline


class SearchQuery(BaseModel):
    """Search query model."""

    query: str = Field(..., description="Search query text")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    score_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    include_metadata: bool = Field(True, description="Include metadata in results")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "machine learning algorithms",
                "limit": 5,
                "score_threshold": 0.8,
                "include_metadata": True,
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
