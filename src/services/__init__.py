"""
Services for the Intelligent Research Assistant.

This module contains business logic and service layer components.
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, pipeline

from .chat_service import ChatService
from .document_service import DocumentService
from .embedding_service import EmbeddingService
from .llm_service import LLMService
from .memory_service import MemoryService
from .metrics_service import MetricsService
from .search_service import SearchService

__all__ = [
    "ChatService",
    "SearchService",
    "DocumentService",
    "EmbeddingService",
    "LLMService",
    "MemoryService",
    "MetricsService",
]
