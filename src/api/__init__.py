"""
API module for the Intelligent Research Assistant.

This module contains FastAPI endpoints and API-related functionality.
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

from .admin_api import admin_router
from .chat_api import chat_router
from .health_api import health_router
from .search_api import search_router

__all__ = ["chat_router", "search_router", "health_router", "admin_router"]
