"""
Security & Compliance Module for Intelligent Research Assistant.

This module implements comprehensive security features:
1. Role-Based Access Control (RBAC)
2. Secrets Management (AWS KMS/Vault)
3. PII Redaction
4. Rate Limiting & Abuse Detection
5. Data Retention Policies & Opt-out Mechanisms

Author: Ashish Mehra
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Ashish Mehra"

import hashlib
import json
import os
import random
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

from .data_retention import DataRetentionManager, RetentionPolicy
from .pii_redaction import PIIPatterns, PIIRedactor
from .rate_limiting import AbuseDetector, RateLimiter
from .rbac import Permission, RBACManager, Role, User
from .secrets import AWSKMSManager, SecretsManager, VaultManager

__all__ = [
    "RBACManager",
    "Role",
    "Permission",
    "User",
    "SecretsManager",
    "AWSKMSManager",
    "VaultManager",
    "PIIRedactor",
    "PIIPatterns",
    "RateLimiter",
    "AbuseDetector",
    "DataRetentionManager",
    "RetentionPolicy",
]
