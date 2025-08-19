"""
Reinforcement Learning from Human Feedback (RLHF) Module for Intelligent Research Assistant.

This module implements the complete RLHF pipeline:
1. Human Feedback Collection Mechanism
2. Reward Model Training
3. RL Fine-Tuning (Policy Optimization)
4. Policy Alignment with Stability Tricks
5. Evaluation and Comparison
6. Integration into Production System

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

from .evaluation import RLHFEvaluator
from .feedback_collection import FeedbackCollector, FeedbackDataset
from .integration import RLHFIntegration
from .policy_optimization import PolicyOptimizer, PPOTrainer
from .reward_model import RewardModel, RewardModelTrainer

__all__ = [
    "FeedbackCollector",
    "FeedbackDataset",
    "RewardModel",
    "RewardModelTrainer",
    "PolicyOptimizer",
    "PPOTrainer",
    "RLHFEvaluator",
    "RLHFIntegration",
]
