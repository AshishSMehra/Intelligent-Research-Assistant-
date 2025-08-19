"""
Multi-Agent Orchestration for the Intelligent Research Assistant.

This module implements a modular agent system with:
- Planner Agent: Task decomposition and tool selection
- Research Agent: Information retrieval and API calls
- Reasoner Agent: Validation and follow-up requests
- Executor Agent: Side effects and external operations
"""

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

from .agent_orchestrator import AgentOrchestrator
from .base_agent import BaseAgent
from .executor_agent import ExecutorAgent
from .planner_agent import PlannerAgent
from .reasoner_agent import ReasonerAgent
from .research_agent import ResearchAgent

__all__ = [
    "BaseAgent",
    "PlannerAgent",
    "ResearchAgent",
    "ReasonerAgent",
    "ExecutorAgent",
    "AgentOrchestrator",
]
