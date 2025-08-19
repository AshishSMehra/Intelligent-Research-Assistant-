"""
Fine-tuning and Domain Adaptation Module for Intelligent Research Assistant.

This module implements Phase 4 of the project:
- Domain-specific instruction dataset preparation (Alpaca/ShareGPT format)
- Fine-tuning LLMs using LoRA/QLoRA with Hugging Face PEFT
- Evaluation with held-out test sets and metrics tracking
- Model registration in MLflow and W&B
- GPU-optimized training for Apple Silicon (MPS) and CUDA
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

from .dataset_preparation import DatasetPreparation
from .evaluation import ModelEvaluation
from .gpu_config import GPUConfig
from .model_finetuning import ModelFineTuning
from .model_registry import ModelRegistry

__all__ = [
    "DatasetPreparation",
    "ModelFineTuning",
    "ModelEvaluation",
    "ModelRegistry",
    "GPUConfig",
]
