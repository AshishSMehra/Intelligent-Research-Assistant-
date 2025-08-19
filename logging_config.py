import hashlib
import json
import os
import random
import re
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
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

# Define the base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Remove default handler
logger.remove()

# Add a handler for console output
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)

# Add a handler for file output with rotation
log_file_path = BASE_DIR / "logs" / "app.log"
logger.add(
    log_file_path,
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",  # Rotate the log file when it reaches 10 MB
    retention="7 days",  # Keep logs for 7 days
    compression="zip",  # Compress old log files
    enqueue=True,  # Make logging non-blocking
    backtrace=True,  # Show full stack trace on exceptions
    diagnose=True,  # Add exception variable values for debugging
)

# Intercept standard logging messages
import logging


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the caller from where the message was logged
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# Export the configured logger
__all__ = ["logger"]
