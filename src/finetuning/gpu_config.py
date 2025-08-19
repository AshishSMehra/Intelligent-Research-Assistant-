"""
GPU Configuration for Fine-tuning with Apple Silicon (MPS) and CUDA support.
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
import torch
from botocore.exceptions import NoCredentialsError
from datasets import Dataset, DatasetDict
from flask import request
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, pipeline


class GPUConfig:
    """GPU configuration and optimization for fine-tuning."""

    def __init__(self):
        """Initialize GPU configuration."""
        self.device = self._detect_device()
        self.device_info = self._get_device_info()
        self.optimization_config = self._get_optimization_config()

        logger.info("GPU Configuration initialized: {self.device_info}")

    def _detect_device(self) -> torch.device:
        """
        Detect the best available device for training.

        Returns:
            torch.device: Best available device (MPS, CUDA, or CPU)
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("CUDA GPU detected and will be used for training")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Apple Silicon MPS detected and will be used for training")
        else:
            device = torch.device("cpu")
            logger.warning(
                "No GPU detected, using CPU for training (this will be slow)"
            )

        return device

    def _get_device_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the detected device.

        Returns:
            Dict containing device information
        """
        info = {
            "device": str(self.device),
            "device_type": self.device.type,
            "is_gpu": self.device.type in ["cuda", "mps"],
        }

        if self.device.type == "cuda":
            info.update(
                {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
                    "gpu_count": torch.cuda.device_count(),
                    "cuda_version": torch.version.cuda,
                }
            )
        elif self.device.type == "mps":
            info.update(
                {
                    "gpu_name": "Apple Silicon GPU",
                    "mps_available": torch.backends.mps.is_available(),
                    "mps_built": torch.backends.mps.is_built(),
                }
            )

        return info

    def _get_optimization_config(self) -> Dict[str, Any]:
        """
        Get optimization configuration based on device.

        Returns:
            Dict containing optimization settings
        """
        config = {
            "mixed_precision": True,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "warmup_steps": 100,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
        }

        if self.device.type == "cuda":
            # CUDA-specific optimizations
            config.update(
                {
                    "bf16": True,  # Use bfloat16 for CUDA
                    "fp16": False,
                    "dataloader_pin_memory": True,
                    "dataloader_num_workers": 4,
                }
            )
        elif self.device.type == "mps":
            # MPS-specific optimizations
            config.update(
                {
                    "bf16": False,  # MPS doesn't support bfloat16
                    "fp16": True,  # Use float16 for MPS
                    "dataloader_pin_memory": False,  # MPS doesn't support pin_memory
                    "dataloader_num_workers": 2,
                }
            )
        else:
            # CPU optimizations
            config.update(
                {
                    "bf16": False,
                    "fp16": False,
                    "dataloader_pin_memory": False,
                    "dataloader_num_workers": 0,
                    "gradient_accumulation_steps": 1,
                }
            )

        return config

    def get_training_config(self, model_size: str = "medium") -> Dict[str, Any]:
        """
        Get training configuration optimized for the detected device.

        Args:
            model_size: Size of the model (small, medium, large)

        Returns:
            Dict containing training configuration
        """
        base_config = {
            "device": self.device,
            "device_info": self.device_info,
            "optimization": self.optimization_config,
        }

        # Model size specific configurations
        size_configs = {
            "small": {
                "batch_size": 4,
                "max_length": 512,
                "gradient_accumulation_steps": 2,
                "learning_rate": 3e-4,
            },
            "medium": {
                "batch_size": 2,
                "max_length": 1024,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2e-4,
            },
            "large": {
                "batch_size": 1,
                "max_length": 2048,
                "gradient_accumulation_steps": 8,
                "learning_rate": 1e-4,
            },
        }

        # Adjust batch size based on available memory
        if self.device.type == "cuda":
            gpu_memory_gb = self.device_info["gpu_memory"] / (1024**3)
            if gpu_memory_gb < 8:
                size_configs[model_size]["batch_size"] = max(
                    1, size_configs[model_size]["batch_size"] // 2
                )
        elif self.device.type == "mps":
            # MPS memory management is different, use conservative settings
            size_configs[model_size]["batch_size"] = max(
                1, size_configs[model_size]["batch_size"] // 2
            )

        base_config.update(size_configs[model_size])
        return base_config

    def get_lora_config(self) -> Dict[str, Any]:
        """
        Get LoRA configuration optimized for the device.

        Returns:
            Dict containing LoRA configuration
        """
        config = {
            "r": 16,  # Rank
            "lora_alpha": 32,  # Alpha parameter
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

        # Adjust LoRA rank based on device capabilities
        if self.device.type == "cuda":
            gpu_memory_gb = self.device_info["gpu_memory"] / (1024**3)
            if gpu_memory_gb >= 16:
                config["r"] = 32
            elif gpu_memory_gb >= 8:
                config["r"] = 16
            else:
                config["r"] = 8
        elif self.device.type == "mps":
            # Conservative settings for MPS
            config["r"] = 8
            config["lora_alpha"] = 16

        return config

    def get_qlora_config(self) -> Dict[str, Any]:
        """
        Get QLoRA configuration for 4-bit quantization.

        Returns:
            Dict containing QLoRA configuration
        """
        config = {
            "load_in_4bit": False,  # Default to False for compatibility
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        }

        # Check if bitsandbytes is properly installed with GPU support
        try:
            import bitsandbytes as bnb

            # Test if 4-bit quantization is available
            if hasattr(bnb, "BitsAndBytesConfig"):
                config["load_in_4bit"] = True
        except ImportError:
            logger.warning("bitsandbytes not available, QLoRA disabled")
            config["load_in_4bit"] = False

        # QLoRA is primarily for CUDA, but we can use it with MPS
        if self.device.type == "cuda" and config["load_in_4bit"]:
            config["bnb_4bit_compute_dtype"] = torch.bfloat16
        elif self.device.type == "mps" and config["load_in_4bit"]:
            # MPS doesn't support bfloat16, use float16
            config["bnb_4bit_compute_dtype"] = torch.float16
        else:
            # CPU doesn't support quantization or bitsandbytes not available
            config["load_in_4bit"] = False

        return config

    def optimize_memory(self) -> None:
        """Apply memory optimization techniques."""
        if self.device.type == "cuda":
            # CUDA memory optimization
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.9)
        elif self.device.type == "mps":
            # MPS memory optimization
            torch.mps.empty_cache()

        logger.info("Memory optimization applied for {self.device.type}")

    def get_dataloader_config(self) -> Dict[str, Any]:
        """
        Get dataloader configuration optimized for the device.

        Returns:
            Dict containing dataloader configuration
        """
        return {
            "pin_memory": self.optimization_config["dataloader_pin_memory"],
            "num_workers": self.optimization_config["dataloader_num_workers"],
            "persistent_workers": self.device.type == "cuda",
            "prefetch_factor": 2 if self.device.type == "cuda" else None,
        }

    def check_compatibility(self, model_name: str) -> Dict[str, Any]:
        """
        Check model compatibility with the current device.

        Args:
            model_name: Name of the model to check

        Returns:
            Dict containing compatibility information
        """
        compatibility = {
            "model": model_name,
            "device": str(self.device),
            "compatible": True,
            "warnings": [],
            "recommendations": [],
        }

        # Check for potential issues
        if self.device.type == "mps":
            if "llama" in model_name.lower() or "mistral" in model_name.lower():
                compatibility["warnings"].append(
                    "Some models may have compatibility issues with MPS"
                )
                compatibility["recommendations"].append(
                    "Consider using smaller models or CPU fallback"
                )

        if self.device.type == "cpu":
            compatibility["warnings"].append("Training on CPU will be very slow")
            compatibility["recommendations"].append(
                "Consider using a GPU-enabled environment"
            )

        return compatibility
