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

from .dataset_preparation import DatasetPreparation
from .model_finetuning import ModelFineTuning
from .evaluation import ModelEvaluation
from .model_registry import ModelRegistry
from .gpu_config import GPUConfig

__all__ = [
    "DatasetPreparation",
    "ModelFineTuning", 
    "ModelEvaluation",
    "ModelRegistry",
    "GPUConfig"
] 