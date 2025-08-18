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

from .feedback_collection import FeedbackCollector, FeedbackDataset
from .reward_model import RewardModel, RewardModelTrainer
from .policy_optimization import PolicyOptimizer, PPOTrainer
from .evaluation import RLHFEvaluator
from .integration import RLHFIntegration

__all__ = [
    "FeedbackCollector",
    "FeedbackDataset", 
    "RewardModel",
    "RewardModelTrainer",
    "PolicyOptimizer",
    "PPOTrainer",
    "RLHFEvaluator",
    "RLHFIntegration"
] 