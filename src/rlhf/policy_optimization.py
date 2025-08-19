"""
Step 3: RL Fine-Tuning (Policy Optimization)

This module implements the PPO-based policy optimization for RLHF.
It adjusts the base LLM to maximize reward signals from the reward model.
"""

import hashlib
import json
import os
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from botocore.exceptions import NoCredentialsError
from datasets import Dataset, DatasetDict
from flask import request
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Import TRL for PPO (if available)
try:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    logger.warning("TRL not available. Install with: pip install trl")


@dataclass
class PPOTrainingConfig:
    """Configuration for PPO training."""

    model_name: str = "microsoft/DialoGPT-small"
    reward_model_path: str = "reward_models/final_reward_model"
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 1
    num_epochs: int = 4
    max_length: int = 512
    target_kl: float = 0.1
    clip_epsilon: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    kl_penalty: float = 0.1
    reward_clip: float = 1.0
    output_dir: str = "ppo_models"
    device: str = "auto"


class PolicyOptimizer:
    """Policy optimizer using PPO for RLHF."""

    def __init__(self, config: PPOTrainingConfig):
        self.config = config
        self.device = (
            config.device
            if config.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load policy model (fine-tuned LLM)
        self.policy_model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.policy_model.to(self.device)

        # Load reward model
        self.reward_model = self._load_reward_model()

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _load_reward_model(self):
        """Load the trained reward model."""
        if TRL_AVAILABLE:
            # Use TRL's value head model
            reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.config.reward_model_path
            )
        else:
            # Fallback to custom reward model
            from .reward_model import RewardModel

            reward_model = RewardModel(self.config.reward_model_path, self.device)

        reward_model.to(self.device)
        return reward_model

    def generate_response(
        self, prompt: str, max_length: int = 100, temperature: float = 0.7
    ) -> str:
        """Generate response using the policy model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return response

    def get_reward(self, prompt: str, response: str) -> float:
        """Get reward for a prompt-response pair."""
        if hasattr(self.reward_model, "get_reward"):
            return self.reward_model.get_reward(prompt, response)
        else:
            # Fallback for TRL models
            text = "{prompt} {response}"
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_length,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.reward_model(**inputs)
                reward = outputs.logits.squeeze().item()

            return reward

    def compute_kl_divergence(self, old_logits, new_logits):
        """Compute KL divergence between old and new policy."""
        old_probs = F.softmax(old_logits, dim=-1)
        new_probs = F.softmax(new_logits, dim=-1)

        kl_div = F.kl_div(new_probs.log(), old_probs, reduction="batchmean")

        return kl_div

    def train_with_ppo(self, prompts: List[str], num_iterations: int = 100):
        """Train the policy model using PPO."""
        logger.info("Starting PPO training...")

        if TRL_AVAILABLE:
            return self._train_with_trl_ppo(prompts, num_iterations)
        else:
            return self._train_with_custom_ppo(prompts, num_iterations)

    def _train_with_trl_ppo(self, prompts: List[str], num_iterations: int):
        """Train using TRL's PPO implementation."""
        logger.info("Using TRL PPO implementation...")

        # PPO configuration
        ppo_config = PPOConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            gradient_accumulation_steps=4,
            optimize_cuda_cache=True,
            early_stopping=True,
            target_kl=self.config.target_kl,
            max_grad_norm=1.0,
            seed=42,
            init_kl_coef=0.2,
            adap_kl_ctrl=True,
            steps=num_iterations,
            verbose=True,
        )

        # Create PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.policy_model,
            ref_model=None,  # Use current model as reference
            tokenizer=self.tokenizer,
            dataset=Dataset.from_dict({"text": prompts}),
            data_collator=self._collator,
            reward_model=self.reward_model,
        )

        # Train
        ppo_trainer.train()

        # Save the trained model
        model_path = os.path.join(self.config.output_dir, "ppo_final_model")
        self.policy_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

        logger.info("PPO training completed. Model saved to: {model_path}")
        return self.policy_model

    def _train_with_custom_ppo(self, prompts: List[str], num_iterations: int):
        """Custom PPO implementation."""
        logger.info("Using custom PPO implementation...")

        optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=self.config.learning_rate
        )

        for iteration in range(num_iterations):
            total_loss = 0
            total_reward = 0

            # Sample batch of prompts
            batch_prompts = np.random.choice(prompts, size=self.config.batch_size)

            for prompt in batch_prompts:
                # Generate response with current policy
                response = self.generate_response(prompt)

                # Get reward
                reward = self.get_reward(prompt, response)

                # Clip reward
                reward = np.clip(
                    reward, -self.config.reward_clip, self.config.reward_clip
                )

                # Compute policy loss (simplified PPO)
                loss = self._compute_policy_loss(prompt, response, reward)

                total_loss += loss.item()
                total_reward += reward

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            optimizer.step()

            # Log progress
            if iteration % 10 == 0:
                avg_loss = total_loss / self.config.batch_size
                avg_reward = total_reward / self.config.batch_size
                logger.info(
                    "Iteration {iteration}: Loss={avg_loss:.4f}, Avg Reward={avg_reward:.4f}"
                )

        # Save the trained model
        model_path = os.path.join(self.config.output_dir, "ppo_final_model")
        self.policy_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

        logger.info("Custom PPO training completed. Model saved to: {model_path}")
        return self.policy_model

    def _compute_policy_loss(self, prompt: str, response: str, reward: float):
        """Compute policy loss for PPO."""
        # This is a simplified version - full PPO would need more complex implementation
        text = "{prompt} {response}"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get logits
        outputs = self.policy_model(**inputs)
        logits = outputs.logits

        # Simple policy gradient loss
        loss = -torch.log(torch.softmax(logits, dim=-1)) * reward

        return loss.mean()

    def _collator(self, data):
        """Data collator for TRL PPO."""
        return {"input_ids": torch.stack([torch.tensor(d["text"]) for d in data])}

    def evaluate_policy(self, test_prompts: List[str]) -> Dict[str, float]:
        """Evaluate the trained policy model."""
        logger.info("Evaluating policy model...")

        total_reward = 0
        responses = []

        for prompt in test_prompts:
            response = self.generate_response(prompt)
            reward = self.get_reward(prompt, response)

            total_reward += reward
            responses.append({"prompt": prompt, "response": response, "reward": reward})

        avg_reward = total_reward / len(test_prompts)

        metrics = {
            "avg_reward": avg_reward,
            "total_reward": total_reward,
            "num_samples": len(test_prompts),
        }

        logger.info("Policy evaluation results: {metrics}")
        return metrics, responses


class PPOTrainer:
    """High-level PPO trainer for RLHF."""

    def __init__(self, config: PPOTrainingConfig):
        self.config = config
        self.policy_optimizer = PolicyOptimizer(config)

    def train(self, prompts: List[str], num_iterations: int = 100):
        """Train the policy model using PPO."""
        return self.policy_optimizer.train_with_ppo(prompts, num_iterations)

    def evaluate(self, test_prompts: List[str]):
        """Evaluate the trained policy model."""
        return self.policy_optimizer.evaluate_policy(test_prompts)

    def generate_response(self, prompt: str, max_length: int = 100):
        """Generate response using the trained policy model."""
        return self.policy_optimizer.generate_response(prompt, max_length)
