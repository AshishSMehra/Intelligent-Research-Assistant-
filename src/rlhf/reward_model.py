"""
Step 2: Train a Reward Model (RM)

This module implements the reward model training system for RLHF.
It converts human preferences into a learned reward signal using pairwise loss.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np


@dataclass
class RewardModelConfig:
    """Configuration for reward model training."""
    model_name: str = "microsoft/DialoGPT-small"
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 512
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "reward_models"
    device: str = "auto"


class RewardModel(nn.Module):
    """Reward model that predicts how good a response is."""
    
    def __init__(self, model_name: str, device: str = "auto"):
        super().__init__()
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained model for sequence classification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=1  # Single scalar reward
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the reward model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def get_reward(self, prompt: str, response: str) -> float:
        """Get reward score for a prompt-response pair."""
        # Concatenate prompt and response
        text = f"{prompt} {response}"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get reward
        with torch.no_grad():
            outputs = self.model(**inputs)
            reward = outputs.logits.squeeze().item()
        
        return reward
    
    def get_rewards_batch(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Get rewards for a batch of prompt-response pairs."""
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = self.get_reward(prompt, response)
            rewards.append(reward)
        return rewards


class RewardModelTrainer:
    """Trainer for the reward model using pairwise preferences."""
    
    def __init__(self, config: RewardModelConfig):
        self.config = config
        self.reward_model = RewardModel(config.model_name, config.device)
        self.tokenizer = self.reward_model.tokenizer
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
    def prepare_preference_dataset(self, feedback_dataset) -> Dataset:
        """Convert feedback dataset to preference pairs for training."""
        preference_data = []
        
        for example in feedback_dataset.examples:
            prompt = example.prompt
            responses = example.responses
            
            # Create preference pairs
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    response_i = responses[i]
                    response_j = responses[j]
                    
                    # Determine which response is preferred
                    if response_i["rating"] > response_j["rating"]:
                        chosen = response_i["text"]
                        rejected = response_j["text"]
                    elif response_j["rating"] > response_i["rating"]:
                        chosen = response_j["text"]
                        rejected = response_i["text"]
                    else:
                        # Skip if ratings are equal
                        continue
                    
                    # Create training example
                    preference_data.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "chosen_rating": max(response_i["rating"], response_j["rating"]),
                        "rejected_rating": min(response_i["rating"], response_j["rating"])
                    })
        
        logger.info(f"Created {len(preference_data)} preference pairs from feedback dataset")
        return Dataset.from_list(preference_data)
    
    def tokenize_function(self, examples):
        """Tokenize the preference pairs."""
        # Tokenize chosen responses
        chosen_texts = [f"{prompt} {chosen}" for prompt, chosen in zip(examples["prompt"], examples["chosen"])]
        chosen_tokens = self.tokenizer(
            chosen_texts,
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            return_tensors="pt"
        )
        
        # Tokenize rejected responses
        rejected_texts = [f"{prompt} {rejected}" for prompt, rejected in zip(examples["prompt"], examples["rejected"])]
        rejected_tokens = self.tokenizer(
            rejected_texts,
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"]
        }
    
    def compute_pairwise_loss(self, chosen_rewards, rejected_rewards, margin: float = 0.1):
        """Compute pairwise loss using Bradley-Terry model."""
        # Bradley-Terry loss: -log(σ(r_chosen - r_rejected))
        reward_diff = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(reward_diff).mean()
        
        # Add margin to encourage larger differences
        margin_loss = F.relu(margin - reward_diff).mean()
        
        return loss + 0.1 * margin_loss
    
    def train(self, feedback_dataset, validation_dataset=None):
        """Train the reward model on preference data."""
        logger.info("Starting reward model training...")
        
        # Prepare training dataset
        train_dataset = self.prepare_preference_dataset(feedback_dataset)
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        
        # Prepare validation dataset if provided
        if validation_dataset:
            val_dataset = self.prepare_preference_dataset(validation_dataset)
            val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        else:
            val_dataset = None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False if val_dataset else None,
            report_to=None,  # Disable wandb for now
            remove_unused_columns=False
        )
        
        # Custom trainer for pairwise loss
        trainer = PreferenceTrainer(
            model=self.reward_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        trainer.train()
        
        # Save the trained model
        model_path = os.path.join(self.config.output_dir, "final_reward_model")
        self.reward_model.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        logger.info(f"Reward model training completed. Model saved to: {model_path}")
        
        return self.reward_model
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = predictions.squeeze()
        
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        # Simple accuracy metric (predictions > 0)
        accuracy = (predictions > 0).mean()
        
        return {
            "accuracy": accuracy,
            "mean_prediction": predictions.mean(),
            "std_prediction": predictions.std()
        }
    
    def evaluate_reward_model(self, test_dataset) -> Dict[str, float]:
        """Evaluate the trained reward model."""
        logger.info("Evaluating reward model...")
        
        self.reward_model.eval()
        
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        test_pairs = self.prepare_preference_dataset(test_dataset)
        
        with torch.no_grad():
            for example in test_pairs:
                prompt = example["prompt"]
                chosen = example["chosen"]
                rejected = example["rejected"]
                
                # Get rewards
                chosen_reward = self.reward_model.get_reward(prompt, chosen)
                rejected_reward = self.reward_model.get_reward(prompt, rejected)
                
                # Check if prediction is correct
                if chosen_reward > rejected_reward:
                    correct_predictions += 1
                total_predictions += 1
                
                # Compute loss
                loss = self.compute_pairwise_loss(
                    torch.tensor([chosen_reward]), 
                    torch.tensor([rejected_reward])
                )
                total_loss += loss.item()
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_loss = total_loss / len(test_pairs) if test_pairs else 0
        
        metrics = {
            "accuracy": accuracy,
            "avg_loss": avg_loss,
            "total_predictions": total_predictions,
            "correct_predictions": correct_predictions
        }
        
        logger.info(f"Reward model evaluation results: {metrics}")
        return metrics


class PreferenceTrainer(Trainer):
    """Custom trainer for preference learning with pairwise loss."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute pairwise loss for preference learning."""
        # Get chosen and rejected inputs
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        
        # Get rewards for chosen responses
        chosen_outputs = model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask
        )
        chosen_rewards = chosen_outputs.logits.squeeze()
        
        # Get rewards for rejected responses
        rejected_outputs = model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask
        )
        rejected_rewards = rejected_outputs.logits.squeeze()
        
        # Compute pairwise loss
        loss = self.compute_pairwise_loss(chosen_rewards, rejected_rewards)
        
        return (loss, None) if return_outputs else loss
    
    def compute_pairwise_loss(self, chosen_rewards, rejected_rewards, margin: float = 0.1):
        """Compute pairwise loss using Bradley-Terry model."""
        reward_diff = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(reward_diff).mean()
        
        # Add margin to encourage larger differences
        margin_loss = F.relu(margin - reward_diff).mean()
        
        return loss + 0.1 * margin_loss 