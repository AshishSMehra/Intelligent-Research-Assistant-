"""
Step 1: Human Feedback Collection Mechanism

This module implements the human feedback collection system for RLHF.
It provides interfaces for collecting user judgments/preferences on LLM outputs.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset
from loguru import logger


@dataclass
class FeedbackExample:
    """Single feedback example with prompt and responses."""

    prompt: str
    responses: List[
        Dict[str, Any]
    ]  # [{"text": "...", "rating": 5, "model": "model_a"}]
    feedback_id: str
    timestamp: float
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FeedbackDataset:
    """Dataset containing human feedback examples."""

    examples: List[FeedbackExample]
    dataset_name: str
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "dataset_name": self.dataset_name,
            "created_at": self.created_at,
            "num_examples": len(self.examples),
            "examples": [asdict(example) for example in self.examples],
        }

    def to_huggingface_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset format."""
        data = []
        for example in self.examples:
            for i, response in enumerate(example.responses):
                data.append(
                    {
                        "prompt": example.prompt,
                        "response": response["text"],
                        "rating": response["rating"],
                        "model": response.get("model", "unknown"),
                        "feedback_id": example.feedback_id,
                        "timestamp": example.timestamp,
                        "user_id": example.user_id,
                        "metadata": example.metadata,
                    }
                )
        return Dataset.from_list(data)

    def save(self, filepath: str) -> None:
        """Save dataset to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Feedback dataset saved: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "FeedbackDataset":
        """Load dataset from file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        examples = []
        for example_data in data["examples"]:
            examples.append(FeedbackExample(**example_data))

        return cls(
            examples=examples,
            dataset_name=data["dataset_name"],
            created_at=data["created_at"],
        )


class FeedbackCollector:
    """Human feedback collection system."""

    def __init__(self, output_dir: str = "feedback_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.feedback_history = []

    def create_feedback_interface_cli(
        self, prompt: str, responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Simple CLI interface for collecting feedback."""
        print("\n" + "=" * 60)
        print("HUMAN FEEDBACK COLLECTION")
        print("=" * 60)
        print(f"Prompt: {prompt}")
        print("\nResponses to rate:")

        for i, response in enumerate(responses):
            print(f"\n--- Response {i+1} ---")
            print(f"Text: {response['text']}")
            if "model" in response:
                print(f"Model: {response['model']}")

        print("\nRate each response on a scale of 1-5:")
        print("1 = Very Poor, 2 = Poor, 3 = Fair, 4 = Good, 5 = Excellent")

        ratings = []
        for i, response in enumerate(responses):
            while True:
                try:
                    rating = int(input(f"Rating for Response {i+1} (1-5): "))
                    if 1 <= rating <= 5:
                        ratings.append(rating)
                        break
                    else:
                        print("Please enter a number between 1 and 5.")
                except ValueError:
                    print("Please enter a valid number.")

        # Update responses with ratings
        for i, response in enumerate(responses):
            response["rating"] = ratings[i]

        return {"prompt": prompt, "responses": responses, "timestamp": time.time()}

    def create_feedback_interface_web(
        self, prompt: str, responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Web interface for collecting feedback (placeholder for Flask integration)."""
        # This would be integrated with Flask web interface
        logger.info("Web feedback interface would be implemented here")
        return self.create_feedback_interface_cli(prompt, responses)

    def collect_feedback_batch(
        self,
        prompts: List[str],
        model_responses: List[List[str]],
        models: List[str] = None,
        interface: str = "cli",
    ) -> FeedbackDataset:
        """Collect feedback for a batch of prompts and responses."""
        if models is None:
            models = [f"model_{i}" for i in range(len(model_responses[0]))]

        examples = []

        for i, (prompt, responses) in enumerate(zip(prompts, model_responses)):
            # Prepare response data
            response_data = []
            for j, response_text in enumerate(responses):
                response_data.append(
                    {
                        "text": response_text,
                        "model": models[j] if j < len(models) else f"model_{j}",
                    }
                )

            # Collect feedback
            if interface == "cli":
                feedback = self.create_feedback_interface_cli(prompt, response_data)
            elif interface == "web":
                feedback = self.create_feedback_interface_web(prompt, response_data)
            else:
                raise ValueError(f"Unknown interface: {interface}")

            # Create feedback example
            example = FeedbackExample(
                prompt=feedback["prompt"],
                responses=feedback["responses"],
                feedback_id=f"feedback_{int(time.time())}_{i}",
                timestamp=feedback["timestamp"],
            )

            examples.append(example)
            self.feedback_history.append(example)

        # Create dataset
        dataset = FeedbackDataset(
            examples=examples,
            dataset_name=f"human_feedback_{int(time.time())}",
            created_at=time.time(),
        )

        return dataset

    def generate_synthetic_feedback(
        self,
        prompts: List[str],
        model_responses: List[List[str]],
        models: List[str] = None,
    ) -> FeedbackDataset:
        """Generate synthetic feedback for testing purposes."""
        if models is None:
            models = [f"model_{i}" for i in range(len(model_responses[0]))]

        examples = []

        for i, (prompt, responses) in enumerate(zip(prompts, model_responses)):
            # Generate synthetic ratings based on response length and content
            response_data = []
            for j, response_text in enumerate(responses):
                # Simple synthetic rating based on response quality heuristics
                rating = self._generate_synthetic_rating(response_text, prompt)

                response_data.append(
                    {
                        "text": response_text,
                        "rating": rating,
                        "model": models[j] if j < len(models) else f"model_{j}",
                    }
                )

            example = FeedbackExample(
                prompt=prompt,
                responses=response_data,
                feedback_id=f"synthetic_feedback_{int(time.time())}_{i}",
                timestamp=time.time(),
            )

            examples.append(example)

        dataset = FeedbackDataset(
            examples=examples,
            dataset_name=f"synthetic_feedback_{int(time.time())}",
            created_at=time.time(),
        )

        return dataset

    def _generate_synthetic_rating(self, response: str, prompt: str) -> int:
        """Generate synthetic rating based on response quality heuristics."""
        # Simple heuristics for synthetic ratings
        score = 3  # Base score

        # Length factor
        if len(response) < 10:
            score -= 1
        elif len(response) > 100:
            score += 1

        # Content quality factors
        if any(
            word in response.lower() for word in ["error", "sorry", "cannot", "unable"]
        ):
            score -= 1

        if any(
            word in response.lower() for word in ["example", "specifically", "detailed"]
        ):
            score += 1

        # Relevance to prompt
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words.intersection(response_words))
        if overlap > 2:
            score += 1

        return max(1, min(5, score))

    def save_feedback_dataset(
        self, dataset: FeedbackDataset, filename: str = None
    ) -> str:
        """Save feedback dataset to file."""
        if filename is None:
            filename = f"{dataset.dataset_name}.json"

        filepath = os.path.join(self.output_dir, filename)
        dataset.save(filepath)
        return filepath

    def load_feedback_dataset(self, filename: str) -> FeedbackDataset:
        """Load feedback dataset from file."""
        filepath = os.path.join(self.output_dir, filename)
        return FeedbackDataset.load(filepath)

    def get_feedback_statistics(self, dataset: FeedbackDataset) -> Dict[str, Any]:
        """Get statistics about the feedback dataset."""
        all_ratings = []
        model_ratings = {}

        for example in dataset.examples:
            for response in example.responses:
                rating = response["rating"]
                model = response.get("model", "unknown")

                all_ratings.append(rating)

                if model not in model_ratings:
                    model_ratings[model] = []
                model_ratings[model].append(rating)

        stats = {
            "total_examples": len(dataset.examples),
            "total_responses": len(all_ratings),
            "average_rating": sum(all_ratings) / len(all_ratings) if all_ratings else 0,
            "rating_distribution": {
                "1": all_ratings.count(1),
                "2": all_ratings.count(2),
                "3": all_ratings.count(3),
                "4": all_ratings.count(4),
                "5": all_ratings.count(5),
            },
            "model_performance": {},
        }

        for model, ratings in model_ratings.items():
            stats["model_performance"][model] = {
                "average_rating": sum(ratings) / len(ratings),
                "num_responses": len(ratings),
            }

        return stats
