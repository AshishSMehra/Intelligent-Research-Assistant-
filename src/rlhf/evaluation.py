"""
Step 5: Evaluation

This module implements comprehensive evaluation for RLHF models.
It compares baseline supervised models vs RLHF models on various metrics.
"""

import hashlib
import json
import os
import random
import re
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
from botocore.exceptions import NoCredentialsError
from datasets import Dataset, DatasetDict
from flask import request
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import evaluation metrics
try:
    import evaluate

    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    logger.warning("evaluate not available. Install with: pip install evaluate")


@dataclass
class EvaluationConfig:
    """Configuration for RLHF evaluation."""

    baseline_model_path: str
    rlhf_model_path: str
    test_prompts: List[str]
    max_length: int = 100
    temperature: float = 0.7
    num_samples: int = 10
    output_dir: str = "rlhf_evaluation"
    device: str = "auto"


class RLHFEvaluator:
    """Comprehensive evaluator for RLHF models."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = (
            config.device
            if config.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load models
        self.baseline_model = self._load_model(config.baseline_model_path)
        self.rlhf_model = self._load_model(config.rlhf_model_path)

        # Load tokenizers
        self.baseline_tokenizer = AutoTokenizer.from_pretrained(
            config.baseline_model_path
        )
        self.rlhf_tokenizer = AutoTokenizer.from_pretrained(config.rlhf_model_path)

        # Add padding tokens if needed
        for tokenizer in [self.baseline_tokenizer, self.rlhf_tokenizer]:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        # Initialize evaluation metrics
        self._init_metrics()

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _load_model(self, model_path: str):
        """Load a model from path."""
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model.to(self.device)
            return model
        except Exception as e:
            logger.error("Failed to load model from {model_path}: {e}")
            return None

    def _init_metrics(self):
        """Initialize evaluation metrics."""
        self.metrics = {}

        if EVALUATE_AVAILABLE:
            try:
                self.metrics["bleu"] = evaluate.load("bleu")
                self.metrics["rouge"] = evaluate.load("rouge")
                self.metrics["bertscore"] = evaluate.load("bertscore")
            except Exception as e:
                logger.warning("Failed to load some metrics: {e}")

    def generate_response(
        self, model, tokenizer, prompt: str, max_length: int = 100
    ) -> str:
        """Generate response using a model."""
        if model is None:
            return "Model not available"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return response

    def evaluate_factuality(self, prompt: str, response: str) -> float:
        """Evaluate factuality of response (simplified heuristic)."""
        # Simple factuality check based on response characteristics
        score = 0.5  # Base score

        # Check for uncertainty indicators
        uncertainty_words = [
            "maybe",
            "perhaps",
            "possibly",
            "might",
            "could",
            "uncertain",
        ]
        if any(word in response.lower() for word in uncertainty_words):
            score -= 0.1

        # Check for specific claims
        specific_indicators = ["specifically", "exactly", "precisely", "definitely"]
        if any(word in response.lower() for word in specific_indicators):
            score += 0.1

        # Check for citations or references
        if any(
            word in response.lower()
            for word in ["according to", "research shows", "study", "paper"]
        ):
            score += 0.2

        # Check for contradictions
        contradiction_words = ["however", "but", "although", "nevertheless"]
        if len([word for word in contradiction_words if word in response.lower()]) > 2:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def evaluate_helpfulness(self, prompt: str, response: str) -> float:
        """Evaluate helpfulness of response (simplified heuristic)."""
        # Simple helpfulness check
        score = 0.5  # Base score

        # Length factor
        if len(response) < 10:
            score -= 0.3
        elif len(response) > 50:
            score += 0.1

        # Relevance to prompt
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words.intersection(response_words))
        if overlap > 2:
            score += 0.2

        # Check for helpful indicators
        helpful_words = ["here", "example", "specifically", "detailed", "explanation"]
        if any(word in response.lower() for word in helpful_words):
            score += 0.1

        # Check for unhelpful indicators
        unhelpful_words = ["sorry", "cannot", "unable", "error", "don't know"]
        if any(word in response.lower() for word in unhelpful_words):
            score -= 0.3

        return max(0.0, min(1.0, score))

    def evaluate_diversity(self, responses: List[str]) -> float:
        """Evaluate diversity of responses."""
        if len(responses) < 2:
            return 0.0

        # Simple diversity metric based on unique words
        all_words = set()
        total_words = 0

        for response in responses:
            words = set(response.lower().split())
            all_words.update(words)
            total_words += len(words)

        if total_words == 0:
            return 0.0

        # Type-token ratio as diversity measure
        diversity = len(all_words) / total_words
        return diversity

    def evaluate_coherence(self, response: str) -> float:
        """Evaluate coherence of response."""
        # Simple coherence check
        score = 0.5  # Base score

        # Check for sentence structure
        sentences = response.split(".")
        if len(sentences) > 1:
            score += 0.2

        # Check for logical connectors
        connectors = ["because", "therefore", "thus", "hence", "so", "as a result"]
        if any(connector in response.lower() for connector in connectors):
            score += 0.1

        # Check for repetition
        words = response.lower().split()
        if len(words) > 0:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            score += repetition_ratio * 0.2

        return max(0.0, min(1.0, score))

    def compute_text_metrics(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute text generation metrics."""
        metrics = {}

        if EVALUATE_AVAILABLE:
            # BLEU Score
            if "bleu" in self.metrics:
                try:
                    bleu_score = self.metrics["bleu"].compute(
                        predictions=predictions, references=references
                    )
                    metrics["bleu_score"] = bleu_score["bleu"]
                except Exception as e:
                    logger.warning("BLEU calculation failed: {e}")
                    metrics["bleu_score"] = 0.0

            # ROUGE Score
            if "rouge" in self.metrics:
                try:
                    rouge_score = self.metrics["rouge"].compute(
                        predictions=predictions, references=references
                    )
                    metrics["rouge_score"] = rouge_score["rouge1"].mid.fmeasure
                except Exception as e:
                    logger.warning("ROUGE calculation failed: {e}")
                    metrics["rouge_score"] = 0.0

            # BERT Score
            if "bertscore" in self.metrics:
                try:
                    bert_score = self.metrics["bertscore"].compute(
                        predictions=predictions, references=references, lang="en"
                    )
                    metrics["bert_score"] = bert_score["f1"][0]
                except Exception as e:
                    logger.warning("BERT Score calculation failed: {e}")
                    metrics["bert_score"] = 0.0

        return metrics

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation comparing baseline vs RLHF models."""
        logger.info("Starting comprehensive RLHF evaluation...")

        results = {"baseline": {}, "rlh": {}, "comparison": {}}

        # Generate responses for both models
        baseline_responses = []
        rlhf_responses = []

        for prompt in self.config.test_prompts:
            # Generate multiple responses for diversity evaluation
            baseline_prompt_responses = []
            rlhf_prompt_responses = []

            for _ in range(self.config.num_samples):
                baseline_response = self.generate_response(
                    self.baseline_model, self.baseline_tokenizer, prompt
                )
                rlhf_response = self.generate_response(
                    self.rlhf_model, self.rlhf_tokenizer, prompt
                )

                baseline_prompt_responses.append(baseline_response)
                rlhf_prompt_responses.append(rlhf_response)

            baseline_responses.extend(baseline_prompt_responses)
            rlhf_responses.extend(rlhf_prompt_responses)

        # Evaluate baseline model
        results["baseline"] = self._evaluate_model_responses(
            self.config.test_prompts, baseline_responses, "baseline"
        )

        # Evaluate RLHF model
        results["rlh"] = self._evaluate_model_responses(
            self.config.test_prompts, rlhf_responses, "rlh"
        )

        # Compare models
        results["comparison"] = self._compare_models(
            results["baseline"], results["rlh"]
        )

        # Save results
        self._save_evaluation_results(results)

        return results

    def _evaluate_model_responses(
        self, prompts: List[str], responses: List[str], model_name: str
    ) -> Dict[str, Any]:
        """Evaluate responses from a specific model."""
        logger.info("Evaluating {model_name} model...")

        # Calculate metrics for each prompt
        factuality_scores = []
        helpfulness_scores = []
        coherence_scores = []

        for i, prompt in enumerate(prompts):
            start_idx = i * self.config.num_samples
            end_idx = start_idx + self.config.num_samples
            prompt_responses = responses[start_idx:end_idx]

            # Calculate average scores for this prompt
            prompt_factuality = np.mean(
                [
                    self.evaluate_factuality(prompt, response)
                    for response in prompt_responses
                ]
            )
            prompt_helpfulness = np.mean(
                [
                    self.evaluate_helpfulness(prompt, response)
                    for response in prompt_responses
                ]
            )
            prompt_coherence = np.mean(
                [self.evaluate_coherence(response) for response in prompt_responses]
            )

            factuality_scores.append(prompt_factuality)
            helpfulness_scores.append(prompt_helpfulness)
            coherence_scores.append(prompt_coherence)

        # Calculate diversity
        diversity_score = self.evaluate_diversity(responses)

        # Calculate text metrics (using first response per prompt as reference)
        reference_responses = [
            responses[i * self.config.num_samples] for i in range(len(prompts))
        ]
        text_metrics = self.compute_text_metrics(responses, reference_responses)

        return {
            "factuality": {
                "mean": np.mean(factuality_scores),
                "std": np.std(factuality_scores),
                "scores": factuality_scores,
            },
            "helpfulness": {
                "mean": np.mean(helpfulness_scores),
                "std": np.std(helpfulness_scores),
                "scores": helpfulness_scores,
            },
            "coherence": {
                "mean": np.mean(coherence_scores),
                "std": np.std(coherence_scores),
                "scores": coherence_scores,
            },
            "diversity": diversity_score,
            "text_metrics": text_metrics,
            "responses": responses,
        }

    def _compare_models(
        self, baseline_results: Dict, rlhf_results: Dict
    ) -> Dict[str, Any]:
        """Compare baseline and RLHF model results."""
        comparison = {}

        # Compare each metric
        for metric in ["factuality", "helpfulness", "coherence"]:
            baseline_mean = baseline_results[metric]["mean"]
            rlhf_mean = rlhf_results[metric]["mean"]

            improvement = rlhf_mean - baseline_mean
            improvement_percent = (
                (improvement / baseline_mean * 100) if baseline_mean > 0 else 0
            )

            comparison[metric] = {
                "baseline": baseline_mean,
                "rlh": rlhf_mean,
                "improvement": improvement,
                "improvement_percent": improvement_percent,
            }

        # Compare diversity
        baseline_diversity = baseline_results["diversity"]
        rlhf_diversity = rlhf_results["diversity"]
        diversity_improvement = rlhf_diversity - baseline_diversity

        comparison["diversity"] = {
            "baseline": baseline_diversity,
            "rlh": rlhf_diversity,
            "improvement": diversity_improvement,
        }

        # Compare text metrics
        comparison["text_metrics"] = {}
        for metric in baseline_results["text_metrics"]:
            baseline_score = baseline_results["text_metrics"][metric]
            rlhf_score = rlhf_results["text_metrics"][metric]

            improvement = rlhf_score - baseline_score
            improvement_percent = (
                (improvement / baseline_score * 100) if baseline_score > 0 else 0
            )

            comparison["text_metrics"][metric] = {
                "baseline": baseline_score,
                "rlh": rlhf_score,
                "improvement": improvement,
                "improvement_percent": improvement_percent,
            }

        return comparison

    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to file."""
        timestamp = int(time.time())
        results_file = os.path.join(
            self.config.output_dir, "evaluation_results_{timestamp}.json"
        )

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        results = convert_numpy(results)

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Evaluation results saved to: {results_file}")

        # Create summary report
        self._create_summary_report(results, results_file)

    def _create_summary_report(self, results: Dict[str, Any], results_file: str):
        """Create a human-readable summary report."""
        report_file = results_file.replace(".json", "_summary.txt")

        with open(report_file, "w") as f:
            f.write("RLHF EVALUATION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write("MODEL COMPARISON:\n")
            f.write("-" * 20 + "\n")

            for metric in ["factuality", "helpfulness", "coherence"]:
                comp = results["comparison"][metric]
                f.write("{metric.capitalize()}:\n")
                f.write("  Baseline: {comp['baseline']:.3f}\n")
                f.write("  RLHF: {comp['rlhf']:.3f}\n")
                f.write(
                    "  Improvement: {comp['improvement']:.3f} ({comp['improvement_percent']:.1f}%)\n\n"
                )

            f.write("Diversity:\n")
            f.write(
                "  Baseline: {results['comparison']['diversity']['baseline']:.3f}\n"
            )
            f.write("  RLHF: {results['comparison']['diversity']['rlhf']:.3f}\n")
            f.write(
                "  Improvement: {results['comparison']['diversity']['improvement']:.3f}\n\n"
            )

            f.write("TEXT METRICS:\n")
            f.write("-" * 15 + "\n")
            for metric, comp in results["comparison"]["text_metrics"].items():
                f.write("{metric}:\n")
                f.write("  Baseline: {comp['baseline']:.3f}\n")
                f.write("  RLHF: {comp['rlhf']:.3f}\n")
                f.write(
                    "  Improvement: {comp['improvement']:.3f} ({comp['improvement_percent']:.1f}%)\n\n"
                )

            f.write("CONCLUSION:\n")
            f.write("-" * 12 + "\n")

            # Determine overall improvement
            improvements = []
            for metric in ["factuality", "helpfulness", "coherence"]:
                improvements.append(
                    results["comparison"][metric]["improvement_percent"]
                )

            avg_improvement = np.mean(improvements)

            if avg_improvement > 5:
                f.write(
                    "✅ RLHF training shows significant improvement across all metrics.\n"
                )
            elif avg_improvement > 0:
                f.write("✅ RLHF training shows modest improvement.\n")
            else:
                f.write(
                    "⚠️ RLHF training did not show clear improvement. Consider adjusting parameters.\n"
                )

            f.write("Average improvement: {avg_improvement:.1f}%\n")

        logger.info("Summary report saved to: {report_file}")

    def run_human_evaluation(self, num_samples: int = 5) -> Dict[str, Any]:
        """Run human evaluation (blind A/B testing)."""
        logger.info("Starting human evaluation...")

        # Select random prompts for human evaluation
        selected_prompts = np.random.choice(
            self.config.test_prompts,
            size=min(num_samples, len(self.config.test_prompts)),
            replace=False,
        )

        evaluation_results = []

        for i, prompt in enumerate(selected_prompts):
            # Generate responses
            baseline_response = self.generate_response(
                self.baseline_model, self.baseline_tokenizer, prompt
            )
            rlhf_response = self.generate_response(
                self.rlhf_model, self.rlhf_tokenizer, prompt
            )

            # Randomly order responses for blind testing
            if np.random.random() > 0.5:
                response_a = baseline_response
                response_b = rlhf_response
                order = ["baseline", "rlh"]
            else:
                response_a = rlhf_response
                response_b = baseline_response
                order = ["rlh", "baseline"]

            # Human evaluation interface
            print("\n{'='*60}")
            print("HUMAN EVALUATION - Sample {i+1}/{len(selected_prompts)}")
            print("{'='*60}")
            print("Prompt: {prompt}")
            print("\nResponse A:")
            print("{response_a}")
            print("\nResponse B:")
            print("{response_b}")
            print("\nWhich response is better? (1=A, 2=B, 3=Equal): ", end="")

            try:
                choice = int(input())
                if choice == 1:
                    winner = order[0]
                elif choice == 2:
                    winner = order[1]
                else:
                    winner = "equal"

                evaluation_results.append(
                    {
                        "prompt": prompt,
                        "response_a": response_a,
                        "response_b": response_b,
                        "order": order,
                        "winner": winner,
                    }
                )

            except ValueError:
                logger.warning("Invalid input, skipping this sample")
                continue

        # Analyze human evaluation results
        baseline_wins = sum(
            1 for result in evaluation_results if result["winner"] == "baseline"
        )
        rlhf_wins = sum(1 for result in evaluation_results if result["winner"] == "rlh")
        ties = sum(1 for result in evaluation_results if result["winner"] == "equal")

        human_eval_results = {
            "total_samples": len(evaluation_results),
            "baseline_wins": baseline_wins,
            "rlhf_wins": rlhf_wins,
            "ties": ties,
            "baseline_win_rate": (
                baseline_wins / len(evaluation_results) if evaluation_results else 0
            ),
            "rlhf_win_rate": (
                rlhf_wins / len(evaluation_results) if evaluation_results else 0
            ),
            "detailed_results": evaluation_results,
        }

        logger.info("Human evaluation results: {human_eval_results}")
        return human_eval_results
