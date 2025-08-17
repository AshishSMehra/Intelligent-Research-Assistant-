"""
Model Evaluation for Fine-tuned Models.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset, DatasetDict
import evaluate

from .gpu_config import GPUConfig


@dataclass
class EvaluationMetrics:
    """Evaluation metrics container."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    perplexity: float
    bleu_score: float
    rouge_score: float
    bert_score: float
    evaluation_time: float


class ModelEvaluation:
    """Evaluate fine-tuned language models."""
    
    def __init__(self, gpu_config: GPUConfig):
        """
        Initialize evaluation.
        
        Args:
            gpu_config: GPU configuration object
        """
        self.gpu_config = gpu_config
        self.device = gpu_config.device
        self.model = None
        self.tokenizer = None
        
        # Load evaluation metrics
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        self.bert_score_metric = evaluate.load("bertscore")
        
        logger.info(f"Model evaluation initialized with device: {self.device}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load fine-tuned model for evaluation.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        logger.info(f"Loading model from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device.type == "mps" else torch.bfloat16,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        # Move to device if needed
        if self.device.type != "cuda" or self.model.device.type == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info(f"Model loaded successfully on {self.model.device}")
    
    def evaluate_perplexity(self, test_dataset: Dataset) -> float:
        """
        Calculate perplexity on test dataset.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Perplexity score
        """
        logger.info("Calculating perplexity...")
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for example in test_dataset:
                # Tokenize input
                inputs = self.tokenizer(
                    example["text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"Perplexity: {perplexity:.4f}")
        return perplexity
    
    def generate_responses(
        self, 
        test_dataset: Dataset, 
        max_length: int = 512,
        temperature: float = 0.7
    ) -> List[str]:
        """
        Generate responses for test examples.
        
        Args:
            test_dataset: Test dataset
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            List of generated responses
        """
        logger.info("Generating responses...")
        
        generated_responses = []
        
        for example in test_dataset:
            # Prepare input
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            generated_responses.append(response.strip())
        
        logger.info(f"Generated {len(generated_responses)} responses")
        return generated_responses
    
    def calculate_text_metrics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate text generation metrics.
        
        Args:
            predictions: Generated responses
            references: Reference responses
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating text metrics...")
        
        metrics = {}
        
        # BLEU Score
        try:
            bleu_score = self.bleu_metric.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )["bleu"]
            metrics["bleu_score"] = bleu_score
        except Exception as e:
            logger.warning(f"Could not calculate BLEU score: {e}")
            metrics["bleu_score"] = 0.0
        
        # ROUGE Score
        try:
            rouge_scores = self.rouge_metric.compute(
                predictions=predictions,
                references=references
            )
            metrics["rouge_score"] = rouge_scores["rouge1"]
        except Exception as e:
            logger.warning(f"Could not calculate ROUGE score: {e}")
            metrics["rouge_score"] = 0.0
        
        # BERT Score
        try:
            bert_scores = self.bert_score_metric.compute(
                predictions=predictions,
                references=references,
                lang="en"
            )
            metrics["bert_score"] = np.mean(bert_scores["f1"])
        except Exception as e:
            logger.warning(f"Could not calculate BERT score: {e}")
            metrics["bert_score"] = 0.0
        
        return metrics
    
    def evaluate_model(
        self, 
        test_dataset: Dataset,
        reference_responses: Optional[List[str]] = None
    ) -> EvaluationMetrics:
        """
        Comprehensive model evaluation.
        
        Args:
            test_dataset: Test dataset
            reference_responses: Reference responses for comparison
            
        Returns:
            Evaluation metrics
        """
        logger.info("Starting comprehensive model evaluation")
        start_time = time.time()
        
        # Calculate perplexity
        perplexity = self.evaluate_perplexity(test_dataset)
        
        # Generate responses
        predictions = self.generate_responses(test_dataset)
        
        # Calculate text metrics if references provided
        if reference_responses:
            text_metrics = self.calculate_text_metrics(predictions, reference_responses)
            bleu_score = text_metrics["bleu_score"]
            rouge_score = text_metrics["rouge_score"]
            bert_score = text_metrics["bert_score"]
        else:
            bleu_score = 0.0
            rouge_score = 0.0
            bert_score = 0.0
        
        # For classification tasks, calculate accuracy metrics
        # This is a simplified version - in practice, you'd need task-specific evaluation
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        
        evaluation_time = time.time() - start_time
        
        metrics = EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            perplexity=perplexity,
            bleu_score=bleu_score,
            rouge_score=rouge_score,
            bert_score=bert_score,
            evaluation_time=evaluation_time
        )
        
        logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        return metrics
    
    def save_evaluation_results(
        self, 
        metrics: EvaluationMetrics, 
        output_path: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save evaluation results to file.
        
        Args:
            metrics: Evaluation metrics
            output_path: Path to save results
            additional_info: Additional information to save
        """
        results = {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "perplexity": metrics.perplexity,
            "bleu_score": metrics.bleu_score,
            "rouge_score": metrics.rouge_score,
            "bert_score": metrics.bert_score,
            "evaluation_time": metrics.evaluation_time,
            "device": str(self.device),
            "model_info": self.get_model_info()
        }
        
        if additional_info:
            results.update(additional_info)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def compare_models(
        self, 
        model_paths: List[str], 
        test_dataset: Dataset
    ) -> Dict[str, EvaluationMetrics]:
        """
        Compare multiple models.
        
        Args:
            model_paths: List of model paths to compare
            test_dataset: Test dataset
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        logger.info(f"Comparing {len(model_paths)} models")
        
        results = {}
        
        for model_path in model_paths:
            logger.info(f"Evaluating model: {model_path}")
            
            # Load model
            self.load_model(model_path)
            
            # Evaluate
            metrics = self.evaluate_model(test_dataset)
            
            # Store results
            model_name = os.path.basename(model_path)
            results[model_name] = metrics
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            "model_type": type(self.model).__name__,
            "device": str(self.model.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters())
        }
        
        return info
    
    def create_evaluation_report(
        self, 
        metrics: EvaluationMetrics,
        model_path: str,
        dataset_info: Dict[str, Any]
    ) -> str:
        """
        Create a comprehensive evaluation report.
        
        Args:
            metrics: Evaluation metrics
            model_path: Path to the evaluated model
            dataset_info: Information about the test dataset
            
        Returns:
            Formatted report string
        """
        report = f"""
# Model Evaluation Report

## Model Information
- **Model Path**: {model_path}
- **Device**: {self.device}
- **Evaluation Time**: {metrics.evaluation_time:.2f}s

## Dataset Information
- **Test Examples**: {dataset_info.get('num_examples', 'Unknown')}
- **Average Length**: {dataset_info.get('avg_length', 'Unknown')}

## Evaluation Metrics

### Language Model Metrics
- **Perplexity**: {metrics.perplexity:.4f}

### Text Generation Metrics
- **BLEU Score**: {metrics.bleu_score:.4f}
- **ROUGE Score**: {metrics.rouge_score:.4f}
- **BERT Score**: {metrics.bert_score:.4f}

### Classification Metrics (if applicable)
- **Accuracy**: {metrics.accuracy:.4f}
- **Precision**: {metrics.precision:.4f}
- **Recall**: {metrics.recall:.4f}
- **F1 Score**: {metrics.f1_score:.4f}

## Model Performance Summary
- **Overall Performance**: {'Good' if metrics.perplexity < 10 else 'Fair' if metrics.perplexity < 20 else 'Poor'}
- **Generation Quality**: {'Good' if metrics.bleu_score > 0.3 else 'Fair' if metrics.bleu_score > 0.1 else 'Poor'}

## Recommendations
- {'Consider fine-tuning for longer if perplexity is high' if metrics.perplexity > 15 else 'Model shows good performance'}
- {'Increase training data diversity' if metrics.bleu_score < 0.2 else 'Generation quality is acceptable'}
"""
        
        return report
    
    def _detect_hallucinations(self, predictions: List[str], references: List[str]) -> float:
        """Detect potential hallucinations in generated text."""
        try:
            # Simple hallucination detection based on:
            # 1. Factual consistency with references
            # 2. Presence of specific claims not in references
            # 3. Confidence vs. reference overlap
            
            hallucination_scores = []
            
            for pred, ref in zip(predictions, references):
                # Convert to lowercase for comparison
                pred_lower = pred.lower()
                ref_lower = ref.lower()
                
                # Split into words
                pred_words = set(pred_lower.split())
                ref_words = set(ref_lower.split())
                
                # Calculate word overlap
                overlap = len(pred_words.intersection(ref_words))
                total_pred_words = len(pred_words)
                
                if total_pred_words == 0:
                    hallucination_scores.append(1.0)  # High hallucination if no words
                else:
                    # Lower overlap = higher hallucination score
                    hallucination_score = 1.0 - (overlap / total_pred_words)
                    hallucination_scores.append(hallucination_score)
            
            # Return average hallucination score
            return sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error in hallucination detection: {e}")
            return 0.0 