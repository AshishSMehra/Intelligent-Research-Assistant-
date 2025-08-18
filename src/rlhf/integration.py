"""
Step 6: Integration into Production System

This module implements the integration of RLHF models into the production system.
It provides A/B testing capabilities and live feedback collection.
"""

import os
import time
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from loguru import logger

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


@dataclass
class IntegrationConfig:
    """Configuration for RLHF integration."""
    baseline_model_path: str
    rlhf_model_path: str
    ab_test_ratio: float = 0.5  # 50% traffic to RLHF model
    enable_feedback_collection: bool = True
    feedback_storage_path: str = "live_feedback"
    model_cache_dir: str = "model_cache"
    device: str = "auto"


class RLHFIntegration:
    """Integration system for RLHF models in production."""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.device = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self.baseline_model = self._load_model(config.baseline_model_path)
        self.rlhf_model = self._load_model(config.rlhf_model_path)
        
        # Load tokenizers
        self.baseline_tokenizer = AutoTokenizer.from_pretrained(config.baseline_model_path)
        self.rlhf_tokenizer = AutoTokenizer.from_pretrained(config.rlhf_model_path)
        
        # Add padding tokens if needed
        for tokenizer in [self.baseline_tokenizer, self.rlhf_tokenizer]:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize feedback collection
        if config.enable_feedback_collection:
            self._init_feedback_collection()
        
        # A/B test tracking
        self.ab_test_stats = {
            "baseline_requests": 0,
            "rlhf_requests": 0,
            "total_requests": 0,
            "baseline_feedback": 0,
            "rlhf_feedback": 0
        }
        
        logger.info("RLHF integration initialized successfully")
    
    def _load_model(self, model_path: str):
        """Load a model from path."""
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model.to(self.device)
            model.eval()  # Set to evaluation mode
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None
    
    def _init_feedback_collection(self):
        """Initialize feedback collection system."""
        os.makedirs(self.config.feedback_storage_path, exist_ok=True)
        self.feedback_file = os.path.join(
            self.config.feedback_storage_path, 
            f"live_feedback_{int(time.time())}.jsonl"
        )
        
        # Initialize feedback file
        with open(self.feedback_file, 'w') as f:
            f.write("")  # Create empty file
        
        logger.info(f"Feedback collection initialized: {self.feedback_file}")
    
    def generate_response(self, prompt: str, user_id: str = None, session_id: str = None, 
                         max_length: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate response using A/B testing between baseline and RLHF models."""
        # Determine which model to use (A/B test)
        use_rlhf = random.random() < self.config.ab_test_ratio
        
        if use_rlhf and self.rlhf_model is not None:
            model = self.rlhf_model
            tokenizer = self.rlhf_tokenizer
            model_type = "rlhf"
            self.ab_test_stats["rlhf_requests"] += 1
        else:
            model = self.baseline_model
            tokenizer = self.baseline_tokenizer
            model_type = "baseline"
            self.ab_test_stats["baseline_requests"] += 1
        
        self.ab_test_stats["total_requests"] += 1
        
        # Generate response
        response = self._generate_with_model(model, tokenizer, prompt, max_length, temperature)
        
        # Create response object
        response_obj = {
            "prompt": prompt,
            "response": response,
            "model_type": model_type,
            "timestamp": time.time(),
            "user_id": user_id,
            "session_id": session_id,
            "request_id": f"req_{int(time.time())}_{random.randint(1000, 9999)}"
        }
        
        logger.info(f"Generated response using {model_type} model for user {user_id}")
        return response_obj
    
    def _generate_with_model(self, model, tokenizer, prompt: str, max_length: int, temperature: float) -> str:
        """Generate response using a specific model."""
        if model is None:
            return "Model not available"
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response
    
    def collect_feedback(self, request_id: str, rating: int, feedback_text: str = None, 
                        user_id: str = None) -> bool:
        """Collect user feedback for a specific response."""
        if not self.config.enable_feedback_collection:
            logger.warning("Feedback collection is disabled")
            return False
        
        # Validate rating
        if not (1 <= rating <= 5):
            logger.error(f"Invalid rating: {rating}. Must be between 1 and 5.")
            return False
        
        # Create feedback object
        feedback_obj = {
            "request_id": request_id,
            "rating": rating,
            "feedback_text": feedback_text,
            "user_id": user_id,
            "timestamp": time.time()
        }
        
        # Save feedback to file
        try:
            with open(self.feedback_file, 'a') as f:
                f.write(json.dumps(feedback_obj) + '\n')
            
            # Update stats
            if "rlhf" in request_id:
                self.ab_test_stats["rlhf_feedback"] += 1
            else:
                self.ab_test_stats["baseline_feedback"] += 1
            
            logger.info(f"Feedback collected for request {request_id}: rating={rating}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return False
    
    def get_ab_test_stats(self) -> Dict[str, Any]:
        """Get A/B test statistics."""
        stats = self.ab_test_stats.copy()
        
        # Calculate percentages
        if stats["total_requests"] > 0:
            stats["baseline_percentage"] = (stats["baseline_requests"] / stats["total_requests"]) * 100
            stats["rlhf_percentage"] = (stats["rlhf_requests"] / stats["total_requests"]) * 100
        else:
            stats["baseline_percentage"] = 0
            stats["rlhf_percentage"] = 0
        
        # Calculate feedback rates
        if stats["baseline_requests"] > 0:
            stats["baseline_feedback_rate"] = (stats["baseline_feedback"] / stats["baseline_requests"]) * 100
        else:
            stats["baseline_feedback_rate"] = 0
        
        if stats["rlhf_requests"] > 0:
            stats["rlhf_feedback_rate"] = (stats["rlhf_feedback"] / stats["rlhf_requests"]) * 100
        else:
            stats["rlhf_feedback_rate"] = 0
        
        return stats
    
    def analyze_feedback(self) -> Dict[str, Any]:
        """Analyze collected feedback to compare model performance."""
        if not os.path.exists(self.feedback_file):
            return {"error": "No feedback file found"}
        
        baseline_ratings = []
        rlhf_ratings = []
        
        try:
            with open(self.feedback_file, 'r') as f:
                for line in f:
                    if line.strip():
                        feedback = json.loads(line)
                        rating = feedback["rating"]
                        
                        if "rlhf" in feedback.get("request_id", ""):
                            rlhf_ratings.append(rating)
                        else:
                            baseline_ratings.append(rating)
            
            # Calculate statistics
            analysis = {
                "total_feedback": len(baseline_ratings) + len(rlhf_ratings),
                "baseline": {
                    "count": len(baseline_ratings),
                    "average_rating": sum(baseline_ratings) / len(baseline_ratings) if baseline_ratings else 0,
                    "ratings": baseline_ratings
                },
                "rlhf": {
                    "count": len(rlhf_ratings),
                    "average_rating": sum(rlhf_ratings) / len(rlhf_ratings) if rlhf_ratings else 0,
                    "ratings": rlhf_ratings
                }
            }
            
            # Calculate improvement
            if baseline_ratings and rlhf_ratings:
                baseline_avg = analysis["baseline"]["average_rating"]
                rlhf_avg = analysis["rlhf"]["average_rating"]
                improvement = rlhf_avg - baseline_avg
                improvement_percent = (improvement / baseline_avg * 100) if baseline_avg > 0 else 0
                
                analysis["improvement"] = {
                    "absolute": improvement,
                    "percentage": improvement_percent
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze feedback: {e}")
            return {"error": str(e)}
    
    def update_ab_test_ratio(self, new_ratio: float):
        """Update the A/B test ratio."""
        if 0.0 <= new_ratio <= 1.0:
            self.config.ab_test_ratio = new_ratio
            logger.info(f"A/B test ratio updated to {new_ratio}")
        else:
            logger.error(f"Invalid A/B test ratio: {new_ratio}. Must be between 0.0 and 1.0.")
    
    def enable_model(self, model_type: str, enable: bool = True):
        """Enable or disable a specific model."""
        if model_type == "rlhf":
            if enable:
                self.config.ab_test_ratio = 0.5  # Reset to 50/50
                logger.info("RLHF model enabled")
            else:
                self.config.ab_test_ratio = 0.0  # Use only baseline
                logger.info("RLHF model disabled")
        elif model_type == "baseline":
            if enable:
                self.config.ab_test_ratio = 1.0  # Use only RLHF
                logger.info("Baseline model disabled (using only RLHF)")
            else:
                logger.error("Cannot disable baseline model completely")
        else:
            logger.error(f"Unknown model type: {model_type}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            "baseline_model": {
                "loaded": self.baseline_model is not None,
                "path": self.config.baseline_model_path
            },
            "rlhf_model": {
                "loaded": self.rlhf_model is not None,
                "path": self.config.rlhf_model_path
            },
            "ab_test_ratio": self.config.ab_test_ratio,
            "feedback_collection": self.config.enable_feedback_collection,
            "device": self.device
        }
        
        return info
    
    def export_feedback_dataset(self, output_path: str = None) -> str:
        """Export collected feedback as a dataset for retraining."""
        if not os.path.exists(self.feedback_file):
            logger.error("No feedback file found")
            return ""
        
        if output_path is None:
            timestamp = int(time.time())
            output_path = os.path.join(
                self.config.feedback_storage_path, 
                f"feedback_dataset_{timestamp}.json"
            )
        
        # Read all feedback
        feedback_data = []
        try:
            with open(self.feedback_file, 'r') as f:
                for line in f:
                    if line.strip():
                        feedback_data.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read feedback file: {e}")
            return ""
        
        # Group feedback by request
        request_feedback = {}
        for feedback in feedback_data:
            request_id = feedback["request_id"]
            if request_id not in request_feedback:
                request_feedback[request_id] = []
            request_feedback[request_id].append(feedback)
        
        # Create dataset format
        dataset = {
            "metadata": {
                "created_at": time.time(),
                "total_feedback": len(feedback_data),
                "unique_requests": len(request_feedback)
            },
            "feedback": feedback_data,
            "request_groups": request_feedback
        }
        
        # Save dataset
        try:
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            logger.info(f"Feedback dataset exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export feedback dataset: {e}")
            return ""
    
    def cleanup_old_feedback(self, days_to_keep: int = 30):
        """Clean up old feedback files."""
        if not os.path.exists(self.config.feedback_storage_path):
            return
        
        current_time = time.time()
        cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
        
        files_removed = 0
        for filename in os.listdir(self.config.feedback_storage_path):
            filepath = os.path.join(self.config.feedback_storage_path, filename)
            
            if os.path.isfile(filepath):
                file_time = os.path.getmtime(filepath)
                if file_time < cutoff_time:
                    try:
                        os.remove(filepath)
                        files_removed += 1
                        logger.info(f"Removed old feedback file: {filename}")
                    except Exception as e:
                        logger.error(f"Failed to remove {filename}: {e}")
        
        logger.info(f"Cleanup completed: {files_removed} files removed")


class ProductionRLHFManager:
    """High-level manager for RLHF in production."""
    
    def __init__(self, config: IntegrationConfig):
        self.integration = RLHFIntegration(config)
        
    def chat(self, prompt: str, user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """Main chat interface with A/B testing."""
        return self.integration.generate_response(prompt, user_id, session_id)
    
    def rate_response(self, request_id: str, rating: int, feedback_text: str = None, 
                     user_id: str = None) -> bool:
        """Rate a response (1-5 scale)."""
        return self.integration.collect_feedback(request_id, rating, feedback_text, user_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get production statistics."""
        return {
            "ab_test": self.integration.get_ab_test_stats(),
            "feedback_analysis": self.integration.analyze_feedback(),
            "model_info": self.integration.get_model_info()
        }
    
    def update_config(self, ab_test_ratio: float = None, enable_feedback: bool = None):
        """Update production configuration."""
        if ab_test_ratio is not None:
            self.integration.update_ab_test_ratio(ab_test_ratio)
        
        if enable_feedback is not None:
            self.integration.config.enable_feedback_collection = enable_feedback
    
    def export_data(self) -> str:
        """Export feedback data for retraining."""
        return self.integration.export_feedback_dataset()
    
    def cleanup(self, days_to_keep: int = 30):
        """Clean up old data."""
        self.integration.cleanup_old_feedback(days_to_keep) 