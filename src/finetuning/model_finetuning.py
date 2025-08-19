"""
Model Fine-tuning with LoRA/QLoRA and GPU Optimization.
"""

import os
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from loguru import logger

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    TaskType, PeftModel
)
from datasets import DatasetDict, Dataset
import bitsandbytes as bnb

from .gpu_config import GPUConfig


@dataclass
class TrainingConfig:
    """Training configuration."""
    model_name: str
    dataset_path: str
    output_dir: str
    num_epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-4
    max_length: int = 1024
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


class ModelFineTuning:
    """Fine-tune language models using LoRA/QLoRA."""
    
    def __init__(self, gpu_config: GPUConfig):
        """
        Initialize fine-tuning.
        
        Args:
            gpu_config: GPU configuration object
        """
        self.gpu_config = gpu_config
        self.device = gpu_config.device
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        logger.info(f"Model fine-tuning initialized with device: {self.device}")
    
    def prepare_model_and_tokenizer(
        self, 
        model_name: str, 
        use_qlora: bool = False,
        load_in_4bit: bool = True
    ) -> None:
        """
        Prepare model and tokenizer for fine-tuning.
        
        Args:
            model_name: Name of the base model
            use_qlora: Whether to use QLoRA (4-bit quantization)
            load_in_4bit: Whether to load model in 4-bit precision
        """
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate configuration
        if use_qlora and load_in_4bit:
            qlora_config = self.gpu_config.get_qlora_config()
            
            # Check if 4-bit quantization is available
            if qlora_config["load_in_4bit"]:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=bnb.BitsAndBytesConfig(
                            load_in_4bit=qlora_config["load_in_4bit"],
                            bnb_4bit_compute_dtype=qlora_config["bnb_4bit_compute_dtype"],
                            bnb_4bit_use_double_quant=qlora_config["bnb_4bit_use_double_quant"],
                            bnb_4bit_quant_type=qlora_config["bnb_4bit_quant_type"]
                        ),
                        device_map="auto" if self.device.type == "cuda" else None,
                        torch_dtype=torch.float16 if self.device.type == "mps" else torch.bfloat16
                    )
                    self.model = prepare_model_for_kbit_training(self.model)
                except Exception as e:
                    logger.warning(f"4-bit quantization failed, falling back to regular loading: {e}")
                    # Fallback to regular loading
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device.type == "mps" else torch.bfloat16,
                        device_map="auto" if self.device.type == "cuda" else None
                    )
            else:
                # Use regular loading if 4-bit quantization is not available
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device.type == "mps" else torch.bfloat16,
                    device_map="auto" if self.device.type == "cuda" else None
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "mps" else torch.bfloat16,
                device_map="auto" if self.device.type == "cuda" else None
            )
        
        # Move model to device if not using device_map
        if self.device.type != "cuda" or self.model.device.type == "cpu":
            self.model = self.model.to(self.device)
        
        logger.info(f"Model loaded successfully on {self.model.device}")
    
    def apply_lora_config(self, lora_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Apply LoRA configuration to the model.
        
        Args:
            lora_config: LoRA configuration (uses default if None)
        """
        if lora_config is None:
            lora_config = self.gpu_config.get_lora_config()
        
        logger.info(f"Applying LoRA configuration: {lora_config}")
        
        # Determine target modules based on model architecture
        model_name_lower = self.model.config.model_type.lower()
        
        if "gpt" in model_name_lower or "dialo" in model_name_lower:
            # GPT-style models (including DialoGPT)
            target_modules = ["c_attn", "c_proj", "c_fc", "c_proj"]
        elif "llama" in model_name_lower or "mistral" in model_name_lower:
            # LLaMA/Mistral models
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "bert" in model_name_lower:
            # BERT-style models
            target_modules = ["query", "key", "value", "output.dense"]
        else:
            # Default for unknown architectures
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA configuration applied successfully")
    
    def prepare_dataset(
        self, 
        dataset_dict: DatasetDict, 
        max_length: int = 1024
    ) -> DatasetDict:
        """
        Prepare dataset for training.
        
        Args:
            dataset_dict: HuggingFace dataset dictionary
            max_length: Maximum sequence length
            
        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            # Combine instruction, input, and output
            texts = []
            for instruction, input_text, output in zip(
                examples["instruction"], examples["input"], examples["output"]
            ):
                # Format: instruction + input + output
                full_text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                texts.append(full_text)
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Set labels to input_ids for causal language modeling
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Apply tokenization to all splits
        tokenized_datasets = {}
        for split_name, dataset in dataset_dict.items():
            tokenized_datasets[split_name] = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
        
        logger.info(f"Dataset tokenized with max_length={max_length}")
        return DatasetDict(tokenized_datasets)
    
    def create_trainer(
        self, 
        train_dataset, 
        eval_dataset, 
        training_config: TrainingConfig
    ) -> Trainer:
        """
        Create trainer for fine-tuning.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            training_config: Training configuration
            
        Returns:
            Configured trainer
        """
        # Training arguments
        training_args = TrainingArguments(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_epochs,
            per_device_train_batch_size=training_config.batch_size,
            per_device_eval_batch_size=training_config.batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            max_length=training_config.max_length,
            warmup_steps=training_config.warmup_steps,
            save_steps=training_config.save_steps,
            eval_steps=training_config.eval_steps,
            logging_steps=training_config.logging_steps,
            save_total_limit=training_config.save_total_limit,
            load_best_model_at_end=training_config.load_best_model_at_end,
            metric_for_best_model=training_config.metric_for_best_model,
            greater_is_better=training_config.greater_is_better,
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            dataloader_pin_memory=self.gpu_config.optimization_config["dataloader_pin_memory"],
            dataloader_num_workers=self.gpu_config.optimization_config["dataloader_num_workers"],
            fp16=self.gpu_config.optimization_config["fp16"],
            bf16=self.gpu_config.optimization_config["bf16"],
            gradient_checkpointing=True,
            report_to=["wandb"] if os.getenv("WANDB_API_KEY") else None,
            run_name=f"finetune-{training_config.model_name.split('/')[-1]}"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Callbacks
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            tokenizer=self.tokenizer
        )
        
        logger.info("Trainer created successfully")
        return self.trainer
    
    def train(self, training_config: TrainingConfig) -> Dict[str, Any]:
        """
        Execute fine-tuning.
        
        Args:
            training_config: Training configuration
            
        Returns:
            Training results
        """
        logger.info("Starting fine-tuning process")
        start_time = time.time()
        
        # Load dataset
        dataset_dict = self._load_dataset(training_config.dataset_path)
        
        # Prepare dataset
        tokenized_datasets = self.prepare_dataset(
            dataset_dict, 
            max_length=training_config.max_length
        )
        
        # Create trainer
        trainer = self.create_trainer(
            tokenized_datasets["train"],
            tokenized_datasets["validation"],
            training_config
        )
        
        # Train
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Evaluate
        logger.info("Evaluating model...")
        eval_result = trainer.evaluate()
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model()
        
        # Save training results
        results = {
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result["eval_loss"],
            "training_time": time.time() - start_time,
            "model_path": training_config.output_dir,
            "config": training_config.__dict__
        }
        
        # Save results
        results_path = os.path.join(training_config.output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training completed. Results saved to {results_path}")
        return results
    
    def _load_dataset(self, dataset_path: str) -> DatasetDict:
        """Load dataset from path."""
        if os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
            return DatasetDict.load_from_disk(dataset_path)
        else:
            # Load from JSON files
            dataset_dict = DatasetDict()
            for split in ["train", "validation", "test"]:
                split_path = os.path.join(dataset_path, f"{split}.json")
                if os.path.exists(split_path):
                    with open(split_path, 'r') as f:
                        data = json.load(f)
                    dataset_dict[split] = Dataset.from_list(data)
            return dataset_dict
    
    def merge_and_save_model(
        self, 
        base_model_name: str, 
        adapter_path: str, 
        output_path: str
    ) -> None:
        """
        Merge LoRA adapter with base model and save.
        
        Args:
            base_model_name: Name of the base model
            adapter_path: Path to the LoRA adapter
            output_path: Path to save the merged model
        """
        logger.info("Merging LoRA adapter with base model")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if self.device.type == "mps" else torch.bfloat16,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Merge weights
        model = model.merge_and_unload()
        
        # Save merged model
        model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"Merged model saved to {output_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            "model_type": type(self.model).__name__,
            "device": str(self.model.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        if hasattr(self.model, 'peft_config'):
            info["lora_config"] = self.model.peft_config
        
        return info 