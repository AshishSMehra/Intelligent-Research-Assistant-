"""
Dataset Preparation for Fine-tuning - Alpaca/ShareGPT Format.
"""

import json
import os
import random
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


@dataclass
class InstructionExample:
    """Single instruction-following example."""
    instruction: str
    input: str
    output: str
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetPreparation:
    """Prepare domain-specific instruction datasets for fine-tuning."""
    
    def __init__(self, output_dir: str = "finetuning_data"):
        """
        Initialize dataset preparation.
        
        Args:
            output_dir: Directory to save prepared datasets
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Instruction templates for different tasks
        self.instruction_templates = {
            "summarization": [
                "Summarize the following text:",
                "Provide a concise summary of:",
                "Create a brief overview of:"
            ],
            "question_answering": [
                "Answer the following question based on the context:",
                "Given the context, answer this question:",
                "Based on the information provided, answer:"
            ],
            "analysis": [
                "Analyze the following content:",
                "Provide an analysis of:",
                "Examine and discuss:"
            ],
            "extraction": [
                "Extract key information from:",
                "Identify the main points in:",
                "Find the important details in:"
            ],
            "comparison": [
                "Compare and contrast:",
                "Analyze the differences between:",
                "Examine the similarities and differences in:"
            ]
        }
        
        logger.info(f"Dataset preparation initialized with output directory: {output_dir}")
    
    def create_instruction_dataset_from_documents(
        self, 
        documents: List[Dict[str, Any]], 
        task_types: List[str] = None,
        examples_per_document: int = 5
    ) -> List[InstructionExample]:
        """
        Create instruction dataset from processed documents.
        
        Args:
            documents: List of document dictionaries with text and metadata
            task_types: Types of tasks to generate (default: all)
            examples_per_document: Number of examples per document
            
        Returns:
            List of instruction examples
        """
        if task_types is None:
            task_types = list(self.instruction_templates.keys())
        
        examples = []
        
        for doc in documents:
            doc_text = doc.get("text", "")
            doc_metadata = doc.get("metadata", {})
            
            if not doc_text.strip():
                continue
            
            # Generate examples for each task type
            for task_type in task_types:
                if task_type in self.instruction_templates:
                    task_examples = self._generate_task_examples(
                        doc_text, doc_metadata, task_type, examples_per_document
                    )
                    examples.extend(task_examples)
        
        logger.info(f"Generated {len(examples)} instruction examples from {len(documents)} documents")
        return examples
    
    def _generate_task_examples(
        self, 
        text: str, 
        metadata: Dict[str, Any], 
        task_type: str, 
        num_examples: int
    ) -> List[InstructionExample]:
        """
        Generate examples for a specific task type.
        
        Args:
            text: Document text
            metadata: Document metadata
            task_type: Type of task
            num_examples: Number of examples to generate
            
        Returns:
            List of instruction examples
        """
        examples = []
        templates = self.instruction_templates[task_type]
        
        # Split text into chunks for different examples
        chunks = self._split_text_for_tasks(text, task_type, num_examples)
        
        for i, chunk in enumerate(chunks):
            if i >= num_examples:
                break
            
            template = random.choice(templates)
            
            if task_type == "summarization":
                example = self._create_summarization_example(chunk, template, metadata)
            elif task_type == "question_answering":
                example = self._create_qa_example(chunk, template, metadata)
            elif task_type == "analysis":
                example = self._create_analysis_example(chunk, template, metadata)
            elif task_type == "extraction":
                example = self._create_extraction_example(chunk, template, metadata)
            elif task_type == "comparison":
                example = self._create_comparison_example(chunk, template, metadata)
            else:
                continue
            
            if example:
                examples.append(example)
        
        return examples
    
    def _split_text_for_tasks(self, text: str, task_type: str, num_chunks: int) -> List[str]:
        """Split text into appropriate chunks for different tasks."""
        sentences = text.split('. ')
        
        if task_type == "summarization":
            # For summarization, use longer chunks
            chunk_size = max(3, len(sentences) // num_chunks)
        else:
            # For other tasks, use shorter chunks
            chunk_size = max(2, len(sentences) // (num_chunks * 2))
        
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = '. '.join(sentences[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks[:num_chunks]
    
    def _create_summarization_example(
        self, text: str, template: str, metadata: Dict[str, Any]
    ) -> InstructionExample:
        """Create a summarization example."""
        # For now, we'll use a simple approach - in production, you'd use an LLM
        # to generate the summary and instruction
        words = text.split()
        summary_length = min(50, len(words) // 4)
        summary = ' '.join(words[:summary_length]) + "..."
        
        return InstructionExample(
            instruction=f"{template} {text[:200]}...",
            input=text,
            output=summary,
            context=metadata.get("document_id", ""),
            metadata={"task_type": "summarization", "source_metadata": metadata}
        )
    
    def _create_qa_example(
        self, text: str, template: str, metadata: Dict[str, Any]
    ) -> InstructionExample:
        """Create a question-answering example."""
        # Generate a simple question based on the text
        sentences = text.split('. ')
        if len(sentences) < 2:
            return None
        
        # Simple question generation (in production, use an LLM)
        first_sentence = sentences[0]
        words = first_sentence.split()
        if len(words) < 5:
            return None
        
        # Create a simple "what" question
        question = f"What is mentioned about {words[2] if len(words) > 2 else 'this topic'}?"
        answer = first_sentence
        
        return InstructionExample(
            instruction=f"{template} {question}",
            input=f"Context: {text}",
            output=answer,
            context=metadata.get("document_id", ""),
            metadata={"task_type": "question_answering", "source_metadata": metadata}
        )
    
    def _create_analysis_example(
        self, text: str, template: str, metadata: Dict[str, Any]
    ) -> InstructionExample:
        """Create an analysis example."""
        analysis = f"This text discusses {len(text.split())} words covering various topics. " \
                  f"The content appears to be {metadata.get('document_type', 'informational')} in nature."
        
        return InstructionExample(
            instruction=f"{template} {text[:150]}...",
            input=text,
            output=analysis,
            context=metadata.get("document_id", ""),
            metadata={"task_type": "analysis", "source_metadata": metadata}
        )
    
    def _create_extraction_example(
        self, text: str, template: str, metadata: Dict[str, Any]
    ) -> InstructionExample:
        """Create an information extraction example."""
        sentences = text.split('. ')
        key_points = sentences[:3]  # Extract first 3 sentences as key points
        
        extracted_info = '\n'.join([f"- {point.strip()}" for point in key_points if point.strip()])
        
        return InstructionExample(
            instruction=f"{template} {text[:150]}...",
            input=text,
            output=extracted_info,
            context=metadata.get("document_id", ""),
            metadata={"task_type": "extraction", "source_metadata": metadata}
        )
    
    def _create_comparison_example(
        self, text: str, template: str, metadata: Dict[str, Any]
    ) -> InstructionExample:
        """Create a comparison example."""
        sentences = text.split('. ')
        if len(sentences) < 4:
            return None
        
        # Split into two parts for comparison
        mid_point = len(sentences) // 2
        part1 = '. '.join(sentences[:mid_point])
        part2 = '. '.join(sentences[mid_point:])
        
        comparison = f"Part 1 focuses on {len(part1.split())} words, while Part 2 contains {len(part2.split())} words. " \
                    f"Both sections contribute to the overall topic."
        
        return InstructionExample(
            instruction=f"{template} the following two sections",
            input=f"Section 1: {part1}\n\nSection 2: {part2}",
            output=comparison,
            context=metadata.get("document_id", ""),
            metadata={"task_type": "comparison", "source_metadata": metadata}
        )
    
    def convert_to_alpaca_format(self, examples: List[InstructionExample]) -> List[Dict[str, str]]:
        """
        Convert examples to Alpaca format.
        
        Args:
            examples: List of instruction examples
            
        Returns:
            List of dictionaries in Alpaca format
        """
        alpaca_data = []
        
        for example in examples:
            alpaca_example = {
                "instruction": example.instruction,
                "input": example.input,
                "output": example.output
            }
            alpaca_data.append(alpaca_example)
        
        logger.info(f"Converted {len(examples)} examples to Alpaca format")
        return alpaca_data
    
    def convert_to_sharegpt_format(self, examples: List[InstructionExample]) -> List[Dict[str, Any]]:
        """
        Convert examples to ShareGPT format.
        
        Args:
            examples: List of instruction examples
            
        Returns:
            List of dictionaries in ShareGPT format
        """
        sharegpt_data = []
        
        for example in examples:
            sharegpt_example = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{example.instruction}\n\n{example.input}"
                    },
                    {
                        "from": "gpt",
                        "value": example.output
                    }
                ]
            }
            sharegpt_data.append(sharegpt_example)
        
        logger.info(f"Converted {len(examples)} examples to ShareGPT format")
        return sharegpt_data
    
    def create_huggingface_dataset(
        self, 
        examples: List[InstructionExample], 
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> DatasetDict:
        """
        Create HuggingFace dataset from examples.
        
        Args:
            examples: List of instruction examples
            test_size: Fraction of data for testing
            validation_size: Fraction of data for validation
            
        Returns:
            DatasetDict with train/validation/test splits
        """
        # Convert to Alpaca format
        alpaca_data = self.convert_to_alpaca_format(examples)
        
        # Split data
        train_data, temp_data = train_test_split(
            alpaca_data, test_size=test_size + validation_size, random_state=42
        )
        
        val_size_adjusted = validation_size / (test_size + validation_size)
        val_data, test_data = train_test_split(
            temp_data, test_size=1 - val_size_adjusted, random_state=42
        )
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)
        
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
        
        logger.info(f"Created dataset with {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test examples")
        return dataset_dict
    
    def save_dataset(self, dataset_dict: DatasetDict, format: str = "alpaca") -> str:
        """
        Save dataset to disk.
        
        Args:
            dataset_dict: HuggingFace dataset dictionary
            format: Format to save (alpaca, sharegpt, huggingface)
            
        Returns:
            Path to saved dataset
        """
        timestamp = str(int(time.time()))
        dataset_path = os.path.join(self.output_dir, f"dataset_{format}_{timestamp}")
        
        if format == "huggingface":
            dataset_dict.save_to_disk(dataset_path)
        else:
            # Save as JSON files
            for split_name, dataset in dataset_dict.items():
                split_path = os.path.join(dataset_path, f"{split_name}.json")
                os.makedirs(os.path.dirname(split_path), exist_ok=True)
                
                with open(split_path, 'w') as f:
                    json.dump(dataset.to_list(), f, indent=2)
        
        logger.info(f"Dataset saved to {dataset_path}")
        return dataset_path
    
    def load_existing_dataset(self, dataset_path: str) -> DatasetDict:
        """
        Load existing dataset from disk.
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            DatasetDict
        """
        if os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
            # HuggingFace format
            dataset_dict = DatasetDict.load_from_disk(dataset_path)
        else:
            # JSON format
            dataset_dict = DatasetDict()
            for split in ["train", "validation", "test"]:
                split_path = os.path.join(dataset_path, f"{split}.json")
                if os.path.exists(split_path):
                    with open(split_path, 'r') as f:
                        data = json.load(f)
                    dataset_dict[split] = Dataset.from_list(data)
        
        logger.info(f"Loaded dataset from {dataset_path}")
        return dataset_dict
    
    def get_dataset_stats(self, dataset_dict: DatasetDict) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Args:
            dataset_dict: HuggingFace dataset dictionary
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {}
        
        for split_name, dataset in dataset_dict.items():
            split_stats = {
                "num_examples": len(dataset),
                "avg_instruction_length": sum(len(ex["instruction"]) for ex in dataset) / len(dataset),
                "avg_input_length": sum(len(ex["input"]) for ex in dataset) / len(dataset),
                "avg_output_length": sum(len(ex["output"]) for ex in dataset) / len(dataset),
                "total_tokens_estimate": sum(
                    len(ex["instruction"].split()) + len(ex["input"].split()) + len(ex["output"].split())
                    for ex in dataset
                )
            }
            stats[split_name] = split_stats
        
        logger.info(f"Dataset statistics: {stats}")
        return stats 