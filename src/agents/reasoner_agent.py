"""
Reasoner Agent for validation and follow-up requests.
"""

import hashlib
import json
import os
import random
import re
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests
from botocore.exceptions import NoCredentialsError
from datasets import Dataset, DatasetDict
from flask import request
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, pipeline

from ..services.llm_service import LLMService
from .base_agent import AgentResult, AgentTask, BaseAgent


class ReasonerAgent(BaseAgent):
    """Agent responsible for validation and follow-up requests."""

    def __init__(self, agent_id: str = "reasoner_001"):
        """
        Initialize the Reasoner Agent.

        Args:
            agent_id: Unique identifier for the agent
        """
        capabilities = [
            "text_analysis",
            "data_analysis",
            "sentiment_analysis",
            "fact_checking",
            "consistency_checking",
            "quality_assessment",
            "text_generation",
            "code_generation",
            "summary_generation",
        ]

        super().__init__(agent_id, "reasoner", capabilities)

        # Initialize LLM service for reasoning tasks
        self.llm_service = LLMService()

        # Analysis methods
        self.analysis_methods = {
            "text_analysis": self._analyze_text,
            "data_analysis": self._analyze_data,
            "sentiment_analysis": self._analyze_sentiment,
            "fact_checking": self._check_facts,
            "consistency_checking": self._check_consistency,
            "quality_assessment": self._assess_quality,
            "text_generation": self._generate_text,
            "code_generation": self._generate_code,
            "summary_generation": self._generate_summary,
        }

        logger.info(
            "Reasoner Agent {agent_id} initialized with {len(self.analysis_methods)} analysis methods"
        )

    async def execute_task(self, task: AgentTask) -> AgentResult:
        """
        Execute a reasoning task.

        Args:
            task: The task to execute

        Returns:
            AgentResult: Reasoning result with analysis and validation
        """
        start_time = time.time()
        self._log_task_start(task)

        try:
            task_type = task.task_type

            if task_type in self.analysis_methods:
                method = self.analysis_methods[task_type]
                result_data = await method(task)
            elif task_type == "analysis":
                result_data = await self._conduct_comprehensive_analysis(task)
            elif task_type == "generation":
                result_data = await self._generate_content(task)
            elif task_type == "validation":
                result_data = await self._validate_content(task)
            else:
                raise ValueError("Unknown task type: {task_type}")

            execution_time = time.time() - start_time
            result = AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                data=result_data,
                execution_time=execution_time,
                metadata={"task_type": task.task_type},
            )

        except Exception as e:
            execution_time = time.time() - start_time
            result = AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                data=None,
                error_message=str(e),
                execution_time=execution_time,
                metadata={"task_type": task.task_type},
            )
            logger.error("Reasoner Agent error: {e}")

        self._log_task_complete(task, result)
        self._update_metrics(result)
        self.task_history.append(result)

        return result

    async def _conduct_comprehensive_analysis(self, task: AgentTask) -> Dict[str, Any]:
        """
        Conduct comprehensive analysis of content.

        Args:
            task: The analysis task

        Returns:
            Dict containing comprehensive analysis results
        """
        analysis_type = task.parameters.get("analysis_type", "comprehensive")
        include_sentiment = task.parameters.get("include_sentiment", True)

        analysis_results = {
            "content": content,
            "analysis_type": analysis_type,
            "results": {},
            "summary": {},
            "recommendations": [],
        }

        # Perform text analysis
        text_analysis = await self._analyze_text(task)
        analysis_results["results"]["text_analysis"] = text_analysis

        # Perform sentiment analysis if requested
        if include_sentiment:
            sentiment_task = AgentTask(
                task_id=self._create_task_id(),
                task_type="sentiment_analysis",
                description="Analyze sentiment of content",
                parameters={"content": content},
            )
            sentiment_analysis = await self._analyze_sentiment(sentiment_task)
            analysis_results["results"]["sentiment_analysis"] = sentiment_analysis

        # Perform quality assessment
        quality_task = AgentTask(
            task_id=self._create_task_id(),
            task_type="quality_assessment",
            description="Assess quality of content",
            parameters={"content": content},
        )
        quality_assessment = await self._assess_quality(quality_task)
        analysis_results["results"]["quality_assessment"] = quality_assessment

        # Create summary
        analysis_results["summary"] = self._create_analysis_summary(analysis_results)

        # Generate recommendations
        analysis_results["recommendations"] = self._generate_recommendations(
            analysis_results
        )

        return analysis_results

    async def _analyze_text(self, task: AgentTask) -> Dict[str, Any]:
        """
        Analyze text content for various characteristics.

        Args:
            task: The text analysis task

        Returns:
            Dict containing text analysis results
        """

        analysis = {
            "word_count": len(content.split()),
            "character_count": len(content),
            "sentence_count": len([s for s in content.split(".") if s.strip()]),
            "paragraph_count": len([p for p in content.split("\n\n") if p.strip()]),
            "readability_score": self._calculate_readability(content),
            "complexity_level": self._assess_complexity(content),
            "key_topics": self._extract_key_topics(content),
            "language_detected": "en",  # Placeholder
        }

        return analysis

    async def _analyze_data(self, task: AgentTask) -> Dict[str, Any]:
        """
        Analyze data content.

        Args:
            task: The data analysis task

        Returns:
            Dict containing data analysis results
        """

        analysis = {
            "data_type": type(data).__name__,
            "data_size": len(str(data)),
            "structure_analysis": self._analyze_data_structure(data),
            "statistics": self._calculate_statistics(data),
        }

        return analysis

    async def _analyze_sentiment(self, task: AgentTask) -> Dict[str, Any]:
        """
        Analyze sentiment of content.

        Args:
            task: The sentiment analysis task

        Returns:
            Dict containing sentiment analysis results
        """

        # Simple sentiment analysis (placeholder)
        # In production, use proper sentiment analysis libraries

        positive_words = [
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "positive",
        ]
        negative_words = ["bad", "terrible", "awful", "horrible", "negative", "poor"]

        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)

        if positive_count > negative_count:
            sentiment = "positive"
            score = 0.7
        elif negative_count > positive_count:
            sentiment = "negative"
            score = 0.3
        else:
            sentiment = "neutral"
            score = 0.5

        return {
            "sentiment": sentiment,
            "sentiment_score": score,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "confidence": 0.8,
        }

    async def _check_facts(self, task: AgentTask) -> Dict[str, Any]:
        """
        Check facts in content.

        Args:
            task: The fact checking task

        Returns:
            Dict containing fact checking results
        """

        # Placeholder fact checking
        # In production, integrate with fact-checking APIs or databases

        return {
            "fact_check_status": "completed",
            "facts_checked": 5,
            "facts_verified": 4,
            "facts_unverified": 1,
            "confidence_score": 0.8,
            "verification_sources": ["internal_knowledge_base"],
        }

    async def _check_consistency(self, task: AgentTask) -> Dict[str, Any]:
        """
        Check consistency of content.

        Args:
            task: The consistency checking task

        Returns:
            Dict containing consistency check results
        """

        # Simple consistency check
        sentences = [s.strip() for s in content.split(".") if s.strip()]

        consistency_issues = []
        if len(sentences) > 1:
            # Check for contradictory statements (simplified)
            if "yes" in content.lower() and "no" in content.lower():
                consistency_issues.append("Potential contradiction detected")

        return {
            "consistency_score": 0.9 if not consistency_issues else 0.6,
            "issues_found": len(consistency_issues),
            "issues": consistency_issues,
            "overall_consistency": "high" if not consistency_issues else "medium",
        }

    async def _assess_quality(self, task: AgentTask) -> Dict[str, Any]:
        """
        Assess quality of content.

        Args:
            task: The quality assessment task

        Returns:
            Dict containing quality assessment results
        """

        # Quality metrics
        word_count = len(content.split())
        sentence_count = len([s for s in content.split(".") if s.strip()])

        quality_score = min(
            1.0, (word_count / 50) * 0.3 + (sentence_count / 3) * 0.3 + 0.4
        )

        return {
            "quality_score": quality_score,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "quality_level": (
                "high"
                if quality_score > 0.8
                else "medium" if quality_score > 0.6 else "low"
            ),
            "improvement_suggestions": self._generate_quality_suggestions(content),
        }

    async def _generate_content(self, task: AgentTask) -> Dict[str, Any]:
        """
        Generate content based on research results.

        Args:
            task: The generation task

        Returns:
            Dict containing generated content
        """
        research_results = task.parameters.get("research_results", {})
        response_type = task.parameters.get("response_type", "comprehensive")
        max_length = task.parameters.get("max_length", 500)

        # Extract content from research results
        content_pieces = []

        if "results" in research_results:
            for i, result in enumerate(
                research_results["results"][:3]
            ):  # Top 3 results
                content_pieces.append(result.get("content", ""))
                if include_citations:
                    citations.append(
                        {
                            "source": result.get("document_id", "source_{i}"),
                            "content": result.get("content", "")[:100] + "...",
                        }
                    )

        # Generate response using LLM
        context_chunks = [
            {"text": content, "document_id": "chunk_{i}"}
            for i, content in enumerate(content_pieces)
        ]

        query = task.parameters.get("query", "Generate a comprehensive response")

        llm_response = await self.llm_service.generate_answer(
            query=query,
            context_chunks=context_chunks,
            max_tokens=max_length,
            temperature=0.3,
        )

        return {
            "generated_content": llm_response["answer"],
            "citations": citations if include_citations else [],
            "response_type": response_type,
            "content_length": len(llm_response["answer"]),
            "metadata": llm_response["metadata"],
        }

    async def _generate_text(self, task: AgentTask) -> Dict[str, Any]:
        """Generate text content."""
        return await self._generate_content(task)

    async def _generate_code(self, task: AgentTask) -> Dict[str, Any]:
        """Generate code content."""
        # Placeholder for code generation
        return {
            "generated_code": "# Placeholder code generation",
            "language": "python",
            "complexity": "medium",
        }

    async def _generate_summary(self, task: AgentTask) -> Dict[str, Any]:
        """Generate summary content."""

        # Simple summary generation
        sentences = [s.strip() for s in content.split(".") if s.strip()]
        summary = ". ".join(sentences[:2]) + "." if len(sentences) > 2 else content

        return {
            "summary": summary,
            "original_length": len(content),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(content) if content else 0,
        }

    async def _validate_content(self, task: AgentTask) -> Dict[str, Any]:
        """
        Validate content for accuracy and quality.

        Args:
            task: The validation task

        Returns:
            Dict containing validation results
        """
        validation_type = task.parameters.get("validation_type", "fact_checking")

        validation_results = {
            "content": content,
            "validation_type": validation_type,
            "checks_performed": [],
            "overall_score": 0.0,
            "issues": [],
            "recommendations": [],
        }

        # Perform fact checking
        if "fact" in validation_type.lower():
            fact_check = await self._check_facts(task)
            validation_results["checks_performed"].append("fact_checking")
            validation_results["overall_score"] += (
                fact_check.get("confidence_score", 0.5) * 0.4
            )

        # Perform consistency check
        if "consistency" in validation_type.lower():
            consistency_check = await self._check_consistency(task)
            validation_results["checks_performed"].append("consistency_checking")
            validation_results["overall_score"] += (
                consistency_check.get("consistency_score", 0.5) * 0.3
            )

        # Perform quality assessment
        quality_check = await self._assess_quality(task)
        validation_results["checks_performed"].append("quality_assessment")
        validation_results["overall_score"] += (
            quality_check.get("quality_score", 0.5) * 0.3
        )

        # Generate recommendations
        if validation_results["overall_score"] < 0.7:
            validation_results["recommendations"].append(
                "Consider improving content quality"
            )

        return validation_results

    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score."""
        words = text.split()
        sentences = [s for s in text.split(".") if s.strip()]

        if not words or not sentences:
            return 0.0

        avg_sentence_length = len(words) / len(sentences)
        return max(0.0, min(1.0, 1.0 - (avg_sentence_length - 10) / 20))

    def _assess_complexity(self, text: str) -> str:
        """Assess text complexity."""
        readability = self._calculate_readability(text)

        if readability > 0.8:
            return "simple"
        elif readability > 0.6:
            return "moderate"
        else:
            return "complex"

    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from text."""
        # Simple keyword extraction
        common_words = [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "o",
            "with",
            "by",
        ]
        words = [
            word.lower()
            for word in text.split()
            if word.lower() not in common_words and len(word) > 3
        ]

        # Count word frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Return top 5 most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:5]]

    def _analyze_data_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze data structure."""
        return {
            "type": type(data).__name__,
            "is_iterable": hasattr(data, "__iter__"),
            "length": len(data) if hasattr(data, "__len__") else None,
        }

    def _calculate_statistics(self, data: Any) -> Dict[str, Any]:
        """Calculate basic statistics."""
        return {"count": len(str(data)), "type": type(data).__name__}

    def _create_analysis_summary(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create summary of analysis results."""
        results = analysis_results.get("results", {})

        summary = {
            "total_analyses": len(results),
            "analysis_types": list(results.keys()),
            "overall_quality": "high",
            "key_findings": [],
        }

        # Extract key findings
        for analysis_type, result in results.items():
            if "score" in result:
                summary["key_findings"].append("{analysis_type}: {result['score']:.2f}")

        return summary

    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        results = analysis_results.get("results", {})

        # Quality recommendations
        if "quality_assessment" in results:
            quality = results["quality_assessment"]
            if quality.get("quality_score", 0) < 0.7:
                recommendations.append("Improve content quality and structure")

        # Sentiment recommendations
        if "sentiment_analysis" in results:
            sentiment = results["sentiment_analysis"]
            if sentiment.get("sentiment") == "negative":
                recommendations.append("Consider more positive language")

        return recommendations

    def _generate_quality_suggestions(self, content: str) -> List[str]:
        """Generate quality improvement suggestions."""
        suggestions = []

        if len(content.split()) < 50:
            suggestions.append("Add more content for better comprehensiveness")

        if len(content.split(".")) < 3:
            suggestions.append("Break content into more sentences for clarity")

        return suggestions
