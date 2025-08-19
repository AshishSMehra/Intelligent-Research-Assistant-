"""
Chat API endpoints for the Intelligent Research Assistant.
"""

import hashlib
import json
import os
import random
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
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from flask import request
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, pipeline

from ..models.chat import ChatQuery, ChatResponse
from ..services.chat_service import ChatService

# Create router
chat_router = APIRouter(prefix="/chat", tags=["Chat"])

# Service instance
chat_service = ChatService()


@chat_router.post("/", response_model=ChatResponse)
async def chat_endpoint(chat_query: ChatQuery):
    """
    Chat endpoint for processing user queries.

    This endpoint receives user queries and returns intelligent responses
    based on the available document knowledge base.

    Args:
        chat_query: User query with optional metadata

    Returns:
        ChatResponse: Generated response with sources and metadata

    Raises:
        HTTPException: If query processing fails
    """
    try:
        logger.info("Received chat query: {chat_query.query[:50]}...")

        # Process the query
        response = await chat_service.process_query(chat_query)

        logger.info(
            "Successfully processed chat query in {response.processing_time:.2f}s"
        )

        return response

    except Exception as e:
        logger.error("Error processing chat query: {e}")
        raise HTTPException(status_code=500, detail="Failed to process query: {str(e)}")


@chat_router.get("/history/{session_id}")
async def get_conversation_history(session_id: str):
    """
    Get conversation history for a session.

    Args:
        session_id: Session identifier

    Returns:
        Conversation history
    """
    try:
        history = chat_service.get_conversation_history(session_id)

        return {
            "session_id": session_id,
            "history": history,
            "turns_count": len(history),
        }

    except Exception as e:
        logger.error("Error getting conversation history: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get conversation history: {str(e)}"
        )


@chat_router.delete("/history/{session_id}")
async def clear_conversation_history(session_id: str):
    """
    Clear conversation history for a session.

    Args:
        session_id: Session identifier

    Returns:
        Success status
    """
    try:
        success = chat_service.clear_conversation(session_id)

        if success:
            return {
                "status": "success",
                "message": "Conversation history cleared for session {session_id}",
            }
        else:
            return {"status": "not_found", "message": "Session {session_id} not found"}

    except Exception as e:
        logger.error("Error clearing conversation history: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to clear conversation history: {str(e)}"
        )


@chat_router.get("/metrics")
async def get_chat_metrics():
    """
    Get chat service metrics.

    Returns:
        Metrics summary
    """
    try:
        metrics = chat_service.get_metrics()

        return {"status": "success", "metrics": metrics}

    except Exception as e:
        logger.error("Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics: {str(e)}")


@chat_router.get("/llm/providers")
async def get_llm_providers():
    """
    Get available LLM providers.

    Returns:
        List of available LLM providers
    """
    try:
        providers = chat_service.llm_service.get_available_providers()

        return {
            "status": "success",
            "available_providers": providers,
            "current_provider": chat_service.llm_service.provider,
            "current_model": chat_service.llm_service.model_name,
        }

    except Exception as e:
        logger.error("Error getting LLM providers: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get LLM providers: {str(e)}"
        )


@chat_router.get("/health")
async def chat_health():
    """
    Health check endpoint for chat service.

    Returns:
        dict: Health status
    """
    return {
        "status": "healthy",
        "service": "chat",
        "message": "Chat service is operational",
        "components": {
            "search_service": "operational",
            "llm_service": "operational",
            "memory_service": "operational",
            "metrics_service": "operational",
        },
        "llm_provider": chat_service.llm_service.provider,
        "llm_model": chat_service.llm_service.model_name,
    }
