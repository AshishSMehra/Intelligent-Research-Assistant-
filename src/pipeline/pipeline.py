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

# Text extraction
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import requests
from botocore.exceptions import NoCredentialsError
from datasets import Dataset, DatasetDict
from flask import request

# Vector DB (Qdrant)
from qdrant_client import QdrantClient, models

# Embedding generation
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from logging_config import logger

# -----------------------------------------------------------------------------
# Text Extraction
# -----------------------------------------------------------------------------


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text content from a given PDF file with page structure preservation.

    Args:
        file_path (str): The local path to the PDF file.

    Returns:
        str: The concatenated text from all pages of the PDF.
    """
    try:
        doc = fitz.open(file_path)
        pages_data = extract_pages_from_pdf(file_path)

        # Concatenate all page texts for backward compatibility
        text = ""
        for page_data in pages_data:
            text += page_data["text"]
            if page_data["text"].strip():  # Add page separator if page has content
                text += "\n\n--- Page Break ---\n\n"

        logger.info(
            "Successfully extracted text from {file_path} ({len(pages_data)} pages)"
        )
        return text
    except Exception as e:
        logger.error("Failed to extract text from {file_path}. Error: {e}")
        return ""


def extract_pages_from_pdf(file_path: str) -> List[dict]:
    """
    Extracts text from PDF with comprehensive edge case handling and metadata.

    Args:
        file_path (str): The local path to the PDF file.

    Returns:
        List[dict]: List of page data with structure:
        [
            {
                'page_num': int,
                'text': str,
                'char_count': int,
                'is_empty': bool,
                'likely_image_only': bool,
                'likely_scanned': bool,
                'has_images': bool,
                'is_corrupted': bool,
                'processing_time_ms': float
            }
        ]
    """

    try:
        # Check if file exists and is readable
        if not os.path.exists(file_path):
            logger.error("PDF file not found: {file_path}")
            return []

        if os.path.getsize(file_path) == 0:
            logger.error("PDF file is empty: {file_path}")
            return []

        start_time = time.time()

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            if "password" in str(e).lower() or "encrypted" in str(e).lower():
                logger.error("PDF is password-protected or encrypted: {file_path}")
                return []
            elif "damaged" in str(e).lower() or "corrupt" in str(e).lower():
                logger.error("PDF appears to be corrupted: {file_path}")
                return []
            else:
                logger.error("Failed to open PDF {file_path}: {e}")
                return []

        # Check if PDF is valid
        if doc.page_count == 0:
            logger.warning("PDF has no pages: {file_path}")
            return []

        logger.info("Processing PDF with {doc.page_count} pages: {file_path}")

        pages_data = []
        empty_pages = 0
        image_heavy_pages = 0
        corrupted_pages = 0
        scanned_pages = 0
        very_large_pages = 0

        for page_num in range(len(doc)):
            page_start_time = time.time()

            try:
                page = doc.load_page(page_num)

                # Extract text with error handling
                try:
                    page_text = page.get_text()
                except Exception as e:
                    logger.error("Failed to extract text from page {page_num + 1}: {e}")
                    page_text = ""

                char_count = len(page_text.strip())

                # Get page images for analysis
                try:
                    image_list = page.get_images(full=True)
                    has_images = len(image_list) > 0
                    image_count = len(image_list)
                except Exception:
                    has_images = False
                    image_count = 0

                # Advanced edge case detection
                is_empty = char_count == 0
                likely_image_only = 0 < char_count < 50
                likely_scanned = (
                    has_images and char_count < 100
                )  # Images but little text
                is_corrupted = False
                is_very_large = char_count > 50000  # Very large page (>50k chars)

                # Detect potential OCR candidates (scanned pages)
                if has_images and char_count == 0:
                    likely_scanned = True
                    scanned_pages += 1
                    logger.warning(
                        "Page {page_num + 1} appears to be scanned (images but no text) - OCR may be needed"
                    )
                elif likely_scanned:
                    scanned_pages += 1
                    logger.info(
                        "Page {page_num + 1} may be scanned ({image_count} images, {char_count} chars)"
                    )

                page_processing_time = (time.time() - page_start_time) * 1000

                page_data = {
                    "page_num": page_num + 1,
                    "text": page_text,
                    "char_count": char_count,
                    "is_empty": is_empty,
                    "likely_image_only": likely_image_only,
                    "likely_scanned": likely_scanned,
                    "has_images": has_images,
                    "image_count": image_count,
                    "is_corrupted": is_corrupted,
                    "is_very_large": is_very_large,
                    "processing_time_ms": round(page_processing_time, 2),
                }

                pages_data.append(page_data)

                # Enhanced edge case logging
                if is_empty:
                    empty_pages += 1
                    if has_images:
                        logger.warning(
                            "Page {page_num + 1} is empty but contains {image_count} image(s) - likely scanned"
                        )
                    else:
                        logger.warning("Page {page_num + 1} is completely empty")
                elif likely_image_only:
                    image_heavy_pages += 1
                    logger.info(
                        "Page {page_num + 1} is image-heavy ({image_count} images, {char_count} chars)"
                    )
                elif is_very_large:
                    very_large_pages += 1
                    logger.info(
                        "Page {page_num + 1} is very large ({char_count:,} characters)"
                    )

                # Log slow pages
                if page_processing_time > 1000:  # > 1 second
                    logger.warning(
                        "Page {page_num + 1} took {page_processing_time:.0f}ms to process"
                    )

            except Exception as e:
                corrupted_pages += 1
                logger.error("Failed to process page {page_num + 1}: {e}")

                # Add corrupted page entry
                page_data = {
                    "page_num": page_num + 1,
                    "text": "",
                    "char_count": 0,
                    "is_empty": True,
                    "likely_image_only": False,
                    "likely_scanned": False,
                    "has_images": False,
                    "image_count": 0,
                    "is_corrupted": True,
                    "is_very_large": False,
                    "processing_time_ms": 0,
                }
                pages_data.append(page_data)

        # Comprehensive summary logging
        total_pages = len(pages_data)
        content_pages = total_pages - empty_pages - corrupted_pages
        processing_time = (time.time() - start_time) * 1000

        logger.info(
            "PDF analysis complete ({processing_time:.0f}ms): "
            "{total_pages} total pages, "
            "{content_pages} with content, "
            "{empty_pages} empty, "
            "{image_heavy_pages} image-heavy, "
            "{scanned_pages} likely scanned, "
            "{very_large_pages} very large, "
            "{corrupted_pages} corrupted"
        )

        # Recommendations based on analysis
        if scanned_pages > 0:
            logger.info(
                "💡 Recommendation: {scanned_pages} pages may benefit from OCR processing"
            )

        if corrupted_pages > 0:
            logger.warning(
                "⚠️  {corrupted_pages} pages could not be processed - PDF may be damaged"
            )

        if empty_pages > total_pages * 0.5:
            logger.warning("⚠️  More than 50% of pages are empty - check PDF quality")

        return pages_data

    except Exception as e:
        logger.error("Critical failure extracting pages from {file_path}: {e}")
        return []


# -----------------------------------------------------------------------------
# Text Chunking
# -----------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Splits a long text into smaller chunks with a specified overlap.
    Respects paragraph, sentence, and page boundaries when possible.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The desired size of each chunk in characters.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    if not text:
        return []

    # Validate overlap is not too large
    if chunk_overlap >= chunk_size:
        logger.warning(
            "Overlap ({chunk_overlap}) >= chunk_size ({chunk_size}). Setting overlap to {chunk_size // 2}"
        )
        chunk_overlap = chunk_size // 2

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # If this would be the last chunk and it's exactly the remaining text, take it all
        if end >= text_length:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        chunk = text[start:end]

        # Try to end chunk at a natural boundary (paragraph, sentence, or page break)
        # Look for page break first (highest priority)
        page_break_pos = chunk.rfind("--- Page Break ---")
        if page_break_pos > chunk_size * 0.5:  # Only if page break is in latter half
            chunk = text[start : start + page_break_pos]
        else:
            # Look for paragraph boundary (double newline)
            paragraph_end = chunk.rfind("\n\n")
            if (
                paragraph_end > chunk_size * 0.6
            ):  # Only if paragraph end is reasonably far
                chunk = text[
                    start : start + paragraph_end + 2
                ]  # Include the double newline
            else:
                # Look for sentence boundary
                sentence_end = max(
                    chunk.rfind(". "),
                    chunk.rfind("! "),
                    chunk.rfind("? "),
                    chunk.rfind(".\n"),
                    chunk.rfind("!\n"),
                    chunk.rfind("?\n"),
                )
                if (
                    sentence_end > chunk_size * 0.7
                ):  # Only if sentence end is reasonably far
                    chunk = text[start : start + sentence_end + 1]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        # Calculate next start position
        actual_chunk_length = len(chunk)
        step_size = max(1, min(actual_chunk_length, chunk_size) - chunk_overlap)
        start += step_size

    return chunks


def chunk_text_by_tokens(
    text: str, chunk_size: int = 500, chunk_overlap: int = 100
) -> List[dict]:
    """
    Splits text into chunks based on token count using tiktoken (OpenAI tokenizer).
    Provides more accurate chunking for embedding models that use token limits.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The desired number of tokens per chunk.
        chunk_overlap (int): The number of tokens to overlap between chunks.

    Returns:
        List[dict]: List of chunk data with token-based metadata:
        [
            {
                'text': str,
                'chunk_id': int,
                'start_token': int,
                'end_token': int,
                'token_count': int,
                'char_count': int
            }
        ]
    """
    try:
        import tiktoken
    except ImportError:
        logger.warning(
            "tiktoken not available, falling back to character-based chunking"
        )
        # Use a simple character-based fallback instead of calling the function
        char_chunks = []
        start = 0
        text_length = len(text)
        rough_chunk_size = chunk_size * 4
        rough_overlap = chunk_overlap * 4

        while start < text_length:
            end = start + rough_chunk_size
            chunk = text[start:end].strip()
            if chunk:
                char_chunks.append(chunk)
            start += rough_chunk_size - rough_overlap
        return [
            {
                "text": chunk,
                "chunk_id": i,
                "start_token": i * (chunk_size - chunk_overlap),
                "end_token": i * (chunk_size - chunk_overlap) + len(chunk) // 4,
                "token_count": len(chunk) // 4,  # Rough estimate
                "char_count": len(chunk),
            }
            for i, chunk in enumerate(char_chunks)
        ]

    if not text:
        return []

    # Validate overlap
    if chunk_overlap >= chunk_size:
        logger.warning(
            "Token overlap ({chunk_overlap}) >= chunk_size ({chunk_size}). Setting overlap to {chunk_size // 2}"
        )
        chunk_overlap = chunk_size // 2

    # Use OpenAI's tokenizer (most common for embeddings)
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    total_tokens = len(tokens)

    chunks = []
    chunk_id = 0
    start = 0

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)

        # If this would be the last chunk and it's exactly the remaining tokens, take them all
        if end >= total_tokens:
            chunk_tokens = tokens[start:]
            if chunk_tokens:
                chunk_text = enc.decode(chunk_tokens)
                chunks.append(
                    {
                        "text": chunk_text,
                        "chunk_id": chunk_id,
                        "start_token": start,
                        "end_token": total_tokens,
                        "token_count": len(chunk_tokens),
                        "char_count": len(chunk_text),
                    }
                )
            break

        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        chunks.append(
            {
                "text": chunk_text,
                "chunk_id": chunk_id,
                "start_token": start,
                "end_token": end,
                "token_count": len(chunk_tokens),
                "char_count": len(chunk_text),
            }
        )

        chunk_id += 1
        # Calculate next start position
        step_size = max(1, chunk_size - chunk_overlap)
        start += step_size

    logger.info("Token-based chunking: {len(chunks)} chunks from {total_tokens} tokens")
    return chunks


def chunk_text_with_pages(
    pages_data: List[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    document_id: str = None,
) -> List[dict]:
    """
    Chunks text while preserving page metadata and handling edge cases.

    Args:
        pages_data (List[dict]): Page data from extract_pages_from_pdf()
        chunk_size (int): The desired size of each chunk in characters.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[dict]: List of chunk data with enhanced metadata:
        [
            {
                'text': str,
                'chunk_id': int,
                'source_pages': List[int],
                'char_count': int,
                'has_scanned_content': bool,
                'quality_issues': List[str]
            }
        ]
    """
    if not pages_data:
        return []

    chunk_data = []
    chunk_id = 0
    skipped_pages = 0
    processed_pages = 0

    # Process each page with enhanced edge case handling
    for page_data in pages_data:
        page_num = page_data["page_num"]
        quality_issues = []

        # Skip problematic pages with detailed logging
        if page_data["is_empty"]:
            skipped_pages += 1
            if page_data.get("has_images", False):
                logger.info(
                    "Skipping page {page_num}: Empty but has images (likely scanned)"
                )
                quality_issues.append("scanned_no_text")
            else:
                logger.debug("Skipping page {page_num}: Completely empty")
                quality_issues.append("empty")
            continue

        if page_data.get("is_corrupted", False):
            skipped_pages += 1
            logger.warning("Skipping page {page_num}: Corrupted or unreadable")
            quality_issues.append("corrupted")
            continue

        page_text = page_data["text"]

        # Handle very small pages (likely OCR candidates)
        if page_data.get("likely_scanned", False):
            quality_issues.append("likely_scanned")
            if page_data["char_count"] < 20:
                logger.info(
                    "Page {page_num} has very little text ({page_data['char_count']} chars) - may need OCR"
                )
                quality_issues.append("needs_ocr")

        # Handle very large pages
        if page_data.get("is_very_large", False):
            quality_issues.append("very_large")
            logger.debug(
                "Page {page_num} is very large ({page_data['char_count']:,} chars) - will create many chunks"
            )

        # Chunk this page's text
        page_chunks = chunk_text(page_text, chunk_size, chunk_overlap)
        processed_pages += 1

        for chunk_content in page_chunks:
            chunk_data.append(
                {
                    "text": chunk_content,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "source_pages": [page_num],
                    "char_count": len(chunk_content),
                    "has_scanned_content": page_data.get("likely_scanned", False),
                    "has_images": page_data.get("has_images", False),
                    "quality_issues": quality_issues.copy(),
                }
            )
            chunk_id += 1

    # Enhanced logging with recommendations
    total_pages = len(pages_data)
    logger.info(
        "Chunking complete: Created {len(chunk_data)} chunks from {processed_pages}/{total_pages} pages "
        "({skipped_pages} pages skipped due to quality issues)"
    )

    # Quality recommendations
    scanned_chunks = sum(1 for chunk in chunk_data if chunk["has_scanned_content"])
    if scanned_chunks > 0:
        logger.info("💡 {scanned_chunks} chunks may have OCR-quality text")

    return chunk_data


# -----------------------------------------------------------------------------
# Embedding Generation
# -----------------------------------------------------------------------------

# Embedding model configuration
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dimensions, fast and efficient
ALTERNATIVE_MODELS = {
    "high_quality": "all-mpnet-base-v2",  # 768 dimensions, better quality
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # 384 dimensions, multilingual
    "qa_optimized": "multi-qa-MiniLM-L6-cos-v1",  # 384 dimensions, Q&A optimized
}

# Global model instance (singleton pattern)
_model: Optional[SentenceTransformer] = None
_model_name: str = DEFAULT_MODEL_NAME


def _get_model(model_name: str = None) -> SentenceTransformer:
    """
    Get or initialize the embedding model with comprehensive error handling.

    Args:
        model_name (str, optional): Name of the model to load. Defaults to DEFAULT_MODEL_NAME.

    Returns:
        SentenceTransformer: The loaded model instance.

    Raises:
        Exception: If model loading fails.
    """
    global _model, _model_name

    # Use default model if none specified
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    # Return existing model if same model requested
    if _model is not None and _model_name == model_name:
        return _model

    try:
        logger.info("🔄 Loading embedding model: {model_name}")

        # Load the model with device auto-detection
        _model = SentenceTransformer(model_name)
        _model_name = model_name

        # Log model information
        device = _model.device
        max_seq_length = _model.max_seq_length
        embedding_dim = _model.get_sentence_embedding_dimension()

        logger.info("✅ Model loaded successfully:")
        logger.info("   📋 Model: {model_name}")
        logger.info("   🖥️  Device: {device}")
        logger.info("   📏 Max sequence length: {max_seq_length}")
        logger.info("   📊 Embedding dimensions: {embedding_dim}")

        return _model

    except Exception as e:
        logger.error("❌ Failed to load embedding model '{model_name}': {e}")

        # Try fallback to default model if different model failed
        if model_name != DEFAULT_MODEL_NAME:
            logger.info("🔄 Attempting fallback to default model: {DEFAULT_MODEL_NAME}")
            try:
                _model = SentenceTransformer(DEFAULT_MODEL_NAME)
                _model_name = DEFAULT_MODEL_NAME
                logger.info("✅ Fallback model loaded: {DEFAULT_MODEL_NAME}")
                return _model
            except Exception as fallback_error:
                logger.error("❌ Fallback model also failed: {fallback_error}")

        raise Exception("Could not load any embedding model. Original error: {e}")


def get_model_info() -> dict:
    """
    Get information about the currently loaded model.

    Returns:
        dict: Model information including name, dimensions, device, etc.
    """
    try:
        model = _get_model()
        return {
            "model_name": _model_name,
            "embedding_dimensions": model.get_sentence_embedding_dimension(),
            "max_sequence_length": model.max_seq_length,
            "device": str(model.device),
            "available_models": {"current": DEFAULT_MODEL_NAME, **ALTERNATIVE_MODELS},
        }
    except Exception as e:
        return {
            "error": str(e),
            "available_models": {"default": DEFAULT_MODEL_NAME, **ALTERNATIVE_MODELS},
        }


def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    """
    Generates vector embeddings for a list of text chunks using Sentence-Transformers.

    Uses all-MiniLM-L6-v2 model (384 dimensions) for fast, high-quality embeddings.
    Implements batch processing for efficiency and comprehensive error handling.

    Args:
        chunks (List[str]): A list of text chunks to embed.

    Returns:
        List[List[float]]: A list of embedding vectors (384 dimensions each).
    """
    if not chunks:
        logger.warning("No chunks provided for embedding generation.")
        return []

    try:
        model = _get_model()

        # Log embedding generation start
        logger.info(
            "Generating embeddings for {len(chunks)} chunks using {_model_name}"
        )

        # Generate embeddings with batch processing
        embeddings = model.encode(
            chunks,
            show_progress_bar=len(chunks) > 10,  # Show progress for larger batches
            batch_size=32,  # Optimal batch size for most systems
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )

        # Validate embeddings
        if len(embeddings) != len(chunks):
            raise ValueError(
                "Embedding count mismatch: {len(embeddings)} != {len(chunks)}"
            )

        # Convert to list format and validate dimensions
        embeddings_list = embeddings.tolist()
        expected_dim = 384  # all-MiniLM-L6-v2 dimensions

        for i, emb in enumerate(embeddings_list):
            if len(emb) != expected_dim:
                raise ValueError(
                    "Unexpected embedding dimension at index {i}: {len(emb)} != {expected_dim}"
                )

        logger.info(
            "✅ Successfully generated {len(embeddings_list)} embeddings ({expected_dim}D)"
        )
        return embeddings_list

    except Exception as e:
        logger.error("❌ Failed to generate embeddings: {e}")
        return []


def generate_embeddings_with_metadata(chunk_data: List[dict]) -> List[List[float]]:
    """
    Generates embeddings for chunks with metadata, handling edge cases.

    Args:
        chunk_data (List[dict]): List of chunk dictionaries with 'text' field.

    Returns:
        List[List[float]]: List of embedding vectors.
    """
    if not chunk_data:
        logger.warning("No chunk data provided for embedding generation.")
        return []

    # Extract text from chunk data and handle empty chunks
    texts = []
    valid_indices = []

    for i, chunk in enumerate(chunk_data):
        text = chunk.get("text", "").strip()
        if text:
            texts.append(text)
            valid_indices.append(i)
        else:
            logger.warning("Skipping empty chunk at index {i}")

    if not texts:
        logger.error("No valid text found in chunk data")
        return []

    # Generate embeddings for valid texts
    embeddings = generate_embeddings(texts)

    if not embeddings:
        return []

    # Create full embedding list with None for invalid chunks
    full_embeddings = [None] * len(chunk_data)
    for i, valid_idx in enumerate(valid_indices):
        full_embeddings[valid_idx] = embeddings[i]

    # Filter out None values and return only valid embeddings
    valid_embeddings = [emb for emb in full_embeddings if emb is not None]

    logger.info(
        "Generated embeddings for {len(valid_embeddings)}/{len(chunk_data)} chunks"
    )
    return valid_embeddings


# -----------------------------------------------------------------------------
# Vector Database (Qdrant) - Enhanced Step 5 Implementation
# -----------------------------------------------------------------------------

# Initialize the Qdrant client
# For local development, Qdrant can be run via Docker.
_client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "research_documents"
VECTOR_SIZE = 384  # Based on all-MiniLM-L6-v2 model


def create_collection_if_not_exists() -> None:
    """Creates the Qdrant collection if it doesn't already exist."""
    try:
        collections = _client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        if COLLECTION_NAME not in collection_names:
            _client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE, distance=models.Distance.COSINE
                ),
            )
            logger.info("Collection '{COLLECTION_NAME}' created.")
    except Exception as e:
        logger.error(
            "Could not connect to or create Qdrant collection. Is Qdrant running? Error: {e}"
        )


def get_collection_info() -> dict:
    """
    Get comprehensive information about the Qdrant collection.

    Returns:
        dict: Collection information including stats, configuration, and health.
    """
    try:
        # Get collection info
        collection_info = _client.get_collection(COLLECTION_NAME)

        # Get collection stats
        collection_info = _client.get_collection(
            collection_name=COLLECTION_NAME,
        )

        # Count points by document
        all_points = _client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,  # Adjust based on expected collection size
            with_payload=["document_id"],
            with_vectors=False,
        )

        document_counts = {}
        for point in all_points[0]:
            doc_id = point.payload.get("document_id", "unknown")
            document_counts[doc_id] = document_counts.get(doc_id, 0) + 1

        return {
            "status": "success",
            "collection_name": COLLECTION_NAME,
            "vectors_count": collection_info.vectors_count or 0,
            "vector_size": collection_info.config.params.vectors.size,
            "distance_metric": collection_info.config.params.vectors.distance.value,
            "documents_count": len(document_counts),
            "document_breakdown": document_counts,
            "indexed": collection_info.status == models.CollectionStatus.GREEN,
            "config": {
                "vector_size": VECTOR_SIZE,
                "distance": "COSINE",
                "hnsw_config": collection_info.config.hnsw_config,
                "optimizer_config": collection_info.config.optimizer_config,
            },
        }

    except Exception as e:
        logger.error("Failed to get collection info: {e}")
        return {"status": "error", "error": str(e), "collection_name": COLLECTION_NAME}


def store_embeddings(
    embeddings: List[List[float]],
    chunks: List[str],
    document_id: str,
) -> None:
    """
    Stores text chunks and their embeddings in the Qdrant collection.
    (Backward compatibility version - converts chunks to simple format)

    Args:
        embeddings (List[List[float]]): The list of embedding vectors.
        chunks (List[str]): The list of original text chunks.
        document_id (str): A unique identifier for the source document.
    """
    # Convert simple chunks to chunk_data format for compatibility
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append(
            {
                "text": chunk,
                "chunk_id": i,
                "source_pages": [],  # No page info in simple format
                "char_count": len(chunk),
            }
        )

    store_embeddings_with_metadata(embeddings, chunk_data, document_id)


def store_embeddings_with_metadata(
    embeddings: List[List[float]],
    chunk_data: List[dict],
    document_id: str,
    tags: Optional[List[str]] = None,
    custom_metadata: Optional[dict] = None,
) -> bool:
    """
    Enhanced Step 5: Store embeddings with comprehensive metadata in Qdrant.

    Creates payload including:
    - Document ID
    - Chunk ID
    - Page number(s)
    - Original chunk text
    - Tags and custom metadata
    - Quality indicators
    - Processing metadata

    Args:
        embeddings (List[List[float]]): The list of embedding vectors.
        chunk_data (List[dict]): List of chunk data with metadata.
        document_id (str): A unique identifier for the source document.
        tags (List[str], optional): Tags for categorization and filtering.
        custom_metadata (dict, optional): Additional custom metadata.

    Returns:
        bool: True if storage succeeded, False otherwise.
    """
    if not embeddings:
        logger.warning("No embeddings to store.")
        return False

    if len(embeddings) != len(chunk_data):
        logger.error(
            "Embedding count ({len(embeddings)}) doesn't match chunk count ({len(chunk_data)})"
        )
        return False

    try:
        points: List[models.PointStruct] = []

        current_timestamp = int(time.time())

        for i, chunk_info in enumerate(chunk_data):
            point_id = str(uuid.uuid4())

            # Enhanced payload with comprehensive metadata
            payload = {
                # Core identification
                "document_id": document_id,
                "chunk_id": chunk_info.get("chunk_id", i),
                "point_id": point_id,
                # Content
                "text": chunk_info["text"],
                "char_count": chunk_info.get("char_count", len(chunk_info["text"])),
                # Page information
                "source_pages": chunk_info.get("source_pages", []),
                "page_count": len(chunk_info.get("source_pages", [])),
                # Quality indicators
                "has_scanned_content": chunk_info.get("has_scanned_content", False),
                "has_images": chunk_info.get("has_images", False),
                "quality_issues": chunk_info.get("quality_issues", []),
                "quality_score": len(chunk_info.get("quality_issues", []))
                == 0,  # True if no issues
                # Token information (if available)
                "token_count": chunk_info.get("token_count"),
                "start_token": chunk_info.get("start_token"),
                "end_token": chunk_info.get("end_token"),
                # Processing metadata
                "created_at": current_timestamp,
                "embedding_model": _model_name,
                "embedding_dimensions": len(embeddings[i]),
                # Tags and custom metadata
                "tags": tags or [],
                "custom_metadata": custom_metadata or {},
                # Search optimization fields
                "text_length_category": _categorize_text_length(
                    len(chunk_info["text"])
                ),
                "has_quality_issues": len(chunk_info.get("quality_issues", [])) > 0,
                "is_scanned_content": chunk_info.get("has_scanned_content", False)
                or chunk_info.get("likely_scanned", False),
            }

            # Validate embedding dimensions
            if len(embeddings[i]) != VECTOR_SIZE:
                logger.error(
                    "Embedding dimension mismatch: expected {VECTOR_SIZE}, got {len(embeddings[i])}"
                )
                continue

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embeddings[i],
                    payload=payload,
                )
            )

        if not points:
            logger.error("No valid points to store")
            return False

        # Upsert with wait=True to ensure consistency
        _client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

        logger.info(
            "✅ Successfully stored {len(points)} points for document {document_id} "
            "with comprehensive metadata (tags: {len(tags or [])}, "
            "custom_metadata: {len(custom_metadata or {})} fields)"
        )
        return True

    except Exception as e:
        logger.error("❌ Failed to store embeddings in Qdrant: {e}")
        return False


def _categorize_text_length(char_count: int) -> str:
    """Categorize text length for search optimization."""
    if char_count < 100:
        return "short"
    elif char_count < 500:
        return "medium"
    elif char_count < 1500:
        return "long"
    else:
        return "very_long"


def search_similar_chunks(
    query_text: str,
    limit: int = 10,
    score_threshold: float = 0.7,
    filter_conditions: Optional[dict] = None,
    include_metadata: bool = True,
) -> List[dict]:
    """
    Search for similar chunks using semantic similarity.

    Args:
        query_text (str): The search query text.
        limit (int): Maximum number of results to return.
        score_threshold (float): Minimum similarity score (0.0 to 1.0).
        filter_conditions (dict, optional): Qdrant filter conditions.
        include_metadata (bool): Whether to include full metadata in results.

    Returns:
        List[dict]: List of similar chunks with scores and metadata.
    """
    try:
        # Generate embedding for query
        query_embeddings = generate_embeddings([query_text])
        if not query_embeddings:
            logger.error("Failed to generate query embedding")
            return []

        query_vector = query_embeddings[0]

        # Build filter if provided
        search_filter = None
        if filter_conditions:
            search_filter = models.Filter(**filter_conditions)

        # Search for similar vectors
        search_results = _client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=search_filter,
            with_payload=True,
            with_vectors=False,
        )

        # Format results
        results = []
        for result in search_results:
            formatted_result = {
                "id": result.id,
                "score": float(result.score),
                "text": result.payload.get("text", ""),
                "document_id": result.payload.get("document_id"),
                "chunk_id": result.payload.get("chunk_id"),
                "source_pages": result.payload.get("source_pages", []),
            }

            if include_metadata:
                formatted_result.update(
                    {
                        "char_count": result.payload.get("char_count"),
                        "has_scanned_content": result.payload.get(
                            "has_scanned_content", False
                        ),
                        "quality_issues": result.payload.get("quality_issues", []),
                        "tags": result.payload.get("tags", []),
                        "created_at": result.payload.get("created_at"),
                        "embedding_model": result.payload.get("embedding_model"),
                        "custom_metadata": result.payload.get("custom_metadata", {}),
                    }
                )

            results.append(formatted_result)

        logger.info(
            "Found {len(results)} similar chunks for query (threshold: {score_threshold})"
        )
        return results

    except Exception as e:
        logger.error("Failed to search similar chunks: {e}")
        return []


def search_by_document(
    document_id: str, limit: int = 100, include_text: bool = True
) -> List[dict]:
    """
    Retrieve all chunks for a specific document.

    Args:
        document_id (str): The document ID to search for.
        limit (int): Maximum number of chunks to return.
        include_text (bool): Whether to include chunk text.

    Returns:
        List[dict]: List of chunks for the document.
    """
    try:
        # Search with document ID filter
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id)
                )
            ]
        )

        search_results = _client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=filter_condition,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        # Format results
        results = []
        for point in search_results[0]:
            result = {
                "id": point.id,
                "chunk_id": point.payload.get("chunk_id"),
                "source_pages": point.payload.get("source_pages", []),
                "char_count": point.payload.get("char_count"),
                "quality_issues": point.payload.get("quality_issues", []),
                "created_at": point.payload.get("created_at"),
            }

            if include_text:
                result["text"] = point.payload.get("text", "")

            results.append(result)

        # Sort by chunk_id for consistent ordering
        results.sort(key=lambda x: x.get("chunk_id", 0))

        logger.info("Retrieved {len(results)} chunks for document {document_id}")
        return results

    except Exception as e:
        logger.error("Failed to search by document: {e}")
        return []


def delete_document(document_id: str) -> bool:
    """
    Delete all chunks for a specific document.

    Args:
        document_id (str): The document ID to delete.

    Returns:
        bool: True if deletion succeeded, False otherwise.
    """
    try:
        # Get all points for the document
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id)
                )
            ]
        )

        # Get point IDs to delete
        search_results = _client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=filter_condition,
            limit=10000,  # Large limit to get all chunks
            with_payload=False,
            with_vectors=False,
        )

        point_ids = [point.id for point in search_results[0]]

        if not point_ids:
            logger.warning("No points found for document {document_id}")
            return True

        # Delete points
        _client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )

        logger.info("✅ Deleted {len(point_ids)} points for document {document_id}")
        return True

    except Exception as e:
        logger.error("❌ Failed to delete document {document_id}: {e}")
        return False


def get_collection_stats() -> dict:
    """
    Get detailed statistics about the vector database collection.

    Returns:
        dict: Comprehensive collection statistics.
    """
    try:
        # Get basic collection info
        collection_info = _client.get_collection(COLLECTION_NAME)

        # Get sample of points for analysis
        sample_points = _client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        # Analyze sample
        stats = {
            "collection_name": COLLECTION_NAME,
            "total_vectors": collection_info.vectors_count or 0,
            "vector_size": VECTOR_SIZE,
            "distance_metric": "COSINE",
            "status": (
                collection_info.status.value if collection_info.status else "unknown"
            ),
        }

        if sample_points[0]:
            # Document analysis
            documents = set()
            models_used = set()
            quality_issues_count = 0
            scanned_content_count = 0
            total_chars = 0
            page_counts = []

            for point in sample_points[0]:
                payload = point.payload
                documents.add(payload.get("document_id", "unknown"))
                models_used.add(payload.get("embedding_model", "unknown"))

                if payload.get("quality_issues"):
                    quality_issues_count += 1
                if payload.get("has_scanned_content"):
                    scanned_content_count += 1

                total_chars += payload.get("char_count", 0)
                page_counts.append(len(payload.get("source_pages", [])))

            stats.update(
                {
                    "documents_count": len(documents),
                    "embedding_models": list(models_used),
                    "sample_size": len(sample_points[0]),
                    "quality_issues_percentage": round(
                        (quality_issues_count / len(sample_points[0])) * 100, 2
                    ),
                    "scanned_content_percentage": round(
                        (scanned_content_count / len(sample_points[0])) * 100, 2
                    ),
                    "avg_chars_per_chunk": round(total_chars / len(sample_points[0])),
                    "avg_pages_per_chunk": round(
                        sum(page_counts) / len(page_counts), 2
                    ),
                }
            )

        return stats

    except Exception as e:
        logger.error("Failed to get collection stats: {e}")
        return {"error": str(e)}


# -----------------------------------------------------------------------------
# Enhanced Pipeline Functions
# -----------------------------------------------------------------------------


def process_pdf_with_page_structure(
    file_path: str,
    document_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    use_token_chunking: bool = False,
) -> bool:
    """
    Enhanced PDF processing pipeline that preserves page structure and metadata.

    Args:
        file_path (str): Path to the PDF file
        document_id (str): Unique identifier for the document
        chunk_size (int): Size of text chunks (characters or tokens)
        chunk_overlap (int): Overlap between chunks (characters or tokens)
        use_token_chunking (bool): If True, use token-based chunking; if False, use character-based

    Returns:
        bool: True if processing succeeded, False otherwise
    """
    try:
        logger.info("Starting enhanced processing for {file_path}")

        # 1. Extract text with page structure
        pages_data = extract_pages_from_pdf(file_path)
        if not pages_data:
            logger.error("No pages extracted from PDF")
            return False

        # 2. Chunk text while preserving page metadata
        if use_token_chunking:
            logger.info("Using token-based chunking for enhanced accuracy")
            # For token-based chunking, we need to process the full text
            full_text = ""
            for page_data in pages_data:
                if not page_data["is_empty"] and not page_data.get(
                    "is_corrupted", False
                ):
                    full_text += page_data["text"] + "\n\n--- Page Break ---\n\n"

            token_chunks = chunk_text_by_tokens(full_text, chunk_size, chunk_overlap)
            # Convert token chunks to page-aware format
            chunk_data = []
            for token_chunk in token_chunks:
                chunk_data.append(
                    {
                        "text": token_chunk["text"],
                        "chunk_id": token_chunk["chunk_id"],
                        "document_id": document_id,
                        "source_pages": [],  # Token-based doesn't track specific pages
                        "char_count": token_chunk["char_count"],
                        "token_count": token_chunk["token_count"],
                        "start_token": token_chunk["start_token"],
                        "end_token": token_chunk["end_token"],
                        "has_scanned_content": False,
                        "has_images": False,
                        "quality_issues": [],
                    }
                )
        else:
            chunk_data = chunk_text_with_pages(
                pages_data, chunk_size, chunk_overlap, document_id
            )

        if not chunk_data:
            logger.error("No chunks created from pages")
            return False

        # 3. Generate embeddings for chunks
        chunk_texts = [chunk["text"] for chunk in chunk_data]
        embeddings = generate_embeddings(chunk_texts)
        if not embeddings:
            logger.error("No embeddings generated")
            return False

        # 4. Store embeddings with metadata
        store_embeddings_with_metadata(embeddings, chunk_data, document_id)

        logger.info("Enhanced processing completed for document {document_id}")
        return True

    except Exception as e:
        logger.error("Enhanced processing failed for {file_path}: {e}")
        return False


def analyze_pdf_quality(file_path: str) -> dict:
    """
    Comprehensive PDF quality analysis with detailed edge case reporting.

    Args:
        file_path (str): Path to the PDF file

    Returns:
        dict: Detailed quality analysis report
    """
    try:
        pages_data = extract_pages_from_pdf(file_path)
        if not pages_data:
            return {
                "status": "failed",
                "error": "Could not extract pages from PDF",
                "recommendations": [
                    "Check if PDF is valid, not corrupted, and not password-protected"
                ],
            }

        # Analyze page quality
        total_pages = len(pages_data)
        empty_pages = sum(1 for p in pages_data if p["is_empty"])
        corrupted_pages = sum(1 for p in pages_data if p.get("is_corrupted", False))
        scanned_pages = sum(1 for p in pages_data if p.get("likely_scanned", False))
        image_heavy_pages = sum(
            1 for p in pages_data if p.get("likely_image_only", False)
        )
        very_large_pages = sum(1 for p in pages_data if p.get("is_very_large", False))

        total_chars = sum(p["char_count"] for p in pages_data)
        avg_chars_per_page = total_chars / max(
            1, total_pages - empty_pages - corrupted_pages
        )

        # Quality assessment
        content_pages = total_pages - empty_pages - corrupted_pages
        quality_score = content_pages / total_pages if total_pages > 0 else 0

        # Generate recommendations
        recommendations = []
        issues = []

        if corrupted_pages > 0:
            issues.append("{corrupted_pages} corrupted pages")
            recommendations.append(
                "PDF may be damaged - consider re-obtaining the source file"
            )

        if scanned_pages > 0:
            issues.append("{scanned_pages} likely scanned pages")
            recommendations.append(
                "Consider using OCR (Optical Character Recognition) for scanned pages"
            )

        if empty_pages > total_pages * 0.3:
            issues.append(
                "{empty_pages} empty pages ({empty_pages/total_pages*100:.1f}%)"
            )
            recommendations.append(
                "High number of empty pages - check PDF content quality"
            )

        if avg_chars_per_page < 100:
            issues.append("Very low text density")
            recommendations.append(
                "Document may be primarily images - OCR processing recommended"
            )

        if very_large_pages > 0:
            issues.append("{very_large_pages} very large pages")
            recommendations.append(
                "Large pages will create many chunks - consider adjusting chunk size"
            )

        # Overall assessment
        if quality_score >= 0.9:
            status = "excellent"
        elif quality_score >= 0.7:
            status = "good"
        elif quality_score >= 0.5:
            status = "fair"
        else:
            status = "poor"

        analysis = {
            "status": status,
            "quality_score": round(quality_score, 2),
            "total_pages": total_pages,
            "content_pages": content_pages,
            "empty_pages": empty_pages,
            "corrupted_pages": corrupted_pages,
            "scanned_pages": scanned_pages,
            "image_heavy_pages": image_heavy_pages,
            "very_large_pages": very_large_pages,
            "total_characters": total_chars,
            "avg_chars_per_page": round(avg_chars_per_page),
            "issues": issues,
            "recommendations": recommendations,
        }

        # Log summary
        logger.info(
            "PDF Quality Analysis: {status.upper()} (score: {quality_score:.2f})"
        )
        if issues:
            logger.warning("Quality issues found: {', '.join(issues)}")
        if recommendations:
            logger.info("Recommendations: {'; '.join(recommendations)}")

        return analysis

    except Exception as e:
        logger.error("Failed to analyze PDF quality for {file_path}: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "recommendations": ["Unable to analyze PDF - check file accessibility"],
        }
