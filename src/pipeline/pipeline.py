from typing import List, Optional
import uuid
import os

from logging_config import logger

# Text extraction
import fitz  # PyMuPDF

# Embedding generation
from sentence_transformers import SentenceTransformer

# Vector DB (Qdrant)
from qdrant_client import QdrantClient, models


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
            text += page_data['text']
            if page_data['text'].strip():  # Add page separator if page has content
                text += "\n\n--- Page Break ---\n\n"
        
        logger.info(f"Successfully extracted text from {file_path} ({len(pages_data)} pages)")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from {file_path}. Error: {e}")
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
    import time
    
    try:
        # Check if file exists and is readable
        if not os.path.exists(file_path):
            logger.error(f"PDF file not found: {file_path}")
            return []
        
        if os.path.getsize(file_path) == 0:
            logger.error(f"PDF file is empty: {file_path}")
            return []
        
        start_time = time.time()
        
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            if "password" in str(e).lower() or "encrypted" in str(e).lower():
                logger.error(f"PDF is password-protected or encrypted: {file_path}")
                return []
            elif "damaged" in str(e).lower() or "corrupt" in str(e).lower():
                logger.error(f"PDF appears to be corrupted: {file_path}")
                return []
            else:
                logger.error(f"Failed to open PDF {file_path}: {e}")
                return []
        
        # Check if PDF is valid
        if doc.page_count == 0:
            logger.warning(f"PDF has no pages: {file_path}")
            return []
        
        logger.info(f"Processing PDF with {doc.page_count} pages: {file_path}")
        
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
                    logger.error(f"Failed to extract text from page {page_num + 1}: {e}")
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
                likely_scanned = has_images and char_count < 100  # Images but little text
                is_corrupted = False
                is_very_large = char_count > 50000  # Very large page (>50k chars)
                
                # Detect potential OCR candidates (scanned pages)
                if has_images and char_count == 0:
                    likely_scanned = True
                    scanned_pages += 1
                    logger.warning(f"Page {page_num + 1} appears to be scanned (images but no text) - OCR may be needed")
                elif likely_scanned:
                    scanned_pages += 1
                    logger.info(f"Page {page_num + 1} may be scanned ({image_count} images, {char_count} chars)")
                
                page_processing_time = (time.time() - page_start_time) * 1000
                
                page_data = {
                    'page_num': page_num + 1,
                    'text': page_text,
                    'char_count': char_count,
                    'is_empty': is_empty,
                    'likely_image_only': likely_image_only,
                    'likely_scanned': likely_scanned,
                    'has_images': has_images,
                    'image_count': image_count,
                    'is_corrupted': is_corrupted,
                    'is_very_large': is_very_large,
                    'processing_time_ms': round(page_processing_time, 2)
                }
                
                pages_data.append(page_data)
                
                # Enhanced edge case logging
                if is_empty:
                    empty_pages += 1
                    if has_images:
                        logger.warning(f"Page {page_num + 1} is empty but contains {image_count} image(s) - likely scanned")
                    else:
                        logger.warning(f"Page {page_num + 1} is completely empty")
                elif likely_image_only:
                    image_heavy_pages += 1
                    logger.info(f"Page {page_num + 1} is image-heavy ({image_count} images, {char_count} chars)")
                elif is_very_large:
                    very_large_pages += 1
                    logger.info(f"Page {page_num + 1} is very large ({char_count:,} characters)")
                
                # Log slow pages
                if page_processing_time > 1000:  # > 1 second
                    logger.warning(f"Page {page_num + 1} took {page_processing_time:.0f}ms to process")
                    
            except Exception as e:
                corrupted_pages += 1
                logger.error(f"Failed to process page {page_num + 1}: {e}")
                
                # Add corrupted page entry
                page_data = {
                    'page_num': page_num + 1,
                    'text': "",
                    'char_count': 0,
                    'is_empty': True,
                    'likely_image_only': False,
                    'likely_scanned': False,
                    'has_images': False,
                    'image_count': 0,
                    'is_corrupted': True,
                    'is_very_large': False,
                    'processing_time_ms': 0
                }
                pages_data.append(page_data)
        
        # Comprehensive summary logging
        total_pages = len(pages_data)
        content_pages = total_pages - empty_pages - corrupted_pages
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"PDF analysis complete ({processing_time:.0f}ms): "
            f"{total_pages} total pages, "
            f"{content_pages} with content, "
            f"{empty_pages} empty, "
            f"{image_heavy_pages} image-heavy, "
            f"{scanned_pages} likely scanned, "
            f"{very_large_pages} very large, "
            f"{corrupted_pages} corrupted"
        )
        
        # Recommendations based on analysis
        if scanned_pages > 0:
            logger.info(f"💡 Recommendation: {scanned_pages} pages may benefit from OCR processing")
        
        if corrupted_pages > 0:
            logger.warning(f"⚠️  {corrupted_pages} pages could not be processed - PDF may be damaged")
        
        if empty_pages > total_pages * 0.5:
            logger.warning(f"⚠️  More than 50% of pages are empty - check PDF quality")
        
        return pages_data
        
    except Exception as e:
        logger.error(f"Critical failure extracting pages from {file_path}: {e}")
        return []


# -----------------------------------------------------------------------------
# Text Chunking
# -----------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
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
        logger.warning(f"Overlap ({chunk_overlap}) >= chunk_size ({chunk_size}). Setting overlap to {chunk_size // 2}")
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
            chunk = text[start:start + page_break_pos]
        else:
            # Look for paragraph boundary (double newline)
            paragraph_end = chunk.rfind("\n\n")
            if paragraph_end > chunk_size * 0.6:  # Only if paragraph end is reasonably far
                chunk = text[start:start + paragraph_end + 2]  # Include the double newline
            else:
                # Look for sentence boundary
                sentence_end = max(
                    chunk.rfind(". "),
                    chunk.rfind("! "),
                    chunk.rfind("? "),
                    chunk.rfind(".\n"),
                    chunk.rfind("!\n"),
                    chunk.rfind("?\n")
                )
                if sentence_end > chunk_size * 0.7:  # Only if sentence end is reasonably far
                    chunk = text[start:start + sentence_end + 1]
        
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        
        # Calculate next start position
        actual_chunk_length = len(chunk)
        step_size = max(1, min(actual_chunk_length, chunk_size) - chunk_overlap)
        start += step_size

    return chunks


def chunk_text_with_pages(pages_data: List[dict], chunk_size: int = 1000, chunk_overlap: int = 100, document_id: str = None) -> List[dict]:
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
        page_num = page_data['page_num']
        quality_issues = []
        
        # Skip problematic pages with detailed logging
        if page_data['is_empty']:
            skipped_pages += 1
            if page_data.get('has_images', False):
                logger.info(f"Skipping page {page_num}: Empty but has images (likely scanned)")
                quality_issues.append("scanned_no_text")
            else:
                logger.debug(f"Skipping page {page_num}: Completely empty")
                quality_issues.append("empty")
            continue
            
        if page_data.get('is_corrupted', False):
            skipped_pages += 1
            logger.warning(f"Skipping page {page_num}: Corrupted or unreadable")
            quality_issues.append("corrupted")
            continue
            
        page_text = page_data['text']
        
        # Handle very small pages (likely OCR candidates)
        if page_data.get('likely_scanned', False):
            quality_issues.append("likely_scanned")
            if page_data['char_count'] < 20:
                logger.info(f"Page {page_num} has very little text ({page_data['char_count']} chars) - may need OCR")
                quality_issues.append("needs_ocr")
        
        # Handle very large pages
        if page_data.get('is_very_large', False):
            quality_issues.append("very_large")
            logger.debug(f"Page {page_num} is very large ({page_data['char_count']:,} chars) - will create many chunks")
        
        # Chunk this page's text
        page_chunks = chunk_text(page_text, chunk_size, chunk_overlap)
        processed_pages += 1
        
        for chunk_content in page_chunks:
            chunk_data.append({
                'text': chunk_content,
                'chunk_id': chunk_id,
                'document_id': document_id,
                'source_pages': [page_num],
                'char_count': len(chunk_content),
                'has_scanned_content': page_data.get('likely_scanned', False),
                'has_images': page_data.get('has_images', False),
                'quality_issues': quality_issues.copy()
            })
            chunk_id += 1
    
    # Enhanced logging with recommendations
    total_pages = len(pages_data)
    logger.info(
        f"Chunking complete: Created {len(chunk_data)} chunks from {processed_pages}/{total_pages} pages "
        f"({skipped_pages} pages skipped due to quality issues)"
    )
    
    # Quality recommendations
    scanned_chunks = sum(1 for chunk in chunk_data if chunk['has_scanned_content'])
    if scanned_chunks > 0:
        logger.info(f"💡 {scanned_chunks} chunks may have OCR-quality text")
    
    return chunk_data


# -----------------------------------------------------------------------------
# Embedding Generation
# -----------------------------------------------------------------------------

# Load a pre-trained model once at module import time.
# all-MiniLM-L6-v2 is a good starting point for speed/quality.
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    """
    Generates vector embeddings for a list of text chunks.

    Args:
        chunks (List[str]): A list of text chunks.

    Returns:
        List[List[float]]: A list of embedding vectors.
    """
    if not chunks:
        logger.warning("No chunks provided for embedding generation.")
        return []

    try:
        model = _get_model()
        embeddings = model.encode(chunks, show_progress_bar=True)
        logger.info(f"Successfully generated {len(embeddings)} embeddings.")
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Failed to generate embeddings. Error: {e}")
        return []


# -----------------------------------------------------------------------------
# Vector Database (Qdrant)
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
                vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
            )
            logger.info(f"Collection '{COLLECTION_NAME}' created.")
    except Exception as e:
        logger.error(
            f"Could not connect to or create Qdrant collection. Is Qdrant running? Error: {e}"
        )


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
        chunk_data.append({
            'text': chunk,
            'chunk_id': i,
            'source_pages': [],  # No page info in simple format
            'char_count': len(chunk)
        })
    
    store_embeddings_with_metadata(embeddings, chunk_data, document_id)


def store_embeddings_with_metadata(
    embeddings: List[List[float]],
    chunk_data: List[dict],
    document_id: str,
) -> None:
    """
    Stores text chunks with metadata and their embeddings in the Qdrant collection.

    Args:
        embeddings (List[List[float]]): The list of embedding vectors.
        chunk_data (List[dict]): List of chunk data with metadata.
        document_id (str): A unique identifier for the source document.
    """
    if not embeddings:
        logger.warning("No embeddings to store.")
        return

    try:
        points: List[models.PointStruct] = []
        for i, chunk_info in enumerate(chunk_data):
            point_id = str(uuid.uuid4())
            
            payload = {
                "text": chunk_info['text'],
                "document_id": document_id,
                "chunk_id": chunk_info['chunk_id'],
                "char_count": chunk_info['char_count'],
                "source_pages": chunk_info.get('source_pages', []),
                "has_scanned_content": chunk_info.get('has_scanned_content', False),
                "has_images": chunk_info.get('has_images', False),
                "quality_issues": chunk_info.get('quality_issues', []),
            }
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embeddings[i],
                    payload=payload,
                )
            )

        _client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        logger.info(
            f"Successfully stored {len(points)} points for document {document_id} "
            f"with page metadata."
        )
    except Exception as e:
        logger.error(f"Failed to store embeddings in Qdrant. Error: {e}")


# -----------------------------------------------------------------------------
# Enhanced Pipeline Functions
# -----------------------------------------------------------------------------

def process_pdf_with_page_structure(
    file_path: str,
    document_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> bool:
    """
    Enhanced PDF processing pipeline that preserves page structure and metadata.
    
    Args:
        file_path (str): Path to the PDF file
        document_id (str): Unique identifier for the document
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    try:
        logger.info(f"Starting enhanced processing for {file_path}")
        
        # 1. Extract text with page structure
        pages_data = extract_pages_from_pdf(file_path)
        if not pages_data:
            logger.error("No pages extracted from PDF")
            return False
        
        # 2. Chunk text while preserving page metadata
        chunk_data = chunk_text_with_pages(pages_data, chunk_size, chunk_overlap, document_id)
        if not chunk_data:
            logger.error("No chunks created from pages")
            return False
        
        # 3. Generate embeddings for chunks
        chunk_texts = [chunk['text'] for chunk in chunk_data]
        embeddings = generate_embeddings(chunk_texts)
        if not embeddings:
            logger.error("No embeddings generated")
            return False
        
        # 4. Store embeddings with metadata
        store_embeddings_with_metadata(embeddings, chunk_data, document_id)
        
        logger.info(f"Enhanced processing completed for document {document_id}")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced processing failed for {file_path}: {e}")
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
                "recommendations": ["Check if PDF is valid, not corrupted, and not password-protected"]
            }
        
        # Analyze page quality
        total_pages = len(pages_data)
        empty_pages = sum(1 for p in pages_data if p['is_empty'])
        corrupted_pages = sum(1 for p in pages_data if p.get('is_corrupted', False))
        scanned_pages = sum(1 for p in pages_data if p.get('likely_scanned', False))
        image_heavy_pages = sum(1 for p in pages_data if p.get('likely_image_only', False))
        very_large_pages = sum(1 for p in pages_data if p.get('is_very_large', False))
        
        total_chars = sum(p['char_count'] for p in pages_data)
        avg_chars_per_page = total_chars / max(1, total_pages - empty_pages - corrupted_pages)
        
        # Quality assessment
        content_pages = total_pages - empty_pages - corrupted_pages
        quality_score = content_pages / total_pages if total_pages > 0 else 0
        
        # Generate recommendations
        recommendations = []
        issues = []
        
        if corrupted_pages > 0:
            issues.append(f"{corrupted_pages} corrupted pages")
            recommendations.append("PDF may be damaged - consider re-obtaining the source file")
        
        if scanned_pages > 0:
            issues.append(f"{scanned_pages} likely scanned pages")
            recommendations.append("Consider using OCR (Optical Character Recognition) for scanned pages")
        
        if empty_pages > total_pages * 0.3:
            issues.append(f"{empty_pages} empty pages ({empty_pages/total_pages*100:.1f}%)")
            recommendations.append("High number of empty pages - check PDF content quality")
        
        if avg_chars_per_page < 100:
            issues.append("Very low text density")
            recommendations.append("Document may be primarily images - OCR processing recommended")
        
        if very_large_pages > 0:
            issues.append(f"{very_large_pages} very large pages")
            recommendations.append("Large pages will create many chunks - consider adjusting chunk size")
        
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
            "recommendations": recommendations
        }
        
        # Log summary
        logger.info(f"PDF Quality Analysis: {status.upper()} (score: {quality_score:.2f})")
        if issues:
            logger.warning(f"Quality issues found: {', '.join(issues)}")
        if recommendations:
            logger.info(f"Recommendations: {'; '.join(recommendations)}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Failed to analyze PDF quality for {file_path}: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "recommendations": ["Unable to analyze PDF - check file accessibility"]
        } 