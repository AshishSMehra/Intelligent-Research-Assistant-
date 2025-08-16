"""
LLM Service for the Intelligent Research Assistant.

This service handles interactions with language models for answer generation.
Supports multiple LLM providers: OpenAI, Ollama, Hugging Face, and fallback.
"""

import os
import time
import random
from typing import List, Dict, Any, Optional
from loguru import logger

# Try to import different LLM libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("Requests library not available. Install with: pip install requests")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.warning("Hugging Face transformers not available. Install with: pip install transformers torch")


class LLMService:
    """Service for handling LLM interactions with multiple providers."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", provider: str = "auto", api_key: Optional[str] = None):
        """
        Initialize the LLM service.
        
        Args:
            model_name: Name of the LLM model to use
            provider: LLM provider ('openai', 'ollama', 'huggingface', 'fallback', 'auto')
            api_key: API key (if using OpenAI)
        """
        self.model_name = model_name
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize provider
        self._initialize_provider()
        
        # Fallback response templates
        self.fallback_responses = [
            "Based on the available documents, I found some relevant information: {context}",
            "According to the sources, {context}",
            "The documents indicate that {context}",
            "From the available information: {context}",
            "Based on the research documents, {context}",
            "The sources suggest that {context}"
        ]
        
        logger.info(f"LLM Service initialized with provider: {self.provider}, model: {self.model_name}")
    
    def _initialize_provider(self):
        """Initialize the selected LLM provider."""
        if self.provider == "auto":
            # Auto-detect best available provider
            if OPENAI_AVAILABLE and self.api_key:
                self.provider = "openai"
                openai.api_key = self.api_key
                logger.info(f"Using OpenAI with model: {self.model_name}")
            elif self._check_ollama_available():
                self.provider = "ollama"
                logger.info(f"Using Ollama with model: {self.model_name}")
            elif HUGGINGFACE_AVAILABLE:
                self.provider = "huggingface"
                logger.info(f"Using Hugging Face with model: {self.model_name}")
            else:
                self.provider = "fallback"
                logger.info("Using fallback response generation")
        
        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI not available, falling back to fallback")
                self.provider = "fallback"
            elif not self.api_key:
                logger.warning("OpenAI API key not provided, falling back to fallback")
                self.provider = "fallback"
            else:
                openai.api_key = self.api_key
        
        elif self.provider == "ollama":
            if not self._check_ollama_available():
                logger.warning("Ollama not available, falling back to fallback")
                self.provider = "fallback"
        
        elif self.provider == "huggingface":
            if not HUGGINGFACE_AVAILABLE:
                logger.warning("Hugging Face not available, falling back to fallback")
                self.provider = "fallback"
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is available and running."""
        if not REQUESTS_AVAILABLE:
            return False
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def generate_answer(
        self, 
        query: str, 
        context_chunks: List[Dict[str, Any]], 
        max_tokens: int = 500,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Generate an answer using the selected LLM provider.
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks
            max_tokens: Maximum tokens for response
            temperature: Response creativity (0-1)
            
        Returns:
            Dictionary containing answer and metadata
        """
        start_time = time.time()
        
        try:
            # Construct prompt
            prompt = self._construct_prompt(query, context_chunks)
            
            # Generate response based on provider
            if self.provider == "openai":
                response = await self._call_openai(prompt, max_tokens, temperature)
            elif self.provider == "ollama":
                response = await self._call_ollama(prompt, max_tokens, temperature)
            elif self.provider == "huggingface":
                response = await self._call_huggingface(prompt, max_tokens, temperature)
            else:
                response = self._generate_fallback_response(query, context_chunks)
            
            processing_time = time.time() - start_time
            
            return {
                "answer": response["answer"],
                "citations": response["citations"],
                "metadata": {
                    "model_used": self.model_name,
                    "provider": self.provider,
                    "processing_time": processing_time,
                    "tokens_used": response.get("tokens_used", 0),
                    "context_chunks_used": len(context_chunks)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            processing_time = time.time() - start_time
            
            return {
                "answer": "I apologize, but I encountered an error while generating the answer. Please try again.",
                "citations": [],
                "metadata": {
                    "model_used": "fallback",
                    "provider": "fallback",
                    "processing_time": processing_time,
                    "error": str(e)
                }
            }
    
    def _construct_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Construct a prompt for the LLM.
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        prompt = """You are an AI research assistant. Use the provided context to answer the question.
Include citations in the format [source_name, page_number].
If the answer is not in the context, say you don't know.

Context:
"""
        
        for i, chunk in enumerate(context_chunks, 1):
            source_name = chunk.get("document_id", "Unknown")
            page_numbers = chunk.get("source_pages", [])
            page_str = f"Page {page_numbers[0]}" if page_numbers else "Unknown page"
            text = chunk.get("text", "")
            
            prompt += f"{i}. {text} (Source: {source_name}, {page_str})\n"
        
        prompt += f"\nQuestion: {query}\n\nAnswer:"
        
        return prompt
    
    async def _call_openai(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Call OpenAI API for answer generation."""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content.strip()
            citations = self._extract_citations(answer, prompt)
            
            return {
                "answer": answer,
                "citations": citations,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _call_ollama(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Call Ollama API for answer generation."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                citations = self._extract_citations(answer, prompt)
                
                return {
                    "answer": answer,
                    "citations": citations,
                    "tokens_used": result.get("eval_count", 0)
                }
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    async def _call_huggingface(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Call Hugging Face model for answer generation."""
        try:
            # For now, use a simple approach with a small model
            # In production, you might want to use a larger model or API
            model_name = "microsoft/DialoGPT-medium"  # Small model for demo
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Tokenize input
            inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response_text[len(prompt):].strip()
            
            citations = self._extract_citations(answer, prompt)
            
            return {
                "answer": answer,
                "citations": citations,
                "tokens_used": len(outputs[0])
            }
            
        except Exception as e:
            logger.error(f"Hugging Face error: {e}")
            raise
    
    def _generate_fallback_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a fallback response when LLM is not available.
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks
            
        Returns:
            Fallback response with citations
        """
        if not context_chunks:
            return {
                "answer": "I couldn't find any relevant information to answer your question. Please try rephrasing your query or check if the relevant documents have been uploaded.",
                "citations": []
            }
        
        # Use the most relevant chunk
        top_chunk = context_chunks[0]
        context_text = top_chunk.get("text", "")[:200] + "..." if len(top_chunk.get("text", "")) > 200 else top_chunk.get("text", "")
        
        template = random.choice(self.fallback_responses)
        answer = template.format(context=context_text)
        
        # Create basic citations
        citations = [{
            "source": top_chunk.get("document_id", "Unknown"),
            "page": top_chunk.get("source_pages", [1])[0] if top_chunk.get("source_pages") else 1,
            "text": top_chunk.get("text", "")[:100] + "..." if len(top_chunk.get("text", "")) > 100 else top_chunk.get("text", "")
        }]
        
        return {
            "answer": answer,
            "citations": citations
        }
    
    def _extract_citations(self, answer: str, prompt: str) -> List[Dict[str, Any]]:
        """
        Extract citations from the LLM response.
        
        Args:
            answer: LLM response
            prompt: Original prompt for context
            
        Returns:
            List of citations
        """
        citations = []
        
        # Simple citation extraction - look for [source, page] patterns
        import re
        citation_pattern = r'\[([^,]+),\s*page\s*(\d+)\]'
        matches = re.findall(citation_pattern, answer, re.IGNORECASE)
        
        for source, page in matches:
            citations.append({
                "source": source.strip(),
                "page": int(page),
                "text": f"Cited in answer: {source}, page {page}"
            })
        
        return citations
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        providers = []
        
        if OPENAI_AVAILABLE and self.api_key:
            providers.append("openai")
        
        if self._check_ollama_available():
            providers.append("ollama")
        
        if HUGGINGFACE_AVAILABLE:
            providers.append("huggingface")
        
        providers.append("fallback")
        
        return providers 