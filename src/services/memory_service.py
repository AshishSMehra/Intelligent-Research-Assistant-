"""
Memory Service for the Intelligent Research Assistant.

This service handles conversation history and context management.
"""

import time
import json
from typing import List, Dict, Any, Optional
from collections import defaultdict
from loguru import logger


class MemoryService:
    """Service for managing conversation memory and context."""
    
    def __init__(self, max_history_length: int = 10, session_timeout: int = 3600):
        """
        Initialize the memory service.
        
        Args:
            max_history_length: Maximum number of conversation turns to remember
            session_timeout: Session timeout in seconds
        """
        self.max_history_length = max_history_length
        self.session_timeout = session_timeout
        
        # In-memory storage (in production, use Redis or database)
        self.conversations = defaultdict(list)
        self.session_timestamps = {}
        
        logger.info(f"Memory Service initialized - Max history: {max_history_length}, Timeout: {session_timeout}s")
    
    def add_conversation_turn(
        self, 
        session_id: str, 
        user_query: str, 
        assistant_response: str,
        context_chunks: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add a conversation turn to memory.
        
        Args:
            session_id: Unique session identifier
            user_query: User's question
            assistant_response: Assistant's response
            context_chunks: Context chunks used for response
            metadata: Additional metadata
        """
        current_time = time.time()
        
        # Clean up old sessions
        self._cleanup_expired_sessions(current_time)
        
        # Create conversation turn
        turn = {
            "timestamp": current_time,
            "user_query": user_query,
            "assistant_response": assistant_response,
            "context_chunks": context_chunks or [],
            "metadata": metadata or {}
        }
        
        # Add to conversation history
        self.conversations[session_id].append(turn)
        
        # Limit history length
        if len(self.conversations[session_id]) > self.max_history_length:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history_length:]
        
        # Update session timestamp
        self.session_timestamps[session_id] = current_time
        
        logger.debug(f"Added conversation turn for session {session_id}")
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation turns
        """
        current_time = time.time()
        self._cleanup_expired_sessions(current_time)
        
        return self.conversations.get(session_id, [])
    
    def get_context_for_query(self, session_id: str, current_query: str) -> str:
        """
        Get conversation context for a new query.
        
        Args:
            session_id: Session identifier
            current_query: Current user query
            
        Returns:
            Formatted context string
        """
        history = self.get_conversation_history(session_id)
        
        if not history:
            return ""
        
        context_parts = []
        context_parts.append("Previous conversation context:")
        
        for i, turn in enumerate(history[-3:], 1):  # Last 3 turns
            context_parts.append(f"Turn {i}:")
            context_parts.append(f"User: {turn['user_query']}")
            context_parts.append(f"Assistant: {turn['assistant_response'][:200]}...")
            context_parts.append("")
        
        context_parts.append("Current query:")
        context_parts.append(f"User: {current_query}")
        context_parts.append("")
        
        return "\n".join(context_parts)
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was cleared, False if not found
        """
        if session_id in self.conversations:
            del self.conversations[session_id]
            del self.session_timestamps[session_id]
            logger.info(f"Cleared conversation history for session {session_id}")
            return True
        return False
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session statistics
        """
        history = self.get_conversation_history(session_id)
        
        if not history:
            return {
                "session_id": session_id,
                "turns_count": 0,
                "last_activity": None,
                "total_tokens": 0
            }
        
        total_tokens = sum(
            len(turn.get("user_query", "")) + len(turn.get("assistant_response", ""))
            for turn in history
        )
        
        return {
            "session_id": session_id,
            "turns_count": len(history),
            "last_activity": history[-1]["timestamp"],
            "total_tokens": total_tokens
        }
    
    def _cleanup_expired_sessions(self, current_time: float) -> None:
        """
        Clean up expired sessions.
        
        Args:
            current_time: Current timestamp
        """
        expired_sessions = []
        
        for session_id, timestamp in self.session_timestamps.items():
            if current_time - timestamp > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.conversations[session_id]
            del self.session_timestamps[session_id]
            logger.info(f"Cleaned up expired session {session_id}")
    
    def get_all_sessions(self) -> List[str]:
        """
        Get all active session IDs.
        
        Returns:
            List of active session IDs
        """
        current_time = time.time()
        self._cleanup_expired_sessions(current_time)
        
        return list(self.conversations.keys())
    
    def export_conversation(self, session_id: str) -> Optional[str]:
        """
        Export conversation as JSON.
        
        Args:
            session_id: Session identifier
            
        Returns:
            JSON string of conversation, or None if not found
        """
        history = self.get_conversation_history(session_id)
        
        if not history:
            return None
        
        export_data = {
            "session_id": session_id,
            "export_timestamp": time.time(),
            "conversation": history
        }
        
        return json.dumps(export_data, indent=2) 