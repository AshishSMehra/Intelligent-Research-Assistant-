"""
Step 4: Rate Limiting & Abuse Detection

This module implements comprehensive rate limiting and abuse detection
to protect the API from malicious usage and ensure fair resource allocation.
"""

import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from loguru import logger

import redis
from flask import request, jsonify, g


class RateLimitType(Enum):
    """Types of rate limiting."""
    IP_BASED = "ip_based"
    USER_BASED = "user_based"
    ENDPOINT_BASED = "endpoint_based"
    GLOBAL = "global"


class AbuseType(Enum):
    """Types of abuse that can be detected."""
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_PATTERNS = "suspicious_patterns"
    MALICIOUS_PAYLOAD = "malicious_payload"
    BOT_ACTIVITY = "bot_activity"
    BRUTE_FORCE = "brute_force"
    DDoS_ATTEMPT = "ddos_attempt"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    window_size: int = 60  # seconds
    penalty_duration: int = 300  # seconds


@dataclass
class AbuseDetectionConfig:
    """Configuration for abuse detection."""
    suspicious_patterns: List[str] = field(default_factory=list)
    max_payload_size: int = 1048576  # 1MB
    max_concurrent_requests: int = 10
    suspicious_headers: List[str] = field(default_factory=list)
    blocked_user_agents: List[str] = field(default_factory=list)


@dataclass
class RateLimitInfo:
    """Information about rate limiting for a client."""
    client_id: str
    request_count: int = 0
    window_start: float = field(default_factory=time.time)
    last_request: float = field(default_factory=time.time)
    penalty_until: Optional[float] = None
    abuse_score: float = 0.0


class RateLimiter:
    """Rate limiting implementation with Redis backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", config: RateLimitConfig = None):
        self.redis_client = redis.from_url(redis_url)
        self.config = config or RateLimitConfig()
        
        # Initialize rate limit windows
        self.windows = {
            "minute": 60,
            "hour": 3600,
            "day": 86400
        }
        
        logger.info("Rate Limiter initialized")
    
    def _get_client_id(self, request) -> str:
        """Get unique client identifier."""
        # Try to get user ID from JWT token first
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            try:
                import jwt
                token = auth_header.split(' ')[1]
                payload = jwt.decode(token, options={"verify_signature": False})
                if 'user_id' in payload:
                    return f"user:{payload['user_id']}"
            except:
                pass
        
        # Fallback to IP address
        return f"ip:{request.remote_addr}"
    
    def _get_rate_limit_key(self, client_id: str, window: str) -> str:
        """Get Redis key for rate limiting."""
        current_window = int(time.time() // self.windows[window])
        return f"rate_limit:{client_id}:{window}:{current_window}"
    
    def is_rate_limited(self, request) -> Tuple[bool, Dict[str, Any]]:
        """Check if request should be rate limited."""
        client_id = self._get_client_id(request)
        
        # Check if client is under penalty
        penalty_key = f"penalty:{client_id}"
        penalty_until = self.redis_client.get(penalty_key)
        if penalty_until:
            penalty_time = float(penalty_until)
            if time.time() < penalty_time:
                return True, {
                    "error": "Rate limit exceeded",
                    "retry_after": int(penalty_time - time.time()),
                    "client_id": client_id
                }
            else:
                # Remove expired penalty
                self.redis_client.delete(penalty_key)
        
        # Check rate limits for different windows
        for window_name, window_seconds in self.windows.items():
            limit_key = self._get_rate_limit_key(client_id, window_name)
            
            # Get current count
            current_count = self.redis_client.get(limit_key)
            current_count = int(current_count) if current_count else 0
            
            # Get limit for this window
            if window_name == "minute":
                limit = self.config.requests_per_minute
            elif window_name == "hour":
                limit = self.config.requests_per_hour
            else:  # day
                limit = self.config.requests_per_day
            
            # Check if limit exceeded
            if current_count >= limit:
                # Apply penalty
                penalty_duration = self.config.penalty_duration
                self.redis_client.setex(penalty_key, penalty_duration, time.time() + penalty_duration)
                
                return True, {
                    "error": f"Rate limit exceeded for {window_name}",
                    "retry_after": penalty_duration,
                    "client_id": client_id,
                    "window": window_name,
                    "limit": limit,
                    "current": current_count
                }
        
        # Increment counters
        for window_name in self.windows.keys():
            limit_key = self._get_rate_limit_key(client_id, window_name)
            self.redis_client.incr(limit_key)
            self.redis_client.expire(limit_key, self.windows[window_name])
        
        return False, {"client_id": client_id}
    
    def get_rate_limit_info(self, client_id: str) -> Dict[str, Any]:
        """Get rate limit information for a client."""
        info = {"client_id": client_id}
        
        for window_name, window_seconds in self.windows.items():
            limit_key = self._get_rate_limit_key(client_id, window_name)
            current_count = self.redis_client.get(limit_key)
            current_count = int(current_count) if current_count else 0
            
            if window_name == "minute":
                limit = self.config.requests_per_minute
            elif window_name == "hour":
                limit = self.config.requests_per_hour
            else:  # day
                limit = self.config.requests_per_day
            
            info[window_name] = {
                "current": current_count,
                "limit": limit,
                "remaining": max(0, limit - current_count)
            }
        
        return info
    
    def reset_rate_limit(self, client_id: str) -> bool:
        """Reset rate limit for a client."""
        try:
            # Remove penalty
            penalty_key = f"penalty:{client_id}"
            self.redis_client.delete(penalty_key)
            
            # Reset counters
            for window_name in self.windows.keys():
                limit_key = self._get_rate_limit_key(client_id, window_name)
                self.redis_client.delete(limit_key)
            
            logger.info(f"Reset rate limit for client: {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset rate limit for {client_id}: {e}")
            return False


class AbuseDetector:
    """Abuse detection and prevention system."""
    
    def __init__(self, config: AbuseDetectionConfig = None):
        self.config = config or AbuseDetectionConfig()
        
        # Initialize suspicious patterns
        self.suspicious_patterns = [
            r'<script.*?>.*?</script>',  # XSS attempts
            r'javascript:',  # JavaScript injection
            r'../',  # Path traversal
            r'exec\(',  # Code execution
            r'union.*select',  # SQL injection
            r'<iframe',  # Iframe injection
            r'data:text/html',  # Data URI attacks
        ]
        
        # Compile patterns
        import re
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.suspicious_patterns]
        
        # Abuse tracking
        self.abuse_scores = defaultdict(float)
        self.blocked_clients = set()
        
        logger.info("Abuse Detector initialized")
    
    def detect_abuse(self, request) -> Tuple[bool, Dict[str, Any]]:
        """Detect potential abuse in request."""
        client_id = self._get_client_id(request)
        abuse_info = {
            "client_id": client_id,
            "abuse_types": [],
            "score": 0.0,
            "blocked": False
        }
        
        # Check if client is already blocked
        if client_id in self.blocked_clients:
            abuse_info["blocked"] = True
            abuse_info["abuse_types"].append(AbuseType.RATE_LIMIT_EXCEEDED.value)
            return True, abuse_info
        
        # Check payload size
        if request.content_length and request.content_length > self.config.max_payload_size:
            abuse_info["abuse_types"].append("payload_too_large")
            abuse_info["score"] += 10.0
        
        # Check suspicious headers
        for header_name in self.config.suspicious_headers:
            if header_name in request.headers:
                abuse_info["abuse_types"].append("suspicious_header")
                abuse_info["score"] += 5.0
        
        # Check user agent
        user_agent = request.headers.get('User-Agent', '')
        for blocked_ua in self.config.blocked_user_agents:
            if blocked_ua.lower() in user_agent.lower():
                abuse_info["abuse_types"].append(AbuseType.BOT_ACTIVITY.value)
                abuse_info["score"] += 15.0
        
        # Check for suspicious patterns in request data
        request_data = self._get_request_data(request)
        if request_data:
            for pattern in self.compiled_patterns:
                if pattern.search(request_data):
                    abuse_info["abuse_types"].append(AbuseType.MALICIOUS_PAYLOAD.value)
                    abuse_info["score"] += 20.0
                    break
        
        # Check for suspicious patterns in URL
        url = request.url
        for pattern in self.compiled_patterns:
            if pattern.search(url):
                abuse_info["abuse_types"].append(AbuseType.MALICIOUS_PAYLOAD.value)
                abuse_info["score"] += 20.0
                break
        
        # Update abuse score
        self.abuse_scores[client_id] += abuse_info["score"]
        
        # Check if client should be blocked
        if self.abuse_scores[client_id] >= 50.0:
            self.blocked_clients.add(client_id)
            abuse_info["blocked"] = True
            abuse_info["abuse_types"].append("client_blocked")
        
        return len(abuse_info["abuse_types"]) > 0, abuse_info
    
    def _get_client_id(self, request) -> str:
        """Get unique client identifier."""
        # Try to get user ID from JWT token first
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            try:
                import jwt
                token = auth_header.split(' ')[1]
                payload = jwt.decode(token, options={"verify_signature": False})
                if 'user_id' in payload:
                    return f"user:{payload['user_id']}"
            except:
                pass
        
        # Fallback to IP address
        return f"ip:{request.remote_addr}"
    
    def _get_request_data(self, request) -> str:
        """Extract request data for analysis."""
        data = ""
        
        # Get form data
        if request.form:
            data += str(request.form)
        
        # Get JSON data
        if request.is_json:
            data += str(request.get_json())
        
        # Get query parameters
        if request.args:
            data += str(request.args)
        
        return data
    
    def get_abuse_info(self, client_id: str) -> Dict[str, Any]:
        """Get abuse information for a client."""
        return {
            "client_id": client_id,
            "abuse_score": self.abuse_scores.get(client_id, 0.0),
            "blocked": client_id in self.blocked_clients
        }
    
    def reset_abuse_score(self, client_id: str) -> bool:
        """Reset abuse score for a client."""
        try:
            self.abuse_scores[client_id] = 0.0
            self.blocked_clients.discard(client_id)
            logger.info(f"Reset abuse score for client: {client_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset abuse score for {client_id}: {e}")
            return False
    
    def block_client(self, client_id: str, duration: int = 3600) -> bool:
        """Manually block a client."""
        try:
            self.blocked_clients.add(client_id)
            self.abuse_scores[client_id] = 100.0  # High abuse score
            
            # Auto-unblock after duration
            import threading
            def unblock_after_delay():
                time.sleep(duration)
                self.blocked_clients.discard(client_id)
                self.abuse_scores[client_id] = 0.0
            
            threading.Thread(target=unblock_after_delay, daemon=True).start()
            
            logger.info(f"Blocked client: {client_id} for {duration} seconds")
            return True
        except Exception as e:
            logger.error(f"Failed to block client {client_id}: {e}")
            return False


class SecurityMiddleware:
    """Middleware for applying security measures to Flask requests."""
    
    def __init__(self, rate_limiter: RateLimiter = None, abuse_detector: AbuseDetector = None):
        self.rate_limiter = rate_limiter or RateLimiter()
        self.abuse_detector = abuse_detector or AbuseDetector()
        
        logger.info("Security Middleware initialized")
    
    def process_request(self, request) -> Optional[Tuple[str, int]]:
        """Process incoming request for security checks."""
        # Check for abuse
        is_abuse, abuse_info = self.abuse_detector.detect_abuse(request)
        if is_abuse:
            logger.warning(f"Abuse detected: {abuse_info}")
            if abuse_info.get("blocked", False):
                return jsonify({
                    "error": "Access denied",
                    "reason": "Suspicious activity detected",
                    "client_id": abuse_info["client_id"]
                }), 403
        
        # Check rate limiting
        is_limited, limit_info = self.rate_limiter.is_rate_limited(request)
        if is_limited:
            logger.warning(f"Rate limit exceeded: {limit_info}")
            return jsonify({
                "error": "Rate limit exceeded",
                "retry_after": limit_info.get("retry_after", 300),
                "client_id": limit_info.get("client_id", "unknown")
            }), 429
        
        # Store security info in request context
        g.security_info = {
            "client_id": limit_info.get("client_id", "unknown"),
            "abuse_score": abuse_info.get("score", 0.0),
            "abuse_types": abuse_info.get("abuse_types", [])
        }
        
        return None
    
    def process_response(self, response) -> Any:
        """Process outgoing response for security headers."""
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        return response


# Global security instances
global_rate_limiter = RateLimiter()
global_abuse_detector = AbuseDetector()
global_security_middleware = SecurityMiddleware(global_rate_limiter, global_abuse_detector)


def require_rate_limit():
    """Decorator to apply rate limiting to Flask routes."""
    def decorator(f):
        def decorated_function(*args, **kwargs):
            # Check rate limiting
            is_limited, limit_info = global_rate_limiter.is_rate_limited(request)
            if is_limited:
                return jsonify({
                    "error": "Rate limit exceeded",
                    "retry_after": limit_info.get("retry_after", 300)
                }), 429
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def require_abuse_check():
    """Decorator to apply abuse detection to Flask routes."""
    def decorator(f):
        def decorated_function(*args, **kwargs):
            # Check for abuse
            is_abuse, abuse_info = global_abuse_detector.detect_abuse(request)
            if is_abuse and abuse_info.get("blocked", False):
                return jsonify({
                    "error": "Access denied",
                    "reason": "Suspicious activity detected"
                }), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator 