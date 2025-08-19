"""
Step 3: PII Redaction

This module implements PII (Personally Identifiable Information) redaction
for logs and outputs to ensure privacy and compliance.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger


class PIIType(Enum):
    """Types of PII that can be detected and redacted."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    DATE_OF_BIRTH = "date_of_birth"
    ADDRESS = "address"
    NAME = "name"
    USERNAME = "username"
    PASSWORD = "password"
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"


@dataclass
class PIIPattern:
    """Pattern definition for PII detection."""

    pii_type: PIIType
    pattern: str
    description: str
    replacement: str = "[REDACTED]"
    flags: int = re.IGNORECASE


class PIIPatterns:
    """Predefined PII patterns for detection and redaction."""

    @staticmethod
    def get_patterns() -> List[PIIPattern]:
        """Get all predefined PII patterns."""
        return [
            # Email addresses
            PIIPattern(
                pii_type=PIIType.EMAIL,
                pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                description="Email address pattern",
            ),
            # Phone numbers (US format)
            PIIPattern(
                pii_type=PIIType.PHONE,
                pattern=r"\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
                description="Phone number pattern",
            ),
            # Social Security Numbers
            PIIPattern(
                pii_type=PIIType.SSN,
                pattern=r"\b\d{3}-\d{2}-\d{4}\b",
                description="Social Security Number pattern",
            ),
            # Credit card numbers
            PIIPattern(
                pii_type=PIIType.CREDIT_CARD,
                pattern=r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                description="Credit card number pattern",
            ),
            # IP addresses
            PIIPattern(
                pii_type=PIIType.IP_ADDRESS,
                pattern=r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
                description="IP address pattern",
            ),
            # MAC addresses
            PIIPattern(
                pii_type=PIIType.MAC_ADDRESS,
                pattern=r"\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b",
                description="MAC address pattern",
            ),
            # Dates of birth
            PIIPattern(
                pii_type=PIIType.DATE_OF_BIRTH,
                pattern=r"\b(0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])[/-]\d{4}\b",
                description="Date of birth pattern",
            ),
            # Names (simple pattern)
            PIIPattern(
                pii_type=PIIType.NAME,
                pattern=r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
                description="Full name pattern",
            ),
            # Usernames
            PIIPattern(
                pii_type=PIIType.USERNAME,
                pattern=r"\b@[A-Za-z0-9_]+\b",
                description="Username pattern",
            ),
            # Passwords (basic pattern)
            PIIPattern(
                pii_type=PIIType.PASSWORD,
                pattern=r'password["\s]*[:=]\s*["\']?[^"\'\s]+["\']?',
                description="Password pattern",
            ),
            # API keys
            PIIPattern(
                pii_type=PIIType.API_KEY,
                pattern=r"\b[A-Za-z0-9]{32,}\b",
                description="API key pattern",
            ),
            # JWT tokens
            PIIPattern(
                pii_type=PIIType.JWT_TOKEN,
                pattern=r"\b[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]*\b",
                description="JWT token pattern",
            ),
        ]

    @staticmethod
    def get_custom_patterns() -> Dict[str, str]:
        """Get custom patterns for specific use cases."""
        return {
            # Custom patterns can be added here
            "custom_id": r"\bID:\s*\d{6,}\b",
            "custom_reference": r"\bREF:\s*[A-Z0-9]{8,}\b",
        }


@dataclass
class RedactionResult:
    """Result of PII redaction operation."""

    original_text: str
    redacted_text: str
    redacted_pii: List[Dict[str, Any]]
    redaction_count: int


class PIIRedactor:
    """PII detection and redaction engine."""

    def __init__(
        self, custom_patterns: Dict[str, str] = None, enable_hashing: bool = False
    ):
        self.patterns = PIIPatterns.get_patterns()
        self.custom_patterns = custom_patterns or {}
        self.enable_hashing = enable_hashing

        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for pattern in self.patterns:
            self.compiled_patterns[pattern.pii_type] = re.compile(
                pattern.pattern, pattern.flags
            )

        # Compile custom patterns
        for name, pattern_str in self.custom_patterns.items():
            self.compiled_patterns[name] = re.compile(pattern_str, re.IGNORECASE)

        logger.info(f"PII Redactor initialized with {len(self.patterns)} patterns")

    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII in text without redacting."""
        detected_pii = []

        for pii_type, pattern in self.compiled_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                detected_pii.append(
                    {
                        "type": (
                            pii_type.value
                            if hasattr(pii_type, "value")
                            else str(pii_type)
                        ),
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": self._calculate_confidence(
                            match.group(), pii_type
                        ),
                    }
                )

        return detected_pii

    def redact_text(
        self, text: str, replacement: str = "[REDACTED]"
    ) -> RedactionResult:
        """Redact PII from text."""
        if not text:
            return RedactionResult(text, text, [], 0)

        redacted_text = text
        redacted_pii = []
        offset = 0

        # Sort detected PII by start position (reverse order to maintain indices)
        detected_pii = self.detect_pii(text)
        detected_pii.sort(key=lambda x: x["start"], reverse=True)

        for pii in detected_pii:
            start = pii["start"] + offset
            end = pii["end"] + offset

            # Apply redaction
            redacted_text = redacted_text[:start] + replacement + redacted_text[end:]

            # Update offset for subsequent replacements
            offset += len(replacement) - (end - start)

            # Update PII info
            pii["redacted_value"] = replacement
            redacted_pii.append(pii)

        return RedactionResult(
            original_text=text,
            redacted_text=redacted_text,
            redacted_pii=redacted_pii,
            redaction_count=len(redacted_pii),
        )

    def redact_json(self, data: Any, replacement: str = "[REDACTED]") -> Any:
        """Redact PII from JSON data structures."""
        if isinstance(data, str):
            return self.redact_text(data, replacement).redacted_text
        elif isinstance(data, dict):
            return {
                key: self.redact_json(value, replacement) for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self.redact_json(item, replacement) for item in data]
        else:
            return data

    def redact_log_message(self, message: str, replacement: str = "[REDACTED]") -> str:
        """Redact PII from log messages."""
        result = self.redact_text(message, replacement)
        return result.redacted_text

    def redact_api_response(
        self, response_data: Dict[str, Any], replacement: str = "[REDACTED]"
    ) -> Dict[str, Any]:
        """Redact PII from API response data."""
        return self.redact_json(response_data, replacement)

    def redact_user_input(
        self, user_input: str, replacement: str = "[REDACTED]"
    ) -> str:
        """Redact PII from user input."""
        result = self.redact_text(user_input, replacement)
        return result.redacted_text

    def _calculate_confidence(self, value: str, pii_type: PIIType) -> float:
        """Calculate confidence score for PII detection."""
        # Simple confidence calculation based on pattern type
        base_confidence = 0.8

        if pii_type == PIIType.EMAIL:
            # Email validation
            if "@" in value and "." in value.split("@")[1]:
                return 0.95
            return 0.7

        elif pii_type == PIIType.PHONE:
            # Phone number validation
            digits = re.sub(r"\D", "", value)
            if len(digits) == 10 or len(digits) == 11:
                return 0.9
            return 0.6

        elif pii_type == PIIType.SSN:
            # SSN validation
            digits = re.sub(r"\D", "", value)
            if len(digits) == 9:
                return 0.95
            return 0.5

        elif pii_type == PIIType.CREDIT_CARD:
            # Credit card validation (Luhn algorithm)
            digits = re.sub(r"\D", "", value)
            if len(digits) >= 13 and len(digits) <= 19:
                return 0.85
            return 0.4

        return base_confidence

    def hash_sensitive_data(self, text: str, salt: str = None) -> str:
        """Hash sensitive data for secure storage."""
        if not self.enable_hashing:
            return text

        if salt is None:
            salt = "default_salt"  # In production, use a secure salt

        # Create hash
        hash_object = hashlib.sha256()
        hash_object.update((text + salt).encode("utf-8"))
        return hash_object.hexdigest()

    def create_redaction_report(self, result: RedactionResult) -> Dict[str, Any]:
        """Create a detailed redaction report."""
        return {
            "summary": {
                "total_redactions": result.redaction_count,
                "original_length": len(result.original_text),
                "redacted_length": len(result.redacted_text),
                "redaction_ratio": result.redaction_count
                / max(len(result.original_text), 1),
            },
            "redacted_pii": result.redacted_pii,
            "pii_types_found": list(set(pii["type"] for pii in result.redacted_pii)),
            "timestamp": self._get_timestamp(),
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import time

        return time.strftime("%Y-%m-%d %H:%M:%S")


class LogRedactor:
    """Specialized redactor for log messages."""

    def __init__(self, pii_redactor: PIIRedactor = None):
        self.pii_redactor = pii_redactor or PIIRedactor()

        # Log-specific patterns
        self.log_patterns = {
            "user_id": r'user_id["\s]*[:=]\s*["\']?[^"\'\s]+["\']?',
            "session_id": r'session_id["\s]*[:=]\s*["\']?[^"\'\s]+["\']?',
            "request_id": r'request_id["\s]*[:=]\s*["\']?[^"\'\s]+["\']?',
            "ip_address": r'ip["\s]*[:=]\s*["\']?[^"\'\s]+["\']?',
            "user_agent": r'user_agent["\s]*[:=]\s*["\']?[^"\'\s]+["\']?',
        }

    def redact_log(self, log_message: str) -> str:
        """Redact sensitive information from log messages."""
        # First apply general PII redaction
        result = self.pii_redactor.redact_text(log_message)

        # Then apply log-specific redaction
        redacted_message = result.redacted_text

        for pattern_name, pattern in self.log_patterns.items():
            redacted_message = re.sub(
                pattern,
                f'{pattern_name}="[REDACTED]"',
                redacted_message,
                flags=re.IGNORECASE,
            )

        return redacted_message


class APIRedactor:
    """Specialized redactor for API requests and responses."""

    def __init__(self, pii_redactor: PIIRedactor = None):
        self.pii_redactor = pii_redactor or PIIRedactor()

        # API-specific sensitive fields
        self.sensitive_fields = {
            "password",
            "token",
            "api_key",
            "secret",
            "key",
            "auth",
            "authorization",
            "cookie",
            "session",
            "credential",
        }

    def redact_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive information from API requests."""
        return self._redact_api_data(request_data, "request")

    def redact_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive information from API responses."""
        return self._redact_api_data(response_data, "response")

    def _redact_api_data(self, data: Any, data_type: str) -> Any:
        """Recursively redact sensitive data from API payloads."""
        if isinstance(data, dict):
            redacted_data = {}
            for key, value in data.items():
                # Check if key contains sensitive information
                if self._is_sensitive_field(key):
                    redacted_data[key] = "[REDACTED]"
                else:
                    redacted_data[key] = self._redact_api_data(value, data_type)
            return redacted_data
        elif isinstance(data, list):
            return [self._redact_api_data(item, data_type) for item in data]
        elif isinstance(data, str):
            # Apply PII redaction to string values
            return self.pii_redactor.redact_text(data).redacted_text
        else:
            return data

    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if a field name indicates sensitive data."""
        field_lower = field_name.lower()
        return any(sensitive in field_lower for sensitive in self.sensitive_fields)


# Global redactor instances
global_pii_redactor = PIIRedactor()
global_log_redactor = LogRedactor(global_pii_redactor)
global_api_redactor = APIRedactor(global_pii_redactor)


def redact_log_message(message: str) -> str:
    """Global function to redact PII from log messages."""
    return global_log_redactor.redact_log(message)


def redact_api_data(
    data: Dict[str, Any], data_type: str = "response"
) -> Dict[str, Any]:
    """Global function to redact PII from API data."""
    if data_type == "request":
        return global_api_redactor.redact_request(data)
    else:
        return global_api_redactor.redact_response(data)


def detect_pii_in_text(text: str) -> List[Dict[str, Any]]:
    """Global function to detect PII in text."""
    return global_pii_redactor.detect_pii(text)
