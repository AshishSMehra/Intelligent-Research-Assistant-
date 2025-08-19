"""
Security & Compliance Module for Intelligent Research Assistant.

This module implements comprehensive security features:
1. Role-Based Access Control (RBAC)
2. Secrets Management (AWS KMS/Vault)
3. PII Redaction
4. Rate Limiting & Abuse Detection
5. Data Retention Policies & Opt-out Mechanisms

Author: Ashish Mehra
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Ashish Mehra"

from .data_retention import DataRetentionManager, RetentionPolicy
from .pii_redaction import PIIPatterns, PIIRedactor
from .rate_limiting import AbuseDetector, RateLimiter
from .rbac import Permission, RBACManager, Role, User
from .secrets import AWSKMSManager, SecretsManager, VaultManager

__all__ = [
    "RBACManager",
    "Role",
    "Permission",
    "User",
    "SecretsManager",
    "AWSKMSManager",
    "VaultManager",
    "PIIRedactor",
    "PIIPatterns",
    "RateLimiter",
    "AbuseDetector",
    "DataRetentionManager",
    "RetentionPolicy",
]
