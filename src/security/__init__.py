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

from .rbac import RBACManager, Role, Permission, User
from .secrets import SecretsManager, AWSKMSManager, VaultManager
from .pii_redaction import PIIRedactor, PIIPatterns
from .rate_limiting import RateLimiter, AbuseDetector
from .data_retention import DataRetentionManager, RetentionPolicy

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
    "RetentionPolicy"
] 