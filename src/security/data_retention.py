"""
Step 5: Data Retention Policies & Opt-out Mechanisms

This module implements comprehensive data retention policies and opt-out mechanisms
to ensure GDPR compliance and user privacy rights.
"""

import hashlib
import json
import os
import random
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import redis
import requests
from botocore.exceptions import NoCredentialsError
from datasets import Dataset, DatasetDict
from flask import g, jsonify, request
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, pipeline


class DataCategory(Enum):
    """Categories of data for retention policies."""

    USER_PROFILE = "user_profile"
    CHAT_HISTORY = "chat_history"
    SEARCH_HISTORY = "search_history"
    UPLOADED_DOCUMENTS = "uploaded_documents"
    EMBEDDINGS = "embeddings"
    FEEDBACK_DATA = "feedback_data"
    SYSTEM_LOGS = "system_logs"
    ANALYTICS = "analytics"
    SESSION_DATA = "session_data"


class RetentionPeriod(Enum):
    """Retention periods for different data categories."""

    IMMEDIATE = 0  # Delete immediately
    ONE_DAY = 86400
    ONE_WEEK = 604800
    ONE_MONTH = 2592000
    THREE_MONTHS = 7776000
    SIX_MONTHS = 15552000
    ONE_YEAR = 31536000
    TWO_YEARS = 63072000
    INDEFINITE = -1  # Keep indefinitely


class OptOutType(Enum):
    """Types of opt-out requests."""

    DATA_COLLECTION = "data_collection"
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    THIRD_PARTY_SHARING = "third_party_sharing"
    AUTOMATED_DECISIONS = "automated_decisions"
    PROFILING = "profiling"


@dataclass
class RetentionPolicy:
    """Data retention policy configuration."""

    category: DataCategory
    retention_period: RetentionPeriod
    description: str
    legal_basis: str
    auto_delete: bool = True
    archive_before_delete: bool = False
    notify_before_deletion: bool = True
    notification_days: int = 30


@dataclass
class OptOutRequest:
    """User opt-out request."""

    user_id: str
    opt_out_type: OptOutType
    request_date: datetime
    effective_date: datetime
    reason: Optional[str] = None
    status: str = "pending"  # pending, active, revoked
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataRetentionRecord:
    """Record of data retention for audit purposes."""

    record_id: str
    user_id: str
    data_category: DataCategory
    created_date: datetime
    retention_policy: RetentionPolicy
    scheduled_deletion_date: datetime
    actual_deletion_date: Optional[datetime] = None
    deletion_reason: Optional[str] = None
    archived: bool = False


class DataRetentionManager:
    """Manages data retention policies and enforcement."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)

        # Initialize default retention policies
        self.retention_policies = self._initialize_default_policies()

        # Opt-out tracking
        self.opt_out_requests = {}

        # Retention records
        self.retention_records = {}

        logger.info("Data Retention Manager initialized")

    def _initialize_default_policies(self) -> Dict[DataCategory, RetentionPolicy]:
        """Initialize default retention policies."""
        policies = {
            DataCategory.USER_PROFILE: RetentionPolicy(
                category=DataCategory.USER_PROFILE,
                retention_period=RetentionPeriod.INDEFINITE,
                description="User profile data retained until account deletion",
                legal_basis="Contract performance",
                auto_delete=False,
            ),
            DataCategory.CHAT_HISTORY: RetentionPolicy(
                category=DataCategory.CHAT_HISTORY,
                retention_period=RetentionPeriod.SIX_MONTHS,
                description="Chat history retained for 6 months",
                legal_basis="Legitimate interest",
                auto_delete=True,
                notify_before_deletion=True,
                notification_days=30,
            ),
            DataCategory.SEARCH_HISTORY: RetentionPolicy(
                category=DataCategory.SEARCH_HISTORY,
                retention_period=RetentionPeriod.THREE_MONTHS,
                description="Search history retained for 3 months",
                legal_basis="Legitimate interest",
                auto_delete=True,
            ),
            DataCategory.UPLOADED_DOCUMENTS: RetentionPolicy(
                category=DataCategory.UPLOADED_DOCUMENTS,
                retention_period=RetentionPeriod.ONE_YEAR,
                description="Uploaded documents retained for 1 year",
                legal_basis="Contract performance",
                auto_delete=True,
                archive_before_delete=True,
            ),
            DataCategory.EMBEDDINGS: RetentionPolicy(
                category=DataCategory.EMBEDDINGS,
                retention_period=RetentionPeriod.ONE_YEAR,
                description="Document embeddings retained for 1 year",
                legal_basis="Contract performance",
                auto_delete=True,
            ),
            DataCategory.FEEDBACK_DATA: RetentionPolicy(
                category=DataCategory.FEEDBACK_DATA,
                retention_period=RetentionPeriod.TWO_YEARS,
                description="Feedback data retained for 2 years",
                legal_basis="Legitimate interest",
                auto_delete=True,
            ),
            DataCategory.SYSTEM_LOGS: RetentionPolicy(
                category=DataCategory.SYSTEM_LOGS,
                retention_period=RetentionPeriod.ONE_MONTH,
                description="System logs retained for 1 month",
                legal_basis="Legal obligation",
                auto_delete=True,
            ),
            DataCategory.ANALYTICS: RetentionPolicy(
                category=DataCategory.ANALYTICS,
                retention_period=RetentionPeriod.SIX_MONTHS,
                description="Analytics data retained for 6 months",
                legal_basis="Legitimate interest",
                auto_delete=True,
            ),
            DataCategory.SESSION_DATA: RetentionPolicy(
                category=DataCategory.SESSION_DATA,
                retention_period=RetentionPeriod.ONE_DAY,
                description="Session data retained for 1 day",
                legal_basis="Contract performance",
                auto_delete=True,
            ),
        }

        return policies

    def create_retention_record(
        self,
        user_id: str,
        data_category: DataCategory,
        data_id: str,
        created_date: datetime = None,
    ) -> str:
        """Create a retention record for data."""
        if created_date is None:
            created_date = datetime.now()

        policy = self.retention_policies[data_category]

        # Calculate scheduled deletion date
        if policy.retention_period == RetentionPeriod.INDEFINITE:
            scheduled_deletion_date = None
        else:
            scheduled_deletion_date = created_date + timedelta(
                seconds=policy.retention_period.value
            )

        record_id = "retention_{int(time.time())}_{hashlib.md5(f'{user_id}_{data_id}'.encode()).hexdigest()[:8]}"

        record = DataRetentionRecord(
            record_id=record_id,
            user_id=user_id,
            data_category=data_category,
            created_date=created_date,
            retention_policy=policy,
            scheduled_deletion_date=scheduled_deletion_date,
        )

        # Store record
        self.retention_records[record_id] = record
        self._store_retention_record(record)

        logger.info("Created retention record: {record_id} for user {user_id}")
        return record_id

    def _store_retention_record(self, record: DataRetentionRecord):
        """Store retention record in Redis."""
        record_data = {
            "record_id": record.record_id,
            "user_id": record.user_id,
            "data_category": record.data_category.value,
            "created_date": record.created_date.isoformat(),
            "scheduled_deletion_date": (
                record.scheduled_deletion_date.isoformat()
                if record.scheduled_deletion_date
                else ""
            ),
            "actual_deletion_date": (
                record.actual_deletion_date.isoformat()
                if record.actual_deletion_date
                else ""
            ),
            "deletion_reason": record.deletion_reason or "",
            "archived": str(record.archived),
        }

        self.redis_client.hset(
            "retention_record:{record.record_id}", mapping=record_data
        )

    def get_retention_records(
        self, user_id: str = None, data_category: DataCategory = None
    ) -> List[DataRetentionRecord]:
        """Get retention records with optional filtering."""
        records = []

        for record in self.retention_records.values():
            if user_id and record.user_id != user_id:
                continue
            if data_category and record.data_category != data_category:
                continue
            records.append(record)

        return records

    def schedule_data_deletion(
        self, user_id: str, data_category: DataCategory, deletion_date: datetime = None
    ) -> bool:
        """Schedule data for deletion."""
        try:
            policy = self.retention_policies[data_category]

            if deletion_date is None:
                if policy.retention_period == RetentionPeriod.INDEFINITE:
                    return False
                deletion_date = datetime.now() + timedelta(
                    seconds=policy.retention_period.value
                )

            # Create deletion task
            deletion_task = {
                "user_id": user_id,
                "data_category": data_category.value,
                "deletion_date": deletion_date.isoformat(),
                "policy": {
                    "retention_period": policy.retention_period.value,
                    "auto_delete": policy.auto_delete,
                    "archive_before_delete": policy.archive_before_delete,
                },
            }

            # Store in Redis with expiration
            task_id = "deletion_task_{int(time.time())}_{hashlib.md5(f'{user_id}_{data_category.value}'.encode()).hexdigest()[:8]}"
            self.redis_client.setex(
                "deletion_task:{task_id}",
                int((deletion_date - datetime.now()).total_seconds()),
                json.dumps(deletion_task),
            )

            logger.info("Scheduled data deletion: {task_id} for user {user_id}")
            return True

        except Exception as e:
            logger.error("Failed to schedule data deletion: {e}")
            return False

    def execute_data_deletion(self, user_id: str, data_category: DataCategory) -> bool:
        """Execute data deletion for a user and category."""
        try:
            logger.info(
                "Executing data deletion for user {user_id}, category {data_category.value}"
            )

            # Update retention records
            records = self.get_retention_records(user_id, data_category)
            for record in records:
                record.actual_deletion_date = datetime.now()
                record.deletion_reason = "Scheduled deletion"
                self._store_retention_record(record)

            # Perform actual data deletion based on category
            if data_category == DataCategory.CHAT_HISTORY:
                self._delete_chat_history(user_id)
            elif data_category == DataCategory.SEARCH_HISTORY:
                self._delete_search_history(user_id)
            elif data_category == DataCategory.UPLOADED_DOCUMENTS:
                self._delete_uploaded_documents(user_id)
            elif data_category == DataCategory.EMBEDDINGS:
                self._delete_embeddings(user_id)
            elif data_category == DataCategory.FEEDBACK_DATA:
                self._delete_feedback_data(user_id)
            elif data_category == DataCategory.SESSION_DATA:
                self._delete_session_data(user_id)

            logger.info(
                "Data deletion completed for user {user_id}, category {data_category.value}"
            )
            return True

        except Exception as e:
            logger.error("Failed to execute data deletion: {e}")
            return False

    def _delete_chat_history(self, user_id: str):
        """Delete chat history for a user."""
        # Implementation would delete from chat service
        logger.info("Deleting chat history for user: {user_id}")

    def _delete_search_history(self, user_id: str):
        """Delete search history for a user."""
        # Implementation would delete from search service
        logger.info("Deleting search history for user: {user_id}")

    def _delete_uploaded_documents(self, user_id: str):
        """Delete uploaded documents for a user."""
        # Implementation would delete from document service
        logger.info("Deleting uploaded documents for user: {user_id}")

    def _delete_embeddings(self, user_id: str):
        """Delete embeddings for a user."""
        # Implementation would delete from vector database
        logger.info("Deleting embeddings for user: {user_id}")

    def _delete_feedback_data(self, user_id: str):
        """Delete feedback data for a user."""
        # Implementation would delete from feedback service
        logger.info("Deleting feedback data for user: {user_id}")

    def _delete_session_data(self, user_id: str):
        """Delete session data for a user."""
        # Implementation would delete from session storage
        logger.info("Deleting session data for user: {user_id}")

    def archive_data(self, user_id: str, data_category: DataCategory) -> bool:
        """Archive data before deletion."""
        try:
            # Implementation would archive data to cold storage
            logger.info(
                "Archiving data for user {user_id}, category {data_category.value}"
            )

            # Update retention records
            records = self.get_retention_records(user_id, data_category)
            for record in records:
                record.archived = True
                self._store_retention_record(record)

            return True

        except Exception as e:
            logger.error("Failed to archive data: {e}")
            return False

    def get_data_summary(self, user_id: str) -> Dict[str, Any]:
        """Get data retention summary for a user."""
        summary = {
            "user_id": user_id,
            "data_categories": {},
            "total_records": 0,
            "scheduled_deletions": 0,
        }

        for category in DataCategory:
            records = self.get_retention_records(user_id, category)
            policy = self.retention_policies[category]

            category_summary = {
                "record_count": len(records),
                "retention_period": policy.retention_period.value,
                "auto_delete": policy.auto_delete,
                "scheduled_deletions": len(
                    [
                        r
                        for r in records
                        if r.scheduled_deletion_date
                        and r.scheduled_deletion_date > datetime.now()
                    ]
                ),
            }

            summary["data_categories"][category.value] = category_summary
            summary["total_records"] += len(records)
            summary["scheduled_deletions"] += category_summary["scheduled_deletions"]

        return summary


class OptOutManager:
    """Manages user opt-out requests and preferences."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.opt_out_requests = {}

        logger.info("Opt-Out Manager initialized")

    def create_opt_out_request(
        self,
        user_id: str,
        opt_out_type: OptOutType,
        reason: str = None,
        effective_date: datetime = None,
    ) -> str:
        """Create an opt-out request."""
        if effective_date is None:
            effective_date = datetime.now()

        opt_out_request = OptOutRequest(
            user_id=user_id,
            opt_out_type=opt_out_type,
            request_date=datetime.now(),
            effective_date=effective_date,
            reason=reason,
            status="pending",
        )

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Store request
        self.opt_out_requests[request_id] = opt_out_request
        self._store_opt_out_request(request_id, opt_out_request)

        logger.info("Created opt-out request: {request_id} for user {user_id}")
        return request_id

    def _store_opt_out_request(self, request_id: str, request: OptOutRequest):
        """Store opt-out request in Redis."""
        request_data = {
            "user_id": request.user_id,
            "opt_out_type": request.opt_out_type.value,
            "request_date": request.request_date.isoformat(),
            "effective_date": request.effective_date.isoformat(),
            "reason": request.reason,
            "status": request.status,
        }

        self.redis_client.hset("optout_request:{request_id}", mapping=request_data)

    def activate_opt_out(self, request_id: str) -> bool:
        """Activate an opt-out request."""
        try:
            if request_id not in self.opt_out_requests:
                return False

            request = self.opt_out_requests[request_id]
            request.status = "active"
            self._store_opt_out_request(request_id, request)

            # Apply opt-out based on type
            self._apply_opt_out(request.user_id, request.opt_out_type)

            logger.info("Activated opt-out request: {request_id}")
            return True

        except Exception as e:
            logger.error("Failed to activate opt-out request: {e}")
            return False

    def _apply_opt_out(self, user_id: str, opt_out_type: OptOutType):
        """Apply opt-out based on type."""
        if opt_out_type == OptOutType.DATA_COLLECTION:
            self._apply_data_collection_opt_out(user_id)
        elif opt_out_type == OptOutType.ANALYTICS:
            self._apply_analytics_opt_out(user_id)
        elif opt_out_type == OptOutType.MARKETING:
            self._apply_marketing_opt_out(user_id)
        elif opt_out_type == OptOutType.THIRD_PARTY_SHARING:
            self._apply_third_party_opt_out(user_id)
        elif opt_out_type == OptOutType.AUTOMATED_DECISIONS:
            self._apply_automated_decisions_opt_out(user_id)
        elif opt_out_type == OptOutType.PROFILING:
            self._apply_profiling_opt_out(user_id)

    def _apply_data_collection_opt_out(self, user_id: str):
        """Apply data collection opt-out."""
        logger.info("Applying data collection opt-out for user: {user_id}")
        # Implementation would stop collecting new data

    def _apply_analytics_opt_out(self, user_id: str):
        """Apply analytics opt-out."""
        logger.info("Applying analytics opt-out for user: {user_id}")
        # Implementation would stop analytics tracking

    def _apply_marketing_opt_out(self, user_id: str):
        """Apply marketing opt-out."""
        logger.info("Applying marketing opt-out for user: {user_id}")
        # Implementation would stop marketing communications

    def _apply_third_party_opt_out(self, user_id: str):
        """Apply third-party sharing opt-out."""
        logger.info("Applying third-party sharing opt-out for user: {user_id}")
        # Implementation would stop third-party data sharing

    def _apply_automated_decisions_opt_out(self, user_id: str):
        """Apply automated decisions opt-out."""
        logger.info("Applying automated decisions opt-out for user: {user_id}")
        # Implementation would provide human review option

    def _apply_profiling_opt_out(self, user_id: str):
        """Apply profiling opt-out."""
        logger.info("Applying profiling opt-out for user: {user_id}")
        # Implementation would stop user profiling

    def revoke_opt_out(self, request_id: str) -> bool:
        """Revoke an opt-out request."""
        try:
            if request_id not in self.opt_out_requests:
                return False

            request = self.opt_out_requests[request_id]
            request.status = "revoked"
            self._store_opt_out_request(request_id, request)

            logger.info("Revoked opt-out request: {request_id}")
            return True

        except Exception as e:
            logger.error("Failed to revoke opt-out request: {e}")
            return False

    def get_user_opt_outs(self, user_id: str) -> List[OptOutRequest]:
        """Get all opt-out requests for a user."""
        return [
            request
            for request in self.opt_out_requests.values()
            if request.user_id == user_id
        ]

    def is_opted_out(self, user_id: str, opt_out_type: OptOutType) -> bool:
        """Check if user has opted out of specific type."""
        user_opt_outs = self.get_user_opt_outs(user_id)
        return any(
            opt_out.opt_out_type == opt_out_type and opt_out.status == "active"
            for opt_out in user_opt_outs
        )


# Global instances
global_retention_manager = DataRetentionManager()
global_opt_out_manager = OptOutManager()


def require_data_retention_check():
    """Decorator to check data retention policies."""

    def decorator(f):
        def decorated_function(*args, **kwargs):
            # Check if user has opted out of data collection
            user_id = (
                g.current_user.get("user_id") if hasattr(g, "current_user") else None
            )
            if user_id:
                if global_opt_out_manager.is_opted_out(
                    user_id, OptOutType.DATA_COLLECTION
                ):
                    return (
                        jsonify(
                            {
                                "error": "Data collection not allowed",
                                "reason": "User has opted out of data collection",
                            }
                        ),
                        403,
                    )

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def require_analytics_check():
    """Decorator to check analytics opt-out."""

    def decorator(f):
        def decorated_function(*args, **kwargs):
            # Check if user has opted out of analytics
            user_id = (
                g.current_user.get("user_id") if hasattr(g, "current_user") else None
            )
            if user_id:
                if global_opt_out_manager.is_opted_out(user_id, OptOutType.ANALYTICS):
                    # Skip analytics but allow the request
                    g.skip_analytics = True

            return f(*args, **kwargs)

        return decorated_function

    return decorator
