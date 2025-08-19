"""
Step 1: Role-Based Access Control (RBAC)

This module implements comprehensive RBAC for securing FastAPI routes.
It provides role-based permissions, user management, and route protection.
"""

import hashlib
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Set

import jwt
import redis
from flask import g, jsonify, request
from loguru import logger


class Permission(Enum):
    """Available permissions in the system."""

    # Document permissions
    UPLOAD_DOCUMENTS = "upload_documents"
    VIEW_DOCUMENTS = "view_documents"
    DELETE_DOCUMENTS = "delete_documents"

    # Search permissions
    SEARCH_DOCUMENTS = "search_documents"
    VIEW_SEARCH_HISTORY = "view_search_history"

    # Chat permissions
    USE_CHAT = "use_chat"
    VIEW_CHAT_HISTORY = "view_chat_history"

    # Agent permissions
    USE_AGENTS = "use_agents"
    MANAGE_AGENTS = "manage_agents"

    # Fine-tuning permissions
    ACCESS_FINETUNING = "access_finetuning"
    MANAGE_MODELS = "manage_models"

    # RLHF permissions
    ACCESS_RLHF = "access_rlhf"
    PROVIDE_FEEDBACK = "provide_feedback"

    # Admin permissions
    VIEW_METRICS = "view_metrics"
    MANAGE_USERS = "manage_users"
    SYSTEM_ADMIN = "system_admin"


@dataclass
class Role:
    """Role definition with permissions."""

    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class User:
    """User definition with roles and metadata."""

    user_id: str
    username: str
    email: str
    roles: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RBACManager:
    """Role-Based Access Control Manager."""

    def __init__(
        self, redis_url: str = "redis://localhost:6379", jwt_secret: str = None
    ):
        self.redis_client = redis.from_url(redis_url)
        self.jwt_secret = jwt_secret or os.getenv("JWT_SECRET", "default-secret-key")
        self.jwt_algorithm = "HS256"
        self.jwt_expiry = 3600  # 1 hour

        # Initialize default roles
        self._initialize_default_roles()

        # Initialize default users
        self._initialize_default_users()

    def _initialize_default_roles(self):
        """Initialize default roles with permissions."""
        default_roles = {
            "admin": Role(
                name="admin",
                description="System administrator with full access",
                permissions={
                    Permission.UPLOAD_DOCUMENTS,
                    Permission.VIEW_DOCUMENTS,
                    Permission.DELETE_DOCUMENTS,
                    Permission.SEARCH_DOCUMENTS,
                    Permission.VIEW_SEARCH_HISTORY,
                    Permission.USE_CHAT,
                    Permission.VIEW_CHAT_HISTORY,
                    Permission.USE_AGENTS,
                    Permission.MANAGE_AGENTS,
                    Permission.ACCESS_FINETUNING,
                    Permission.MANAGE_MODELS,
                    Permission.ACCESS_RLHF,
                    Permission.PROVIDE_FEEDBACK,
                    Permission.VIEW_METRICS,
                    Permission.MANAGE_USERS,
                    Permission.SYSTEM_ADMIN,
                },
            ),
            "researcher": Role(
                name="researcher",
                description="Research user with document and chat access",
                permissions={
                    Permission.UPLOAD_DOCUMENTS,
                    Permission.VIEW_DOCUMENTS,
                    Permission.SEARCH_DOCUMENTS,
                    Permission.VIEW_SEARCH_HISTORY,
                    Permission.USE_CHAT,
                    Permission.VIEW_CHAT_HISTORY,
                    Permission.USE_AGENTS,
                    Permission.ACCESS_FINETUNING,
                    Permission.ACCESS_RLHF,
                    Permission.PROVIDE_FEEDBACK,
                    Permission.VIEW_METRICS,
                },
            ),
            "user": Role(
                name="user",
                description="Basic user with limited access",
                permissions={
                    Permission.VIEW_DOCUMENTS,
                    Permission.SEARCH_DOCUMENTS,
                    Permission.USE_CHAT,
                    Permission.VIEW_CHAT_HISTORY,
                    Permission.PROVIDE_FEEDBACK,
                },
            ),
            "guest": Role(
                name="guest",
                description="Guest user with minimal access",
                permissions={Permission.VIEW_DOCUMENTS, Permission.SEARCH_DOCUMENTS},
            ),
        }

        for role_name, role in default_roles.items():
            self.create_role(role)

    def _initialize_default_users(self):
        """Initialize default users."""
        default_users = [
            User(
                user_id="admin-001",
                username="admin",
                email="admin@example.com",
                roles=["admin"],
            ),
            User(
                user_id="researcher-001",
                username="researcher",
                email="researcher@example.com",
                roles=["researcher"],
            ),
            User(
                user_id="user-001",
                username="user",
                email="user@example.com",
                roles=["user"],
            ),
        ]

        for user in default_users:
            self.create_user(user)

    def create_role(self, role: Role) -> bool:
        """Create a new role."""
        try:
            role_data = {
                "name": role.name,
                "description": role.description,
                "permissions": ",".join([perm.value for perm in role.permissions]),
                "created_at": role.created_at,
                "updated_at": role.updated_at,
            }

            self.redis_client.hset(f"role:{role.name}", mapping=role_data)
            logger.info(f"Created role: {role.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create role {role.name}: {e}")
            return False

    def get_role(self, role_name: str) -> Optional[Role]:
        """Get a role by name."""
        try:
            role_data = self.redis_client.hgetall(f"role:{role_name}")
            if not role_data:
                return None

            permissions = {
                Permission(perm)
                for perm in role_data.get("permissions", "").split(",")
                if perm
            }

            return Role(
                name=role_data["name"],
                description=role_data["description"],
                permissions=permissions,
                created_at=float(role_data.get("created_at", 0)),
                updated_at=float(role_data.get("updated_at", 0)),
            )

        except Exception as e:
            logger.error(f"Failed to get role {role_name}: {e}")
            return None

    def update_role(self, role_name: str, permissions: Set[Permission]) -> bool:
        """Update role permissions."""
        try:
            role = self.get_role(role_name)
            if not role:
                return False

            role.permissions = permissions
            role.updated_at = time.time()

            role_data = {
                "permissions": ",".join([perm.value for perm in permissions]),
                "updated_at": role.updated_at,
            }

            self.redis_client.hset(f"role:{role_name}", mapping=role_data)
            logger.info(f"Updated role: {role_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to update role {role_name}: {e}")
            return False

    def delete_role(self, role_name: str) -> bool:
        """Delete a role."""
        try:
            self.redis_client.delete(f"role:{role_name}")
            logger.info(f"Deleted role: {role_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete role {role_name}: {e}")
            return False

    def create_user(self, user: User) -> bool:
        """Create a new user."""
        try:
            user_data = {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": ",".join(user.roles),
                "is_active": str(user.is_active),
                "created_at": str(user.created_at),
                "last_login": str(user.last_login) if user.last_login else "",
                "metadata": str(user.metadata),
            }

            self.redis_client.hset(f"user:{user.user_id}", mapping=user_data)
            logger.info(f"Created user: {user.username}")
            return True

        except Exception as e:
            logger.error(f"Failed to create user {user.username}: {e}")
            return False

    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        try:
            user_data = self.redis_client.hgetall(f"user:{user_id}")
            if not user_data:
                return None

            return User(
                user_id=user_data["user_id"],
                username=user_data["username"],
                email=user_data["email"],
                roles=(
                    user_data.get("roles", "").split(",")
                    if user_data.get("roles")
                    else []
                ),
                is_active=user_data.get("is_active", "True").lower() == "true",
                created_at=float(user_data.get("created_at", 0)),
                last_login=(
                    float(user_data.get("last_login", 0))
                    if user_data.get("last_login")
                    else None
                ),
                metadata=eval(user_data.get("metadata", "{}")),
            )

        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            return None

    def update_user_roles(self, user_id: str, roles: List[str]) -> bool:
        """Update user roles."""
        try:
            user_data = {"roles": ",".join(roles)}
            self.redis_client.hset(f"user:{user_id}", mapping=user_data)
            logger.info(f"Updated roles for user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update roles for user {user_id}: {e}")
            return False

    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        try:
            self.redis_client.delete(f"user:{user_id}")
            logger.info(f"Deleted user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return JWT token."""
        try:
            # Find user by username
            user_keys = self.redis_client.keys("user:*")
            user = None

            for key in user_keys:
                user_data = self.redis_client.hgetall(key)
                if user_data.get("username") == username:
                    user = self.get_user(user_data["user_id"])
                    break

            if not user or not user.is_active:
                return None

            # In production, verify password hash
            # For demo, accept any password
            if password:  # Simple check for demo
                # Update last login
                self.redis_client.hset(
                    f"user:{user.user_id}", "last_login", str(time.time())
                )

                # Generate JWT token
                payload = {
                    "user_id": user.user_id,
                    "username": user.username,
                    "roles": user.roles,
                    "exp": time.time() + self.jwt_expiry,
                }

                token = jwt.encode(
                    payload, self.jwt_secret, algorithm=self.jwt_algorithm
                )
                return token

            return None

        except Exception as e:
            logger.error(f"Authentication failed for {username}: {e}")
            return None

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(
                token, self.jwt_secret, algorithms=[self.jwt_algorithm]
            )
            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None

    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission."""
        try:
            user = self.get_user(user_id)
            if not user or not user.is_active:
                return False

            # Check all user roles for permission
            for role_name in user.roles:
                role = self.get_role(role_name)
                if role and permission in role.permissions:
                    return True

            return False

        except Exception as e:
            logger.error(f"Permission check failed for user {user_id}: {e}")
            return False

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user."""
        try:
            user = self.get_user(user_id)
            if not user or not user.is_active:
                return set()

            permissions = set()
            for role_name in user.roles:
                role = self.get_role(role_name)
                if role:
                    permissions.update(role.permissions)

            return permissions

        except Exception as e:
            logger.error(f"Failed to get permissions for user {user_id}: {e}")
            return set()


def require_permission(permission: Permission):
    """Decorator to require specific permission for Flask routes."""

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get token from request
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return (
                    jsonify({"error": "Missing or invalid authorization header"}),
                    401,
                )

            token = auth_header.split(" ")[1]

            # Verify token
            rbac_manager = RBACManager()
            payload = rbac_manager.verify_token(token)
            if not payload:
                return jsonify({"error": "Invalid or expired token"}), 401

            # Check permission
            if not rbac_manager.has_permission(payload["user_id"], permission):
                return jsonify({"error": "Insufficient permissions"}), 403

            # Add user info to request context
            g.current_user = payload

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def require_role(role_name: str):
    """Decorator to require specific role for Flask routes."""

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get token from request
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return (
                    jsonify({"error": "Missing or invalid authorization header"}),
                    401,
                )

            token = auth_header.split(" ")[1]

            # Verify token
            rbac_manager = RBACManager()
            payload = rbac_manager.verify_token(token)
            if not payload:
                return jsonify({"error": "Invalid or expired token"}), 401

            # Check role
            if role_name not in payload.get("roles", []):
                return jsonify({"error": "Insufficient role"}), 403

            # Add user info to request context
            g.current_user = payload

            return f(*args, **kwargs)

        return decorated_function

    return decorator
