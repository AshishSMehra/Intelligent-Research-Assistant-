"""
Step 2: Secure Secrets Management

This module implements secure secrets management using AWS KMS and HashiCorp Vault.
It provides encryption, decryption, and secure storage of sensitive data.
"""

import os
import json
import base64
import hashlib
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from loguru import logger

# Optional imports for AWS KMS and Vault
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    logger.warning("AWS SDK not available. Install with: pip install boto3")

try:
    import hvac
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False
    logger.warning("HashiCorp Vault client not available. Install with: pip install hvac")


@dataclass
class SecretMetadata:
    """Metadata for stored secrets."""
    secret_id: str
    name: str
    description: str
    created_at: float
    updated_at: float
    version: int
    tags: Dict[str, str] = None


class SecretsManager(ABC):
    """Abstract base class for secrets management."""
    
    @abstractmethod
    def encrypt(self, plaintext: str, key_id: str = None) -> str:
        """Encrypt plaintext data."""
        pass
    
    @abstractmethod
    def decrypt(self, ciphertext: str, key_id: str = None) -> str:
        """Decrypt ciphertext data."""
        pass
    
    @abstractmethod
    def store_secret(self, name: str, value: str, description: str = None, tags: Dict[str, str] = None) -> str:
        """Store a secret."""
        pass
    
    @abstractmethod
    def get_secret(self, secret_id: str) -> Optional[str]:
        """Retrieve a secret."""
        pass
    
    @abstractmethod
    def update_secret(self, secret_id: str, value: str) -> bool:
        """Update a secret."""
        pass
    
    @abstractmethod
    def delete_secret(self, secret_id: str) -> bool:
        """Delete a secret."""
        pass
    
    @abstractmethod
    def list_secrets(self) -> list:
        """List all secrets."""
        pass


class AWSKMSManager(SecretsManager):
    """AWS KMS-based secrets manager."""
    
    def __init__(self, region_name: str = None, key_id: str = None):
        if not AWS_AVAILABLE:
            raise ImportError("AWS SDK (boto3) is required for AWS KMS manager")
        
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.key_id = key_id or os.getenv("AWS_KMS_KEY_ID")
        
        # Initialize KMS client
        self.kms_client = boto3.client('kms', region_name=self.region_name)
        
        # Initialize S3 client for storing encrypted secrets
        self.s3_client = boto3.client('s3', region_name=self.region_name)
        self.bucket_name = os.getenv("AWS_S3_BUCKET", "intelligent-research-secrets")
        
        # Create bucket if it doesn't exist
        self._ensure_bucket_exists()
        
        logger.info(f"AWS KMS Manager initialized in region: {self.region_name}")
    
    def _ensure_bucket_exists(self):
        """Ensure S3 bucket exists for storing encrypted secrets."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Bucket doesn't exist, create it
                try:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.region_name}
                    )
                    logger.info(f"Created S3 bucket: {self.bucket_name}")
                except ClientError as create_error:
                    logger.error(f"Failed to create S3 bucket: {create_error}")
            else:
                logger.error(f"Error checking S3 bucket: {e}")
    
    def encrypt(self, plaintext: str, key_id: str = None) -> str:
        """Encrypt plaintext using AWS KMS."""
        try:
            key_id = key_id or self.key_id
            if not key_id:
                raise ValueError("KMS key ID is required for encryption")
            
            response = self.kms_client.encrypt(
                KeyId=key_id,
                Plaintext=plaintext.encode('utf-8')
            )
            
            # Return base64 encoded ciphertext
            return base64.b64encode(response['CiphertextBlob']).decode('utf-8')
            
        except ClientError as e:
            logger.error(f"AWS KMS encryption failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, ciphertext: str, key_id: str = None) -> str:
        """Decrypt ciphertext using AWS KMS."""
        try:
            # Decode base64 ciphertext
            ciphertext_blob = base64.b64decode(ciphertext.encode('utf-8'))
            
            response = self.kms_client.decrypt(
                CiphertextBlob=ciphertext_blob
            )
            
            return response['Plaintext'].decode('utf-8')
            
        except ClientError as e:
            logger.error(f"AWS KMS decryption failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def store_secret(self, name: str, value: str, description: str = None, tags: Dict[str, str] = None) -> str:
        """Store a secret in S3 with KMS encryption."""
        try:
            import time
            
            # Encrypt the secret value
            encrypted_value = self.encrypt(value)
            
            # Create metadata
            metadata = SecretMetadata(
                secret_id=f"secret_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}",
                name=name,
                description=description or "",
                created_at=time.time(),
                updated_at=time.time(),
                version=1,
                tags=tags or {}
            )
            
            # Store encrypted value and metadata
            secret_data = {
                "encrypted_value": encrypted_value,
                "metadata": {
                    "secret_id": metadata.secret_id,
                    "name": metadata.name,
                    "description": metadata.description,
                    "created_at": metadata.created_at,
                    "updated_at": metadata.updated_at,
                    "version": metadata.version,
                    "tags": metadata.tags
                }
            }
            
            # Store in S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=f"secrets/{metadata.secret_id}.json",
                Body=json.dumps(secret_data),
                ContentType='application/json'
            )
            
            logger.info(f"Stored secret: {metadata.secret_id}")
            return metadata.secret_id
            
        except Exception as e:
            logger.error(f"Failed to store secret {name}: {e}")
            raise
    
    def get_secret(self, secret_id: str) -> Optional[str]:
        """Retrieve and decrypt a secret."""
        try:
            # Get secret from S3
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=f"secrets/{secret_id}.json"
            )
            
            secret_data = json.loads(response['Body'].read())
            encrypted_value = secret_data['encrypted_value']
            
            # Decrypt the value
            decrypted_value = self.decrypt(encrypted_value)
            
            logger.info(f"Retrieved secret: {secret_id}")
            return decrypted_value
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"Secret not found: {secret_id}")
                return None
            else:
                logger.error(f"Failed to retrieve secret {secret_id}: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_id}: {e}")
            raise
    
    def update_secret(self, secret_id: str, value: str) -> bool:
        """Update an existing secret."""
        try:
            # Get existing metadata
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=f"secrets/{secret_id}.json"
            )
            
            secret_data = json.loads(response['Body'].read())
            metadata = secret_data['metadata']
            
            # Encrypt new value
            encrypted_value = self.encrypt(value)
            
            # Update metadata
            metadata['updated_at'] = time.time()
            metadata['version'] += 1
            
            # Store updated secret
            updated_secret_data = {
                "encrypted_value": encrypted_value,
                "metadata": metadata
            }
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=f"secrets/{secret_id}.json",
                Body=json.dumps(updated_secret_data),
                ContentType='application/json'
            )
            
            logger.info(f"Updated secret: {secret_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update secret {secret_id}: {e}")
            return False
    
    def delete_secret(self, secret_id: str) -> bool:
        """Delete a secret."""
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=f"secrets/{secret_id}.json"
            )
            
            logger.info(f"Deleted secret: {secret_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete secret {secret_id}: {e}")
            return False
    
    def list_secrets(self) -> list:
        """List all secrets."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix="secrets/"
            )
            
            secrets = []
            for obj in response.get('Contents', []):
                secret_id = obj['Key'].replace('secrets/', '').replace('.json', '')
                secrets.append(secret_id)
            
            return secrets
            
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []


class VaultManager(SecretsManager):
    """HashiCorp Vault-based secrets manager."""
    
    def __init__(self, vault_url: str = None, token: str = None, mount_point: str = "secret"):
        if not VAULT_AVAILABLE:
            raise ImportError("HashiCorp Vault client (hvac) is required for Vault manager")
        
        self.vault_url = vault_url or os.getenv("VAULT_URL", "http://localhost:8200")
        self.token = token or os.getenv("VAULT_TOKEN")
        self.mount_point = mount_point
        
        # Initialize Vault client
        self.client = hvac.Client(url=self.vault_url, token=self.token)
        
        # Test connection
        if not self.client.is_authenticated():
            raise ValueError("Failed to authenticate with Vault")
        
        logger.info(f"Vault Manager initialized with URL: {self.vault_url}")
    
    def encrypt(self, plaintext: str, key_id: str = None) -> str:
        """Encrypt plaintext using Vault's transit engine."""
        try:
            response = self.client.secrets.transit.encrypt_data(
                name=key_id or "default-key",
                plaintext=base64.b64encode(plaintext.encode('utf-8')).decode('utf-8')
            )
            
            return response['data']['ciphertext']
            
        except Exception as e:
            logger.error(f"Vault encryption failed: {e}")
            raise
    
    def decrypt(self, ciphertext: str, key_id: str = None) -> str:
        """Decrypt ciphertext using Vault's transit engine."""
        try:
            response = self.client.secrets.transit.decrypt_data(
                name=key_id or "default-key",
                ciphertext=ciphertext
            )
            
            # Decode base64 plaintext
            return base64.b64decode(response['data']['plaintext']).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Vault decryption failed: {e}")
            raise
    
    def store_secret(self, name: str, value: str, description: str = None, tags: Dict[str, str] = None) -> str:
        """Store a secret in Vault."""
        try:
            import time
            
            # Create secret ID
            secret_id = f"secret_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
            
            # Prepare secret data
            secret_data = {
                "value": value,
                "description": description or "",
                "created_at": time.time(),
                "updated_at": time.time(),
                "version": 1,
                "tags": tags or {}
            }
            
            # Store in Vault
            self.client.secrets.kv.v2.create_or_update_secret(
                path=secret_id,
                secret=secret_data,
                mount_point=self.mount_point
            )
            
            logger.info(f"Stored secret in Vault: {secret_id}")
            return secret_id
            
        except Exception as e:
            logger.error(f"Failed to store secret {name} in Vault: {e}")
            raise
    
    def get_secret(self, secret_id: str) -> Optional[str]:
        """Retrieve a secret from Vault."""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=secret_id,
                mount_point=self.mount_point
            )
            
            secret_data = response['data']['data']
            value = secret_data['value']
            
            logger.info(f"Retrieved secret from Vault: {secret_id}")
            return value
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_id} from Vault: {e}")
            return None
    
    def update_secret(self, secret_id: str, value: str) -> bool:
        """Update an existing secret in Vault."""
        try:
            import time
            
            # Get existing secret data
            response = self.client.secrets.kv.v2.read_secret_version(
                path=secret_id,
                mount_point=self.mount_point
            )
            
            secret_data = response['data']['data']
            
            # Update value and metadata
            secret_data['value'] = value
            secret_data['updated_at'] = time.time()
            secret_data['version'] += 1
            
            # Store updated secret
            self.client.secrets.kv.v2.create_or_update_secret(
                path=secret_id,
                secret=secret_data,
                mount_point=self.mount_point
            )
            
            logger.info(f"Updated secret in Vault: {secret_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update secret {secret_id} in Vault: {e}")
            return False
    
    def delete_secret(self, secret_id: str) -> bool:
        """Delete a secret from Vault."""
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=secret_id,
                mount_point=self.mount_point
            )
            
            logger.info(f"Deleted secret from Vault: {secret_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete secret {secret_id} from Vault: {e}")
            return False
    
    def list_secrets(self) -> list:
        """List all secrets in Vault."""
        try:
            response = self.client.secrets.kv.v2.list_secrets(
                path="",
                mount_point=self.mount_point
            )
            
            return response['data']['keys']
            
        except Exception as e:
            logger.error(f"Failed to list secrets from Vault: {e}")
            return []


class SecretsManager:
    """Main secrets manager that can use different backends."""
    
    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self.manager = self._initialize_backend()
        
        logger.info(f"Secrets Manager initialized with backend: {backend}")
    
    def _initialize_backend(self) -> SecretsManager:
        """Initialize the appropriate secrets backend."""
        if self.backend == "aws" or (self.backend == "auto" and AWS_AVAILABLE):
            try:
                return AWSKMSManager()
            except Exception as e:
                logger.warning(f"Failed to initialize AWS KMS: {e}")
        
        if self.backend == "vault" or (self.backend == "auto" and VAULT_AVAILABLE):
            try:
                return VaultManager()
            except Exception as e:
                logger.warning(f"Failed to initialize Vault: {e}")
        
        # Fallback to environment variables
        logger.warning("No secure backend available, using environment variables")
        return EnvironmentSecretsManager()
    
    def encrypt(self, plaintext: str, key_id: str = None) -> str:
        """Encrypt plaintext data."""
        return self.manager.encrypt(plaintext, key_id)
    
    def decrypt(self, ciphertext: str, key_id: str = None) -> str:
        """Decrypt ciphertext data."""
        return self.manager.decrypt(ciphertext, key_id)
    
    def store_secret(self, name: str, value: str, description: str = None, tags: Dict[str, str] = None) -> str:
        """Store a secret."""
        return self.manager.store_secret(name, value, description, tags)
    
    def get_secret(self, secret_id: str) -> Optional[str]:
        """Retrieve a secret."""
        return self.manager.get_secret(secret_id)
    
    def update_secret(self, secret_id: str, value: str) -> bool:
        """Update a secret."""
        return self.manager.update_secret(secret_id, value)
    
    def delete_secret(self, secret_id: str) -> bool:
        """Delete a secret."""
        return self.manager.delete_secret(secret_id)
    
    def list_secrets(self) -> list:
        """List all secrets."""
        return self.manager.list_secrets()


class EnvironmentSecretsManager(SecretsManager):
    """Fallback secrets manager using environment variables."""
    
    def __init__(self):
        self.secrets = {}
        logger.info("Using environment variables for secrets management")
    
    def encrypt(self, plaintext: str, key_id: str = None) -> str:
        """Simple base64 encoding for environment variables."""
        return base64.b64encode(plaintext.encode('utf-8')).decode('utf-8')
    
    def decrypt(self, ciphertext: str, key_id: str = None) -> str:
        """Simple base64 decoding for environment variables."""
        return base64.b64decode(ciphertext.encode('utf-8')).decode('utf-8')
    
    def store_secret(self, name: str, value: str, description: str = None, tags: Dict[str, str] = None) -> str:
        """Store secret in memory (for demo purposes)."""
        import time
        secret_id = f"env_secret_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        self.secrets[secret_id] = value
        return secret_id
    
    def get_secret(self, secret_id: str) -> Optional[str]:
        """Get secret from memory."""
        return self.secrets.get(secret_id)
    
    def update_secret(self, secret_id: str, value: str) -> bool:
        """Update secret in memory."""
        if secret_id in self.secrets:
            self.secrets[secret_id] = value
            return True
        return False
    
    def delete_secret(self, secret_id: str) -> bool:
        """Delete secret from memory."""
        if secret_id in self.secrets:
            del self.secrets[secret_id]
            return True
        return False
    
    def list_secrets(self) -> list:
        """List all secrets in memory."""
        return list(self.secrets.keys()) 