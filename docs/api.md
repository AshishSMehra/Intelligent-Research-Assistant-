# 🔧 API Documentation

Complete API reference for the Intelligent Research Assistant.

## 📋 **Table of Contents**

- [Authentication](#authentication)
- [Base URL](#base-url)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Core Endpoints](#core-endpoints)
- [Agent Endpoints](#agent-endpoints)
- [Admin Endpoints](#admin-endpoints)

---

## 🔐 **Authentication**

All API endpoints require JWT authentication.

### **Authentication Header**
```http
Authorization: Bearer <jwt_token>
```

### **Login Endpoint**
```http
POST /auth/login
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "user123",
    "username": "your_username",
    "roles": ["researcher"],
    "permissions": ["upload", "search", "chat"]
  }
}
```

---

## 🌐 **Base URL**

- **Development**: `http://localhost:8008`
- **Staging**: `https://staging-api.example.com`
- **Production**: `https://api.example.com`

---

## ⚠️ **Error Handling**

### **Error Response Format**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "query",
      "issue": "Query cannot be empty"
    },
    "timestamp": "2024-08-18T12:00:00Z",
    "request_id": "req_123456789"
  }
}
```

### **HTTP Status Codes**
| Code | Description |
|------|-------------|
| `200` | Success |
| `201` | Created |
| `400` | Bad Request |
| `401` | Unauthorized |
| `403` | Forbidden |
| `404` | Not Found |
| `429` | Rate Limited |
| `500` | Internal Server Error |

---

## 🚦 **Rate Limiting**

### **Rate Limits**
- **General API**: 100 requests per minute
- **Upload Endpoint**: 10 requests per minute
- **Chat Endpoint**: 50 requests per minute
- **Search Endpoint**: 200 requests per minute

### **Rate Limit Headers**
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
Retry-After: 60
```

---

## 🎯 **Core Endpoints**

### **1. Upload Document**

**Endpoint:** `POST /upload`

**Description:** Upload and process documents (PDF, DOCX, TXT)

**Headers:**
```http
Authorization: Bearer <jwt_token>
Content-Type: multipart/form-data
```

**Request:**
```bash
curl -X POST -F "file=@document.pdf" \
  -H "Authorization: Bearer <jwt_token>" \
  http://localhost:8008/upload
```

**Response:**
```json
{
  "success": true,
  "document_id": "doc_123456789",
  "filename": "document.pdf",
  "file_size": 1024000,
  "pages_processed": 15,
  "chunks_created": 45,
  "processing_time": 2.5,
  "metadata": {
    "title": "Research Paper",
    "author": "John Doe",
    "date_created": "2024-01-15",
    "language": "en"
  },
  "status": "completed"
}
```

### **2. Chat with RAG**

**Endpoint:** `POST /chat`

**Description:** Chat with the AI assistant using RAG

**Headers:**
```http
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Request:**
```json
{
  "query": "What are the main findings in the research papers about machine learning?",
  "context": "research",
  "stream": false,
  "max_tokens": 1000,
  "temperature": 0.7,
  "include_sources": true
}
```

**Response:**
```json
{
  "response": "Based on the research papers in your collection, the main findings about machine learning include...",
  "sources": [
    {
      "document_id": "doc_123456789",
      "page_number": 5,
      "chunk_id": "chunk_001",
      "text": "The study found that deep learning models...",
      "similarity_score": 0.95
    }
  ],
  "metadata": {
    "tokens_used": 450,
    "response_time": 1.2,
    "model_used": "gpt-3.5-turbo",
    "confidence_score": 0.92
  }
}
```

### **3. Search Documents**

**Endpoint:** `POST /search`

**Description:** Perform semantic search across uploaded documents

**Headers:**
```http
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Request:**
```json
{
  "query": "artificial intelligence applications",
  "limit": 10,
  "threshold": 0.7,
  "filters": {
    "document_ids": ["doc_123", "doc_456"],
    "date_range": {
      "start": "2023-01-01",
      "end": "2024-12-31"
    },
    "tags": ["research", "technology"]
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "document_id": "doc_123456789",
      "chunk_id": "chunk_001",
      "page_number": 3,
      "text": "Artificial intelligence has found applications in various fields...",
      "similarity_score": 0.95,
      "metadata": {
        "title": "AI Applications in Healthcare",
        "author": "Dr. Smith",
        "date": "2024-03-15",
        "tags": ["AI", "healthcare", "research"]
      }
    }
  ],
  "total_results": 25,
  "search_time": 0.15
}
```

---

## 🤖 **Agent Endpoints**

### **1. Get Agent Status**

**Endpoint:** `GET /agents`

**Description:** Get status and metrics for all agents

**Headers:**
```http
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "agents": {
    "planner": {
      "status": "active",
      "tasks_processed": 150,
      "success_rate": 0.98,
      "average_response_time": 0.5,
      "last_activity": "2024-08-18T12:00:00Z"
    },
    "research": {
      "status": "active",
      "searches_performed": 300,
      "success_rate": 0.95,
      "average_response_time": 1.2,
      "last_activity": "2024-08-18T12:00:00Z"
    },
    "reasoner": {
      "status": "active",
      "analyses_performed": 200,
      "success_rate": 0.97,
      "average_response_time": 2.1,
      "last_activity": "2024-08-18T12:00:00Z"
    },
    "executor": {
      "status": "active",
      "actions_executed": 50,
      "success_rate": 0.99,
      "average_response_time": 0.8,
      "last_activity": "2024-08-18T12:00:00Z"
    }
  },
  "orchestrator": {
    "status": "active",
    "workflows_processed": 100,
    "active_workflows": 5,
    "queue_size": 2
  }
}
```

### **2. Activate Agent**

**Endpoint:** `POST /agents/{agent_type}/activate`

**Description:** Activate a specific agent

**Headers:**
```http
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Request:**
```json
{
  "reason": "Manual activation",
  "configuration": {
    "max_concurrent_tasks": 5,
    "timeout": 30
  }
}
```

**Response:**
```json
{
  "success": true,
  "agent_type": "planner",
  "status": "active",
  "message": "Agent activated successfully",
  "activation_time": "2024-08-18T12:00:00Z"
}
```

### **3. Get Workflow History**

**Endpoint:** `GET /workflows`

**Description:** Get history of agent workflows

**Headers:**
```http
Authorization: Bearer <jwt_token>
```

**Query Parameters:**
- `limit`: Number of workflows to return (default: 20)
- `offset`: Number of workflows to skip (default: 0)
- `status`: Filter by status (completed, failed, running)
- `agent_type`: Filter by agent type

**Response:**
```json
{
  "workflows": [
    {
      "workflow_id": "wf_123456789",
      "user_id": "user_123",
      "query": "Analyze the research papers about AI",
      "status": "completed",
      "created_at": "2024-08-18T10:00:00Z",
      "completed_at": "2024-08-18T10:05:00Z",
      "duration": 300,
      "agents_used": ["planner", "research", "reasoner"],
      "result": {
        "summary": "Analysis completed successfully",
        "documents_analyzed": 5,
        "insights_generated": 10
      }
    }
  ],
  "total_workflows": 150,
  "pagination": {
    "limit": 20,
    "offset": 0,
    "has_more": true
  }
}
```

---

## 🔧 **Admin Endpoints**

### **1. Health Check**

**Endpoint:** `GET /admin/health`

**Description:** Check system health and status

**Headers:**
```http
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-08-18T12:00:00Z",
  "version": "1.0.0",
  "services": {
    "api": {
      "status": "healthy",
      "uptime": 86400,
      "memory_usage": "45%",
      "cpu_usage": "30%"
    },
    "qdrant": {
      "status": "healthy",
      "collections": 1,
      "total_vectors": 15000,
      "storage_used": "2.5GB"
    },
    "redis": {
      "status": "healthy",
      "connected_clients": 5,
      "memory_used": "128MB",
      "keys": 1500
    },
    "worker": {
      "status": "healthy",
      "active_jobs": 2,
      "queue_size": 5
    }
  },
  "system": {
    "disk_usage": "60%",
    "memory_usage": "70%",
    "cpu_usage": "45%",
    "network": "stable"
  }
}
```

### **2. Get Model Information**

**Endpoint:** `GET /model-info`

**Description:** Get information about AI models and configurations

**Headers:**
```http
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "embedding_model": {
    "name": "all-MiniLM-L6-v2",
    "version": "2.2.0",
    "dimensions": 384,
    "max_length": 256,
    "device": "cpu"
  },
  "llm_model": {
    "name": "gpt-3.5-turbo",
    "provider": "openai",
    "max_tokens": 4096,
    "temperature": 0.7,
    "status": "available"
  },
  "fine_tuned_models": [
    {
      "name": "research-assistant-v1",
      "base_model": "gpt-3.5-turbo",
      "version": "1.0.0",
      "training_date": "2024-08-01",
      "performance": {
        "accuracy": 0.92,
        "bleu_score": 0.85
      }
    }
  ],
  "available_providers": ["openai", "ollama", "huggingface"]
}
```

### **3. Get Collection Information**

**Endpoint:** `GET /collection-info`

**Description:** Get information about vector database collections

**Headers:**
```http
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "collection_name": "documents",
  "status": "active",
  "vector_size": 384,
  "distance_metric": "cosine",
  "total_points": 15000,
  "total_vectors": 15000,
  "indexed_vectors": 15000,
  "storage_size": "2.5GB",
  "created_at": "2024-08-01T00:00:00Z",
  "last_updated": "2024-08-18T12:00:00Z",
  "configuration": {
    "optimizers_config": {
      "default_segment_number": 2,
      "memmap_threshold": 20000
    },
    "hnsw_config": {
      "m": 16,
      "ef_construct": 100
    }
  }
}
```

---

## 📊 **Response Examples**

### **Success Response**
```json
{
  "success": true,
  "data": {
    // Response data here
  },
  "metadata": {
    "timestamp": "2024-08-18T12:00:00Z",
    "request_id": "req_123456789",
    "processing_time": 0.15
  }
}
```

### **Error Response**
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "query",
      "issue": "Query cannot be empty"
    },
    "timestamp": "2024-08-18T12:00:00Z",
    "request_id": "req_123456789"
  }
}
```

---

## 🔧 **SDK Examples**

### **Python SDK**
```python
import requests

class IntelligentResearchAssistant:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def upload_document(self, file_path):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f'{self.base_url}/upload',
                headers={'Authorization': self.headers['Authorization']},
                files=files
            )
        return response.json()
    
    def chat(self, query, context="general"):
        data = {
            'query': query,
            'context': context
        }
        response = requests.post(
            f'{self.base_url}/chat',
            headers=self.headers,
            json=data
        )
        return response.json()
    
    def search(self, query, limit=10):
        data = {
            'query': query,
            'limit': limit
        }
        response = requests.post(
            f'{self.base_url}/search',
            headers=self.headers,
            json=data
        )
        return response.json()

# Usage
client = IntelligentResearchAssistant('http://localhost:8008', 'your_api_key')
result = client.chat("What is machine learning?")
print(result['response'])
```

---

*This API documentation provides comprehensive coverage of all endpoints, request/response formats, and usage examples for the Intelligent Research Assistant platform.* 