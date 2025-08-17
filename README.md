# Intelligent Research Assistant

> A production-ready RAG (Retrieval-Augmented Generation) system with multi-agent orchestration that transforms PDFs into intelligent, searchable knowledge bases using advanced NLP, vector search, and AI agents.

**For**: Researchers, developers, and organizations needing intelligent document processing, semantic search, and AI-powered research assistance.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-orange.svg)](https://qdrant.tech)
[![Multi-Agent](https://img.shields.io/badge/Multi--Agent-AI%20Orchestration-purple.svg)](https://github.com)

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Multi-Agent System](#-multi-agent-system)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [API Usage](#-api-usage)
- [Project Structure](#-project-structure)
- [Data Schema](#-data-schema)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Security](#-security)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🔄 **Complete RAG Pipeline (9 Steps)**
- **📄 PDF Processing** - Extract text with page structure preservation and edge case handling
- **🧩 Smart Chunking** - Intelligent text segmentation with configurable overlap and boundary detection
- **🧠 Embedding Generation** - GPU-accelerated vector generation using SentenceTransformers
- **💾 Vector Storage** - High-performance Qdrant database with rich metadata (21+ fields)
- **🔍 Semantic Search** - Real-time similarity search with configurable thresholds
- **🤖 LLM Integration** - Multi-provider support (OpenAI, Ollama, Hugging Face)
- **💬 Conversation Memory** - Multi-turn chat with context preservation
- **📊 Quality Monitoring** - Automatic PDF analysis, OCR detection, and quality scoring
- **📈 Metrics & Logging** - Comprehensive performance tracking and error handling

### 🤖 **Multi-Agent Orchestration System**
- **🧠 Planner Agent** - Task decomposition and workflow planning
- **🔍 Research Agent** - Information retrieval and document search
- **🧮 Reasoner Agent** - Content analysis and response generation
- **⚙️ Executor Agent** - Side effects and system operations
- **🎯 Agent Orchestrator** - Intelligent task coordination and workflow management

### 🌐 **Production-Ready API**
- **📚 Complete REST API** - 15+ endpoints with comprehensive functionality
- **📖 Swagger Documentation** - Interactive API documentation
- **🔒 Input Validation** - Pydantic models for request/response validation
- **⚡ Performance Optimized** - Async operations and batch processing
- **🛡️ Error Handling** - Graceful failure handling and recovery

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FLASK WEB FRAMEWORK                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Upload    │  │    Chat     │  │   Search    │         │
│  │   API       │  │    API      │  │    API      │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 MULTI-AGENT ORCHESTRATION                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Planner   │  │  Research   │  │  Reasoner   │         │
│  │   Agent     │  │   Agent     │  │   Agent     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐                                          │
│  │  Executor   │                                          │
│  │   Agent     │                                          │
│  └─────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    SERVICE LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    Chat     │  │   Search    │  │     LLM     │         │
│  │  Service    │  │  Service    │  │  Service    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Memory    │  │  Metrics    │  │ Document    │         │
│  │  Service    │  │  Service    │  │  Service    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   PIPELINE LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    Text     │  │    Text     │  │ Embedding   │         │
│  │ Extraction  │  │  Chunking   │  │ Generation  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐                                          │
│  │   Vector    │                                          │
│  │  Database   │                                          │
│  └─────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    PDFs     │  │   Qdrant    │  │   Logs      │         │
│  │  (Uploads)  │  │   Vector    │  │  (Loguru)   │         │
│  │             │  │   Database  │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

**Tech Stack:**
- **Backend**: Flask 3.0+, PyMuPDF, SentenceTransformers
- **Vector DB**: Qdrant (Docker)
- **ML**: all-MiniLM-L6-v2 (384D embeddings)
- **LLM**: OpenAI, Ollama, Hugging Face
- **Hardware**: MPS/GPU acceleration support
- **Monitoring**: Loguru logging, real-time metrics

## 🤖 Multi-Agent System

### **Agent Capabilities**

#### **🧠 Planner Agent**
- **Task Decomposition** - Break complex queries into subtasks
- **Tool Selection** - Choose appropriate tools for each task
- **Workflow Planning** - Create execution sequences
- **Priority Assignment** - Manage task dependencies

#### **🔍 Research Agent**
- **Vector Search** - Semantic document retrieval
- **Web Search** - External information gathering (placeholder)
- **API Search** - Third-party data integration (placeholder)
- **Document Retrieval** - Specific document access

#### **🧮 Reasoner Agent**
- **Text Analysis** - Content understanding and processing
- **Content Generation** - Response creation using LLMs
- **Fact Checking** - Information validation (placeholder)
- **Quality Assessment** - Response quality evaluation

#### **⚙️ Executor Agent**
- **Logging Operations** - System event recording
- **Metrics Collection** - Performance data gathering
- **External API Calls** - Third-party integrations (placeholder)
- **File Operations** - System file management (placeholder)

### **Workflow Execution**
```
User Query → Planning → Research → Reasoning → Execution → Response
     ↓         ↓         ↓          ↓          ↓         ↓
  Validation Task Decomp Info Gather Analysis Side Effects Formatted
```

## 📋 Prerequisites

- **Python 3.8+**
- **Docker** (for Qdrant vector database)
- **Git**
- **8GB+ RAM** (for large document processing)
- **macOS/Linux/Windows** (tested on Apple Silicon)

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone <repository-url>
cd Intelligent-Research-Assistant-
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Vector Database
```bash
docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 3. Run Application
```bash
python app.py
```

Visit `http://127.0.0.1:8008` for the API and `http://127.0.0.1:8008/apidocs/` for interactive docs.

## 🌐 API Usage

### **Document Processing**
```bash
# Upload PDF
curl -X POST -F "file=@document.pdf" http://127.0.0.1:8008/upload

# Get document chunks
curl -X GET http://127.0.0.1:8008/documents/{document_id}

# Delete document
curl -X DELETE http://127.0.0.1:8008/documents/{document_id}
```

### **Search & Retrieval**
```bash
# Semantic search
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "limit": 5, "score_threshold": 0.7}' \
  http://127.0.0.1:8008/search

# Get collection statistics
curl -X GET http://127.0.0.1:8008/collection-stats
```

### **Multi-Agent System**
```bash
# Process query with multi-agent system
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "What are the main topics in the uploaded documents?"}' \
  http://127.0.0.1:8008/chat

# Get agent status
curl -X GET http://127.0.0.1:8008/agents

# Test workflow
curl -X POST "http://127.0.0.1:8008/test/workflow?query=test%20query"

# Get agent capabilities
curl -X GET http://127.0.0.1:8008/capabilities
```

### **Agent Management**
```bash
# Activate/deactivate agents
curl -X POST http://127.0.0.1:8008/agents/{agent_type}/activate
curl -X POST http://127.0.0.1:8008/agents/{agent_type}/deactivate

# Reset all agents
curl -X POST http://127.0.0.1:8008/agents/reset

# Get workflow history
curl -X GET http://127.0.0.1:8008/workflows?limit=10
```

## 📁 Project Structure

```
Intelligent-Research-Assistant-/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── logging_config.py               # Logging configuration
├── uploads/                        # PDF upload directory
├── qdrant_storage/                 # Vector database storage
├── logs/                          # Application logs
└── src/                           # Source code
    ├── __init__.py
    ├── main.py                     # FastAPI entry point (alternative)
    ├── pipeline/                   # Document processing pipeline
    │   ├── __init__.py
    │   └── pipeline.py             # Core pipeline functions
    ├── agents/                     # Multi-agent system
    │   ├── __init__.py
    │   ├── base_agent.py           # Base agent class
    │   ├── planner_agent.py        # Task planning agent
    │   ├── research_agent.py       # Information retrieval agent
    │   ├── reasoner_agent.py       # Content analysis agent
    │   ├── executor_agent.py       # System operations agent
    │   └── agent_orchestrator.py   # Agent coordination
    ├── services/                   # Business logic services
    │   ├── __init__.py
    │   ├── chat_service.py         # Chat functionality
    │   ├── search_service.py       # Search operations
    │   ├── llm_service.py          # LLM integration
    │   ├── memory_service.py       # Conversation memory
    │   ├── metrics_service.py      # Performance metrics
    │   ├── document_service.py     # Document operations
    │   └── embedding_service.py    # Embedding operations
    ├── models/                     # Data models
    │   ├── __init__.py
    │   ├── chat.py                 # Chat request/response models
    │   └── search.py               # Search models
    └── api/                        # API endpoints (FastAPI)
        ├── __init__.py
        ├── chat_api.py             # Chat endpoints
        ├── search_api.py           # Search endpoints
        ├── health_api.py           # Health check endpoints
        └── admin_api.py            # Admin endpoints
```

## 📊 Data Schema

### **Document Processing Pipeline**
```json
{
  "document_id": "uuid",
  "filename": "document.pdf",
  "upload_timestamp": "2025-08-17T14:30:00Z",
  "processing_status": "completed",
  "pages": [
    {
      "page_num": 1,
      "text": "extracted content",
      "char_count": 1500,
      "is_empty": false,
      "likely_scanned": false,
      "has_images": true,
      "processing_time_ms": 45.2
    }
  ]
}
```

### **Vector Database Schema**
```json
{
  "id": "vector_uuid",
  "vector": [0.1, 0.2, ...],  // 384-dimensional
  "payload": {
    "document_id": "doc_123",
    "chunk_id": "chunk_001",
    "text_content": "chunk text",
    "page_numbers": [1, 2],
    "char_count": 450,
    "token_count": 120,
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dimensions": 384,
    "chunk_size": 500,
    "chunk_overlap": 50,
    "boundary_type": "paragraph",
    "quality_score": 0.95,
    "extraction_timestamp": "2025-08-17T14:30:00Z",
    "tags": ["research", "technical"],
    "language": "en",
    "has_images": true,
    "is_scanned": false,
    "ocr_confidence": 0.98
  }
}
```

### **Multi-Agent Workflow**
```json
{
  "workflow_id": "workflow_123",
  "query": "user query",
  "execution_time": 10.5,
  "stages": {
    "planning": {
      "agent_id": "planner_001",
      "success": true,
      "data": {
        "subtasks": [...],
        "workflow": [...]
      }
    },
    "research": {
      "agent_id": "research_001",
      "success": true,
      "data": {
        "results": [...],
        "sources": [...]
      }
    },
    "reasoning": {
      "agent_id": "reasoner_001",
      "success": true,
      "data": {
        "generated_content": "response text"
      }
    },
    "execution": {
      "agent_id": "executor_001",
      "success": true,
      "data": {
        "operations_executed": [...]
      }
    }
  },
  "final_response": "generated response",
  "sources": [...],
  "metadata": {
    "total_agents_used": 4,
    "workflow_success": true
  }
}
```

## ⚙️ Configuration

### **Environment Variables**
Create `.env` file:
```env
FLASK_ENV=development
FLASK_DEBUG=True
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216
```

### **Pipeline Settings** (`src/pipeline/pipeline.py`)
```python
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 50        # Overlap between chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformers model
VECTOR_DIMENSIONS = 384   # Embedding dimensions
```

### **LLM Configuration** (`src/services/llm_service.py`)
```python
# Priority order for LLM providers
LLM_PROVIDERS = ["openai", "ollama", "huggingface", "fallback"]
```

## 🚀 Deployment

### **Docker Deployment**
```bash
# Build and run with Docker Compose
docker-compose up -d
```

### **Production Considerations**
- **Environment Variables** - Configure production settings
- **Database Backups** - Regular Qdrant data backups
- **Monitoring** - Set up application monitoring
- **Security** - Implement authentication and authorization
- **Scaling** - Consider horizontal scaling for high load

## 🔧 Troubleshooting

### **Common Issues**

#### **Qdrant Connection Error**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker restart qdrant_container
```

#### **Port Already in Use**
```bash
# Check what's using port 8008
lsof -i :8008

# Kill the process
kill -9 <PID>
```

#### **Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt

# Check Python version
python --version
```

### **Logs and Debugging**
- **Application Logs**: Check `logs/` directory
- **Qdrant Logs**: `docker logs qdrant_container`
- **API Documentation**: `http://127.0.0.1:8008/apidocs/`

## 🔒 Security

### **Security Features**
- **Input Validation** - Pydantic models for request validation
- **File Type Validation** - PDF files only
- **Error Handling** - Graceful failure handling
- **Logging** - Comprehensive audit trail
- **Rate Limiting** - Configurable request limits

### **Best Practices**
- **Environment Variables** - Secure configuration management
- **Docker Security** - Container security best practices
- **API Security** - Implement authentication for production
- **Data Privacy** - Secure document processing

## 🤝 Contributing

### **Development Setup**
```bash
# Fork and clone the repository
git clone <your-fork-url>
cd Intelligent-Research-Assistant-

# Create feature branch
git checkout -b feature/your-feature

# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Submit pull request
```

### **Code Style**
- **Python**: PEP 8 compliance
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit tests for new features
- **Logging**: Proper error and info logging

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Qdrant** - Vector database technology
- **SentenceTransformers** - Embedding generation
- **PyMuPDF** - PDF processing
- **Flask** - Web framework
- **OpenAI** - LLM integration

---

**Built with ❤️ for intelligent document processing and research assistance.**