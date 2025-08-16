# Intelligent Research Assistant - Modular Structure

This document explains the new modular architecture of the Intelligent Research Assistant.

## 🏗️ Architecture Overview

The application has been refactored into a clean, modular structure following best practices:

```
src/
├── __init__.py              # Package initialization
├── main.py                  # FastAPI application entry point
├── models/                  # Data models and schemas
│   ├── __init__.py
│   ├── chat.py             # Chat-related models
│   └── search.py           # Search-related models
├── services/               # Business logic layer
│   ├── __init__.py
│   ├── chat_service.py     # Chat functionality
│   ├── search_service.py   # Search functionality
│   ├── document_service.py # Document processing
│   └── embedding_service.py # Embedding generation
├── api/                    # API endpoints
│   ├── __init__.py
│   ├── chat_api.py         # Chat endpoints
│   ├── search_api.py       # Search endpoints
│   └── health_api.py       # Health check endpoints
└── pipeline/               # Existing pipeline (unchanged)
    └── pipeline.py         # Document processing pipeline
```

## 🚀 Step 1: Receive User Query - Implementation

### FastAPI Endpoint: `/chat`

The chat endpoint receives user queries with the following structure:

```python
POST /chat
{
    "query": "What are the main concepts of machine learning?",
    "metadata": {
        "user_id": "user123",
        "session_id": "session456",
        "document_type": "research_paper",
        "tags": ["machine-learning", "ai"],
        "quality_threshold": 0.7
    },
    "max_results": 5,
    "include_sources": true,
    "include_metadata": true
}
```

### Pydantic Validation

All inputs are validated using Pydantic models:

- **ChatQuery**: Validates the main query structure
- **ChatMetadata**: Validates optional metadata
- **ChatResponse**: Structures the response format

### Key Features

1. **Input Validation**: Automatic validation of query length, format, and metadata
2. **Error Handling**: Comprehensive error handling with meaningful messages
3. **Async Support**: Full async/await support for better performance
4. **Type Safety**: Complete type hints throughout the codebase
5. **Documentation**: Auto-generated API documentation with FastAPI

## 📦 Modules Explained

### Models (`src/models/`)

**Purpose**: Define data structures and validation rules

- `chat.py`: Chat query and response models
- `search.py`: Search query and result models

**Key Features**:
- Pydantic validation
- JSON serialization
- Type safety
- Auto-generated documentation

### Services (`src/services/`)

**Purpose**: Business logic and orchestration

- `chat_service.py`: Handles chat query processing
- `search_service.py`: Manages search operations
- `document_service.py`: Document processing (placeholder)
- `embedding_service.py`: Embedding generation (placeholder)

**Key Features**:
- Separation of concerns
- Reusable business logic
- Error handling
- Logging integration

### API (`src/api/`)

**Purpose**: HTTP endpoints and request/response handling

- `chat_api.py`: Chat endpoints with validation
- `search_api.py`: Search endpoints
- `health_api.py`: Health check endpoints

**Key Features**:
- FastAPI integration
- Automatic validation
- Error responses
- Request logging

## 🔧 Usage

### Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI application
python main.py

# Or run directly with uvicorn
uvicorn src.main:app --host 127.0.0.1 --port 8008 --reload
```

### Testing the Modular Structure

```bash
# Run modular tests
python test_modular.py
```

### API Documentation

- **Swagger UI**: http://127.0.0.1:8008/docs
- **ReDoc**: http://127.0.0.1:8008/redoc

## 📋 API Endpoints

### Chat Endpoints

- `POST /chat/` - Process user queries
- `GET /chat/health` - Chat service health check

### Search Endpoints

- `POST /search/` - Semantic search
- `GET /search/documents/{document_id}` - Get document chunks
- `GET /search/stats` - Get search statistics

### Health Endpoints

- `GET /health/` - Basic health check
- `GET /health/detailed` - Detailed health check

## 🧪 Testing

The modular structure includes comprehensive testing:

1. **Model Validation**: Tests Pydantic model validation
2. **Service Logic**: Tests business logic in services
3. **API Endpoints**: Tests HTTP endpoints
4. **Integration**: Tests end-to-end functionality

## 🔄 Migration from Flask

The existing Flask application (`app.py`) remains unchanged and functional. The new modular structure provides:

1. **Better Organization**: Clear separation of concerns
2. **Type Safety**: Full type hints and validation
3. **Async Support**: Better performance with async/await
4. **Auto Documentation**: FastAPI generates API docs automatically
5. **Modern Standards**: Follows current Python best practices

## 🚀 Next Steps

1. **Implement Response Generation**: Add actual LLM integration to `ChatService`
2. **Complete Services**: Implement `DocumentService` and `EmbeddingService`
3. **Add Authentication**: Implement user authentication and authorization
4. **Add Caching**: Implement response caching for better performance
5. **Add Monitoring**: Add metrics and monitoring capabilities

## 📝 Example Usage

```python
import asyncio
from src.models.chat import ChatQuery, ChatMetadata
from src.services.chat_service import ChatService

async def example():
    # Create a chat query
    query = ChatQuery(
        query="What is machine learning?",
        metadata=ChatMetadata(
            user_id="user123",
            tags=["ai", "ml"]
        )
    )
    
    # Process the query
    service = ChatService()
    response = await service.process_query(query)
    
    print(f"Response: {response.response}")
    print(f"Sources: {len(response.sources)}")

# Run the example
asyncio.run(example())
```

This modular structure provides a solid foundation for building a production-ready RAG system with clean, maintainable, and scalable code. 