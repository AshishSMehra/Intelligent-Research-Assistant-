# Intelligent Research Assistant

> A production-ready RAG (Retrieval-Augmented Generation) system that transforms PDFs into searchable knowledge bases using advanced NLP and vector search.

**For**: Researchers, developers, and organizations needing intelligent document processing and semantic search capabilities.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-orange.svg)](https://qdrant.tech)

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [API Usage](#-api-usage)
- [Project Structure](#-project-structure)
- [Data Schema](#-data-schema)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Security](#-security)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **📄 Process PDFs** - Extract text with page structure preservation and edge case handling
- **🧩 Smart Chunking** - Intelligent text segmentation with configurable overlap and boundary detection
- **🧠 Generate Embeddings** - GPU-accelerated vector generation using SentenceTransformers
- **💾 Store Vectors** - High-performance vector database with rich metadata (21+ fields)
- **🔍 Semantic Search** - Real-time similarity search with configurable thresholds
- **📊 Monitor Quality** - Automatic PDF analysis, OCR detection, and quality scoring
- **🌐 REST API** - Complete API with Swagger documentation and 6 endpoints
- **⚡ Production Ready** - Comprehensive logging, error handling, and performance optimization

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   PDF       │    │   Text       │    │   Smart     │    │   Vector    │    │   Qdrant    │
│   Upload    │───▶│  Extraction  │───▶│  Chunking   │───▶│ Embeddings  │───▶│   Storage   │
│  (Flask)    │    │ (PyMuPDF)    │    │(Boundary)   │    │(SentenceT)  │    │ (Vector DB) │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Tech Stack:**
- **Backend**: Flask, PyMuPDF, SentenceTransformers
- **Vector DB**: Qdrant (Docker)
- **ML**: all-MiniLM-L6-v2 (384D embeddings)
- **Hardware**: MPS/GPU acceleration support
- **Monitoring**: Loguru logging, real-time stats

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

## ⚙️ Configuration

Create `.env` file:
```env
FLASK_ENV=development
FLASK_DEBUG=True
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216
```

**Key Settings** (in `src/pipeline/pipeline.py`):
```python
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 100       # Overlap between chunks
VECTOR_SIZE = 384         # Embedding dimensions
COLLECTION_NAME = "research_documents"
```

## 🔌 API Usage

### Upload PDF Document
```bash
# File upload
curl -X POST -F "file=@document.pdf" http://127.0.0.1:8008/upload

# URL upload
curl -X POST -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/document.pdf"}' \
  http://127.0.0.1:8008/upload
```

**Response:**
```json
{
  "message": "File 'document.pdf' processed successfully with page structure preservation.",
  "document_id": "uuid-here"
}
```

### Semantic Search
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "machine learning algorithms", "limit": 5}' \
  http://127.0.0.1:8008/search
```

**Response:**
```json
{
  "status": "success",
  "query": "machine learning algorithms",
  "results_count": 3,
  "results": [
    {
      "id": "point-id",
      "score": 0.847,
      "text": "Machine learning is a subset of artificial intelligence...",
      "document_id": "doc-uuid",
      "source_pages": [1]
    }
  ]
}
```

### Get Document Chunks
```bash
curl "http://127.0.0.1:8008/documents/doc-uuid?include_text=true"
```

### Collection Statistics
```bash
curl http://127.0.0.1:8008/collection-stats
```

## 📁 Project Structure

```
Intelligent-Research-Assistant-/
├── app.py                    # Flask application & API endpoints
├── requirements.txt          # Python dependencies
├── src/
│   └── pipeline/
│       └── pipeline.py       # Core RAG pipeline (Steps 1-5)
├── uploads/                  # Document storage
├── logs/                     # Application logs
├── qdrant_storage/          # Vector database files
└── .env.example             # Environment template
```

**Key Files:**
- `app.py` - API endpoints, file handling, error management
- `src/pipeline/pipeline.py` - Complete RAG pipeline implementation
- `logging_config.py` - Structured logging configuration

## 🗄️ Data Schema

### Vector Database (Qdrant)
**Collection**: `research_documents`
**Vector Size**: 384 dimensions (all-MiniLM-L6-v2)

**Point Payload Schema:**
```json
{
  "document_id": "uuid",
  "chunk_id": 0,
  "text": "chunk content",
  "char_count": 500,
  "source_pages": [1, 2],
  "has_scanned_content": false,
  "quality_issues": [],
  "created_at": 1692123456,
  "embedding_model": "all-MiniLM-L6-v2",
  "tags": ["research", "ml"],
  "custom_metadata": {}
}
```

### Processing Pipeline Data Flow
1. **PDF Input** → Page extraction with metadata
2. **Text Chunks** → Boundary-aware segmentation
3. **Embeddings** → 384D vectors with normalization
4. **Vector Storage** → Qdrant with 21+ metadata fields

## 🚀 Deployment

### Docker Compose
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8008:8008"
    environment:
      - FLASK_ENV=production
    depends_on:
      - qdrant
  
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

### Cloud Deployment
1. **AWS/GCP**: Use managed container services
2. **Qdrant Cloud**: Replace local Qdrant with cloud instance
3. **Environment**: Set production environment variables
4. **Scaling**: Horizontal scaling with load balancer

## 🔧 Troubleshooting

### Common Issues

**Qdrant Connection Failed**
```bash
# Check if Docker is running
docker ps | grep qdrant

# Restart Qdrant container
docker restart qdrant-container
```

**PDF Processing Errors**
```bash
# Check file permissions
ls -la uploads/

# Verify PDF integrity
file document.pdf
```

**Memory Issues**
```bash
# Increase Docker memory limit
docker run -m 4g -p 6333:6333 qdrant/qdrant

# Monitor memory usage
docker stats
```

**Embedding Model Loading**
```bash
# Clear model cache
rm -rf ~/.cache/torch/sentence_transformers/

# Check GPU availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Performance Tuning
- **Batch Size**: Adjust `batch_size` in embedding generation
- **Chunk Size**: Optimize `chunk_size` for your documents
- **Memory**: Increase Docker memory for large documents
- **GPU**: Enable MPS/CUDA for faster embeddings

## 🔒 Security

### Data Privacy
- **Local Processing**: All data processed locally, no external API calls
- **File Storage**: Temporary file storage with automatic cleanup
- **Vector DB**: Local Qdrant instance, no data sent to cloud services

### Security Measures
- **Input Validation**: Comprehensive file type and size validation
- **Error Handling**: No sensitive data in error messages
- **Access Control**: No authentication (add for production use)
- **File Sanitization**: Secure file handling and cleanup

### Production Security Checklist
- [ ] Add authentication/authorization
- [ ] Enable HTTPS
- [ ] Implement rate limiting
- [ ] Add input sanitization
- [ ] Configure CORS policies
- [ ] Set up monitoring and alerting

## 🗺️ Roadmap

### Near-term (Q1 2024)
- [ ] **OCR Integration** - Automatic text extraction from scanned pages
- [ ] **Multi-format Support** - DOCX, TXT, and other document types
- [ ] **Batch Processing** - Efficient handling of multiple documents
- [ ] **Advanced Search** - Filtering by document type, date, quality
- [ ] **RAG Implementation** - Retrieval-augmented generation endpoints

### Medium-term (Q2 2024)
- [ ] **Cloud Storage** - S3/GCS integration for document storage
- [ ] **Distributed Processing** - Multi-worker document processing
- [ ] **Real-time Updates** - Live document processing status
- [ ] **Advanced Analytics** - Processing insights and performance metrics

### Long-term (Q3 2024+)
- [ ] **Multi-language Support** - Internationalization and localization
- [ ] **Advanced ML Models** - Custom embedding models and fine-tuning
- [ ] **Enterprise Features** - SSO, audit logs, compliance features

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup
```bash
git clone <repository-url>
cd Intelligent-Research-Assistant-
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### Code Style
- **Python**: Follow PEP 8 with Black formatting
- **Tests**: Write unit tests for new features
- **Documentation**: Update README and docstrings
- **Commits**: Use conventional commit messages

### Pull Request Process
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Run tests (`python -m pytest`)
5. Submit pull request with description

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Acknowledgments
- **SentenceTransformers** - For high-quality embeddings
- **Qdrant** - For the excellent vector database
- **PyMuPDF** - For robust PDF processing
- **Flask** - For the web framework

## 👥 Maintainers

**Ashish Mehra** - [GitHub](https://github.com/ashishsmehra)

### Contact
- **Issues**: [GitHub Issues](https://github.com/ashishsmehra/Intelligent-Research-Assistant-/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ashishsmehra/Intelligent-Research-Assistant-/discussions)
- **Email**: [Your Email]

---

**Built with ❤️ for intelligent document processing and search**

*Star this repository if it helped you! ⭐*