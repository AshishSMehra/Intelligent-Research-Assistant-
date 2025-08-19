# 🐳 Docker Deployment Guide

Complete guide for deploying the Intelligent Research Assistant using Docker and Docker Compose.

## 📋 **Table of Contents**

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Development Environment](#development-environment)
- [Production Deployment](#production-deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## 🔧 **Prerequisites**

### **Required Software**
- **Docker**: Version 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 2.0+ ([Install Docker Compose](https://docs.docker.com/compose/install/))
- **Git**: For cloning the repository

### **System Requirements**
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: Minimum 10GB free space
- **CPU**: 2+ cores recommended
- **Network**: Internet connection for pulling images

---

## 🚀 **Quick Start**

### **1. Clone Repository**
```bash
git clone <repository-url>
cd Intelligent-Research-Assistant-
```

### **2. Run Setup Script**
```bash
# Make script executable
chmod +x scripts/docker-setup.sh

# Run setup
./scripts/docker-setup.sh setup
```

### **3. Access Services**
- **API**: http://localhost:8008
- **Qdrant Web UI**: http://localhost:8080
- **Redis Commander**: http://localhost:8081

---

## 🛠️ **Development Environment**

### **Development Setup**

The development environment includes additional tools for easier development:

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d --build

# View logs
docker-compose -f docker-compose.dev.yml logs -f api

# Stop services
docker-compose -f docker-compose.dev.yml down
```

### **Development Services**

| Service | Port | Description |
|---------|------|-------------|
| **API** | 8008 | Flask application with hot-reload |
| **Worker** | - | Background task processing |
| **Qdrant** | 6333 | Vector database |
| **Qdrant Web UI** | 8080 | Web interface for Qdrant |
| **Redis** | 6379 | Cache and session storage |
| **Redis Commander** | 8081 | Web interface for Redis |

### **Development Features**

- **Hot Reload**: Code changes automatically restart the API
- **Volume Mounting**: Local code is mounted into containers
- **Development Tools**: Web UIs for databases
- **Debug Mode**: Enhanced logging and error reporting

---

## 🏭 **Production Deployment**

### **Production Setup**

```bash
# Start production environment
docker-compose up -d --build

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### **Production Services**

| Service | Port | Description |
|---------|------|-------------|
| **API** | 8008 | Production Flask application |
| **Worker** | - | Background task processing |
| **Qdrant** | 6333 | Vector database |
| **Redis** | 6379 | Cache and session storage |
| **Nginx** | 80/443 | Reverse proxy (optional) |

### **Production Features**

- **Optimized Images**: Smaller, security-hardened containers
- **Health Checks**: Automatic health monitoring
- **Logging**: Structured logging with rotation
- **Security**: Non-root users, minimal attack surface

---

## ⚙️ **Configuration**

### **Environment Variables**

Create a `.env` file in the project root:

```bash
# Database Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Redis Configuration
REDIS_PASSWORD=your-secure-password

# Security Configuration
JWT_SECRET=your-super-secret-jwt-key
AWS_KMS_KEY_ID=your-aws-kms-key-id
VAULT_URL=your-vault-url
VAULT_TOKEN=your-vault-token

# LLM Configuration
OPENAI_API_KEY=your-openai-api-key
OLLAMA_BASE_URL=http://localhost:11434

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=microsoft/DialoGPT-small

# Fine-tuning Configuration
WANDB_API_KEY=your-wandb-api-key
MLFLOW_TRACKING_URI=your-mlflow-uri

# Development Configuration
FLASK_ENV=production
FLASK_DEBUG=0
```

### **Docker Compose Configuration**

#### **Development Configuration** (`docker-compose.dev.yml`)

```yaml
version: '3.8'

services:
  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    container_name: intelligent-research-qdrant-dev
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data_dev:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    networks:
      - intelligent-research-network-dev
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Redis for caching and session management
  redis:
    image: redis:7-alpine
    container_name: intelligent-research-redis-dev
    ports:
      - "6379:6379"
    volumes:
      - redis_data_dev:/data
    command: redis-server --appendonly yes
    networks:
      - intelligent-research-network-dev
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  # API Service (Development)
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: intelligent-research-api-dev
    ports:
      - "8008:8008"
    volumes:
      - .:/app
      - ./uploads:/app/uploads
      - ./logs:/app/logs
      - ./qdrant_storage:/app/qdrant_storage
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - REDIS_URL=redis://redis:6379
      - FLASK_ENV=development
      - FLASK_DEBUG=1
      - JWT_SECRET=dev-secret-key-change-in-production
    depends_on:
      qdrant:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - intelligent-research-network-dev
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8008/admin/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Worker Service (Development)
  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: intelligent-research-worker-dev
    volumes:
      - .:/app
      - ./uploads:/app/uploads
      - ./logs:/app/logs
      - ./qdrant_storage:/app/qdrant_storage
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - REDIS_URL=redis://redis:6379
      - WORKER_ENV=development
      - JWT_SECRET=dev-secret-key-change-in-production
    depends_on:
      qdrant:
        condition: service_healthy
      redis:
        condition: service_healthy
      api:
        condition: service_healthy
    networks:
      - intelligent-research-network-dev
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "ps", "aux", "|", "grep", "python", "|", "grep", "worker.py"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Development Tools
  # Qdrant Web UI
  qdrant-web:
    image: qdrant/qdrant-web:latest
    container_name: intelligent-research-qdrant-web-dev
    ports:
      - "8080:80"
    environment:
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - qdrant
    networks:
      - intelligent-research-network-dev
    restart: unless-stopped

  # Redis Commander (Web UI for Redis)
  redis-commander:
    image: rediscommander/redis-commander:latest
    container_name: intelligent-research-redis-commander-dev
    ports:
      - "8081:8081"
    environment:
      - REDIS_HOSTS=local:redis:6379
    depends_on:
      - redis
    networks:
      - intelligent-research-network-dev
    restart: unless-stopped

volumes:
  qdrant_data_dev:
    driver: local
  redis_data_dev:
    driver: local

networks:
  intelligent-research-network-dev:
    driver: bridge
```

#### **Production Configuration** (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    container_name: intelligent-research-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    networks:
      - intelligent-research-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Redis for caching and session management
  redis:
    image: redis:7-alpine
    container_name: intelligent-research-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD:-redis123}
    networks:
      - intelligent-research-network
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  # API Service
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: intelligent-research-api
    ports:
      - "8008:8008"
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
      - ./qdrant_storage:/app/qdrant_storage
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - REDIS_URL=redis://:${REDIS_PASSWORD:-redis123}@redis:6379
      - FLASK_ENV=production
      - JWT_SECRET=${JWT_SECRET:-your-secret-key-change-in-production}
      - AWS_KMS_KEY_ID=${AWS_KMS_KEY_ID}
      - VAULT_URL=${VAULT_URL}
      - VAULT_TOKEN=${VAULT_TOKEN}
    depends_on:
      qdrant:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - intelligent-research-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8008/admin/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Worker Service
  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: intelligent-research-worker
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
      - ./qdrant_storage:/app/qdrant_storage
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - REDIS_URL=redis://:${REDIS_PASSWORD:-redis123}@redis:6379
      - WORKER_ENV=production
      - JWT_SECRET=${JWT_SECRET:-your-secret-key-change-in-production}
      - AWS_KMS_KEY_ID=${AWS_KMS_KEY_ID}
      - VAULT_URL=${VAULT_URL}
      - VAULT_TOKEN=${VAULT_TOKEN}
    depends_on:
      qdrant:
        condition: service_healthy
      redis:
        condition: service_healthy
      api:
        condition: service_healthy
    networks:
      - intelligent-research-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "ps", "aux", "|", "grep", "python", "|", "grep", "worker.py"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Nginx Reverse Proxy (Optional)
  nginx:
    image: nginx:alpine
    container_name: intelligent-research-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    networks:
      - intelligent-research-network
    restart: unless-stopped
    profiles:
      - production

volumes:
  qdrant_data:
    driver: local
  redis_data:
    driver: local

networks:
  intelligent-research-network:
    driver: bridge
```

### **Dockerfile Configuration**

#### **API Dockerfile** (`Dockerfile`)

```dockerfile
# Dockerfile for Intelligent Research Assistant API
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create necessary directories
RUN mkdir -p uploads logs qdrant_storage

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8008

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8008/admin/health || exit 1

# Run the application
CMD ["python", "app.py"]
```

#### **Worker Dockerfile** (`Dockerfile.worker`)

```dockerfile
# Dockerfile for Intelligent Research Assistant Worker
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WORKER_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create necessary directories
RUN mkdir -p uploads logs qdrant_storage

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash worker && \
    chown -R worker:worker /app
USER worker

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD ps aux | grep python | grep worker.py || exit 1

# Run the worker
CMD ["python", "worker.py"]
```

---

## 📊 **Monitoring**

### **Health Checks**

All services include health checks:

```bash
# Check all services
docker-compose ps

# Check specific service
docker-compose exec api curl -f http://localhost:8008/admin/health

# View health check logs
docker-compose logs api | grep health
```

### **Logging**

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api
docker-compose logs -f worker
docker-compose logs -f qdrant

# View logs with timestamps
docker-compose logs -f --timestamps api
```

### **Resource Monitoring**

```bash
# Check resource usage
docker stats

# Check disk usage
docker system df

# Check volume usage
docker volume ls
docker volume inspect intelligent-research-qdrant_data
```

### **Performance Monitoring**

```bash
# Check API response time
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8008/admin/health

# Monitor Qdrant performance
curl http://localhost:6333/collections/documents

# Check Redis performance
docker-compose exec redis redis-cli info memory
```

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **1. Port Already in Use**
```bash
# Check what's using the port
sudo lsof -i :8008

# Kill the process
sudo kill -9 <PID>

# Or use different ports
docker-compose -f docker-compose.dev.yml up -d --build
```

#### **2. Permission Denied**
```bash
# Fix file permissions
sudo chown -R $USER:$USER uploads logs qdrant_storage

# Or run with sudo (not recommended for production)
sudo docker-compose up -d
```

#### **3. Out of Memory**
```bash
# Check memory usage
docker stats

# Increase Docker memory limit
# In Docker Desktop: Settings > Resources > Memory

# Or reduce service memory usage
docker-compose down
docker system prune -f
docker-compose up -d
```

#### **4. Service Won't Start**
```bash
# Check service logs
docker-compose logs api

# Check service status
docker-compose ps

# Restart specific service
docker-compose restart api

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

#### **5. Database Connection Issues**
```bash
# Check if Qdrant is running
docker-compose ps qdrant

# Check Qdrant logs
docker-compose logs qdrant

# Test Qdrant connection
curl http://localhost:6333/health

# Restart Qdrant
docker-compose restart qdrant
```

### **Debug Commands**

```bash
# Enter container shell
docker-compose exec api bash
docker-compose exec worker bash

# Check environment variables
docker-compose exec api env

# Check file permissions
docker-compose exec api ls -la

# Check network connectivity
docker-compose exec api ping qdrant
docker-compose exec api ping redis

# Check service dependencies
docker-compose exec api curl -f http://qdrant:6333/health
docker-compose exec api redis-cli -h redis ping
```

### **Cleanup Commands**

```bash
# Stop all services
docker-compose down

# Remove containers and networks
docker-compose down --remove-orphans

# Remove volumes (WARNING: This will delete all data)
docker-compose down -v

# Clean up Docker system
docker system prune -f
docker volume prune -f
docker network prune -f

# Remove all images
docker rmi $(docker images -q)
```

---

## 🚀 **Advanced Configuration**

### **Custom Nginx Configuration**

Create a custom `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log;

    # Basic settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 100M;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=2r/s;

    # Upstream API server
    upstream api_backend {
        server api:8008;
        keepalive 32;
    }

    # HTTP server (redirect to HTTPS in production)
    server {
        listen 80;
        server_name _;

        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }

        # Redirect all other traffic to HTTPS (in production)
        location / {
            return 301 https://$host$request_uri;
        }
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name _;

        # SSL configuration (for production, use proper certificates)
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;

            proxy_pass http://api_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;

            # Buffer settings
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
        }

        # Upload endpoint (higher rate limit)
        location /upload {
            limit_req zone=upload burst=5 nodelay;
            client_max_body_size 100M;

            proxy_pass http://api_backend/upload;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Longer timeouts for file uploads
            proxy_connect_timeout 300s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }

        # Chat endpoint (streaming support)
        location /chat {
            limit_req zone=api burst=20 nodelay;

            proxy_pass http://api_backend/chat;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Streaming support
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_cache off;

            # Timeouts for streaming
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }

        # Admin endpoints
        location /admin/ {
            limit_req zone=api burst=10 nodelay;

            proxy_pass http://api_backend/admin/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }

        # Default location
        location / {
            proxy_pass http://api_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### **SSL Certificate Setup**

For production, set up proper SSL certificates:

```bash
# Create SSL directory
mkdir -p ssl

# Generate self-signed certificate (for development)
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# For production, use Let's Encrypt or your CA
# Copy your certificates to ssl/cert.pem and ssl/key.pem
```

### **Environment-Specific Configurations**

#### **Development Environment**
```bash
# Development environment variables
FLASK_ENV=development
FLASK_DEBUG=1
JWT_SECRET=dev-secret-key-change-in-production
REDIS_PASSWORD=
```

#### **Staging Environment**
```bash
# Staging environment variables
FLASK_ENV=staging
FLASK_DEBUG=0
JWT_SECRET=staging-secret-key
REDIS_PASSWORD=staging-redis-password
```

#### **Production Environment**
```bash
# Production environment variables
FLASK_ENV=production
FLASK_DEBUG=0
JWT_SECRET=your-super-secure-production-secret
REDIS_PASSWORD=your-super-secure-redis-password
AWS_KMS_KEY_ID=your-aws-kms-key-id
VAULT_URL=your-vault-url
VAULT_TOKEN=your-vault-token
```

---

## 📈 **Scaling**

### **Horizontal Scaling**

```bash
# Scale API service
docker-compose up -d --scale api=3

# Scale worker service
docker-compose up -d --scale worker=2

# Check scaled services
docker-compose ps
```

### **Load Balancing**

Update `docker-compose.yml` to include a load balancer:

```yaml
# Add to docker-compose.yml
services:
  # ... existing services ...

  # Load Balancer
  nginx-lb:
    image: nginx:alpine
    container_name: intelligent-research-nginx-lb
    ports:
      - "80:80"
    volumes:
      - ./nginx-lb.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api
    networks:
      - intelligent-research-network
    restart: unless-stopped
```

### **Resource Limits**

Add resource limits to services:

```yaml
services:
  api:
    # ... existing configuration ...
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M

  worker:
    # ... existing configuration ...
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
```

---

*This Docker deployment guide provides comprehensive instructions for deploying the Intelligent Research Assistant in both development and production environments.* 