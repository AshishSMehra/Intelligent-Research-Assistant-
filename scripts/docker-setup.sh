#!/bin/bash

# Intelligent Research Assistant - Docker Setup Script
# This script helps set up the Docker environment for the Intelligent Research Assistant

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Docker installation
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        print_status "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        print_status "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p uploads logs qdrant_storage ssl
    
    print_success "Directories created"
}

# Function to generate SSL certificates for development
generate_ssl_certs() {
    print_status "Generating SSL certificates for development..."
    
    if [ ! -f ssl/cert.pem ] || [ ! -f ssl/key.pem ]; then
        openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        print_success "SSL certificates generated"
    else
        print_status "SSL certificates already exist"
    fi
}

# Function to create environment file
create_env_file() {
    print_status "Creating environment file..."
    
    if [ ! -f .env ]; then
        cat > .env << EOF
# Intelligent Research Assistant Environment Variables

# Database Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Redis Configuration
REDIS_PASSWORD=redis123

# Security Configuration
JWT_SECRET=your-secret-key-change-in-production
AWS_KMS_KEY_ID=
VAULT_URL=
VAULT_TOKEN=

# LLM Configuration
OPENAI_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=microsoft/DialoGPT-small

# Fine-tuning Configuration
WANDB_API_KEY=
MLFLOW_TRACKING_URI=

# Development Configuration
FLASK_ENV=development
FLASK_DEBUG=1
EOF
        print_success "Environment file created (.env)"
        print_warning "Please update the .env file with your actual configuration values"
    else
        print_status "Environment file already exists"
    fi
}

# Function to build and start services
start_services() {
    local environment=${1:-dev}
    
    print_status "Starting services in $environment environment..."
    
    if [ "$environment" = "prod" ]; then
        docker-compose up -d --build
        print_success "Production services started"
        print_status "API available at: https://localhost"
        print_status "Qdrant available at: http://localhost:6333"
        print_status "Redis available at: localhost:6379"
    else
        docker-compose -f docker-compose.dev.yml up -d --build
        print_success "Development services started"
        print_status "API available at: http://localhost:8008"
        print_status "Qdrant available at: http://localhost:6333"
        print_status "Qdrant Web UI available at: http://localhost:8080"
        print_status "Redis available at: localhost:6379"
        print_status "Redis Commander available at: http://localhost:8081"
    fi
}

# Function to check service health
check_health() {
    print_status "Checking service health..."
    
    # Wait for services to start
    sleep 30
    
    # Check API health
    if curl -f http://localhost:8008/admin/health >/dev/null 2>&1; then
        print_success "API is healthy"
    else
        print_warning "API health check failed"
    fi
    
    # Check Qdrant health
    if curl -f http://localhost:6333/health >/dev/null 2>&1; then
        print_success "Qdrant is healthy"
    else
        print_warning "Qdrant health check failed"
    fi
    
    # Check Redis health
    if docker exec intelligent-research-redis-dev redis-cli ping >/dev/null 2>&1; then
        print_success "Redis is healthy"
    else
        print_warning "Redis health check failed"
    fi
}

# Function to show logs
show_logs() {
    local service=${1:-api}
    
    print_status "Showing logs for $service service..."
    docker-compose -f docker-compose.dev.yml logs -f $service
}

# Function to stop services
stop_services() {
    local environment=${1:-dev}
    
    print_status "Stopping services in $environment environment..."
    
    if [ "$environment" = "prod" ]; then
        docker-compose down
    else
        docker-compose -f docker-compose.dev.yml down
    fi
    
    print_success "Services stopped"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    
    # Stop all services
    docker-compose down 2>/dev/null || true
    docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
    
    # Remove containers
    docker container prune -f
    
    # Remove images
    docker image prune -f
    
    # Remove volumes
    docker volume prune -f
    
    print_success "Cleanup completed"
}

# Function to show help
show_help() {
    echo "Intelligent Research Assistant - Docker Setup Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     - Set up the Docker environment (default)"
    echo "  start     - Start services in development mode"
    echo "  start-prod - Start services in production mode"
    echo "  stop      - Stop services"
    echo "  logs      - Show logs for a service (default: api)"
    echo "  health    - Check service health"
    echo "  cleanup   - Clean up Docker resources"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 start"
    echo "  $0 logs worker"
    echo "  $0 cleanup"
}

# Main script logic
main() {
    local command=${1:-setup}
    
    case $command in
        setup)
            print_status "Setting up Intelligent Research Assistant Docker environment..."
            check_docker
            create_directories
            generate_ssl_certs
            create_env_file
            start_services dev
            check_health
            print_success "Setup completed successfully!"
            print_status "You can now access the services at:"
            print_status "  - API: http://localhost:8008"
            print_status "  - Qdrant Web UI: http://localhost:8080"
            print_status "  - Redis Commander: http://localhost:8081"
            ;;
        start)
            start_services dev
            ;;
        start-prod)
            start_services prod
            ;;
        stop)
            stop_services dev
            ;;
        stop-prod)
            stop_services prod
            ;;
        logs)
            show_logs $2
            ;;
        health)
            check_health
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 