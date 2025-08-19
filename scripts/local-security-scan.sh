#!/bin/bash

# Local Security Scan Script
# This script runs security scans locally without the disk space issues of CI/CD

set -e

echo "🔍 Starting local security scan..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Clean up Docker resources
echo "🧹 Cleaning up Docker resources..."
docker system prune -f || true

# Build the image for scanning
echo "🏗️ Building Docker image for scanning..."
docker build -t intelligent-research-api:local-scan .

# Check available disk space
echo "💾 Available disk space:"
df -h

# Run Trivy scan with limited severity and timeout
echo "🔍 Running Trivy vulnerability scan..."
if command -v trivy &> /dev/null; then
    trivy image --format sarif --output trivy-results.sarif --severity CRITICAL,HIGH --timeout 5m intelligent-research-api:local-scan
    echo "✅ Trivy scan completed successfully!"
    echo "📄 Results saved to: trivy-results.sarif"
else
    echo "⚠️ Trivy not installed. Install with: brew install trivy (macOS) or see https://aquasecurity.github.io/trivy/latest/getting-started/installation/"
fi

# Run Safety check for Python dependencies
echo "🐍 Running Safety check for Python dependencies..."
if command -v safety &> /dev/null; then
    safety check --json --output safety-report.json || true
    echo "✅ Safety check completed!"
else
    echo "⚠️ Safety not installed. Install with: pip install safety"
fi

# Run Bandit for Python code security
echo "🔒 Running Bandit security scan..."
if command -v bandit &> /dev/null; then
    bandit -r src/ -f json -o bandit-report.json || true
    echo "✅ Bandit scan completed!"
else
    echo "⚠️ Bandit not installed. Install with: pip install bandit"
fi

# Clean up
echo "🧹 Cleaning up..."
docker rmi intelligent-research-api:local-scan || true
docker system prune -f || true

echo "✅ Local security scan completed!"
echo "📊 Check the generated reports for details:"
echo "   - trivy-results.sarif (if Trivy is installed)"
echo "   - safety-report.json (if Safety is installed)"
echo "   - bandit-report.json (if Bandit is installed)" 