#!/bin/bash

echo "🚀 Starting Intelligent Research Assistant Frontend"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "frontend/package.json" ]; then
    echo "❌ Error: frontend/package.json not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Navigate to frontend directory
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed successfully"
    echo ""
fi

# Check if backend is running
echo "🔍 Checking backend status..."
if curl -s http://localhost:8008/admin/health > /dev/null; then
    echo "✅ Backend is running on port 8008"
else
    echo "⚠️  Backend not detected on port 8008"
    echo "💡 Make sure to start your backend server first:"
    echo "   python3 auth_backend.py"
    echo ""
fi

# Start the frontend
echo "🌟 Starting React development server..."
echo "📱 Frontend will be available at: http://localhost:3000"
echo "🔐 JWT Authentication enabled"
echo "🌐 CORS configured for backend communication"
echo "🛑 Press Ctrl+C to stop"
echo ""

# Start Vite dev server
npm run dev 