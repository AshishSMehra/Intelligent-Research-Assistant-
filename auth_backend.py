#!/usr/bin/env python3
"""
Enhanced Mock Backend for Intelligent Research Assistant
Includes JWT authentication, CORS, and secure token handling
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import random
import uuid
from datetime import datetime

app = Flask(__name__)

# Configure CORS for secure frontend communication
CORS(app, 
     origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=True)  # Enable credentials for JWT cookies

print("🚀 Starting Enhanced Backend Server with Authentication...")
print("📡 API will be available at: http://localhost:8008")
print("🔐 JWT Authentication enabled")
print("🌐 CORS configured for http://localhost:3000")
print("-" * 50)

# Authentication endpoints
@app.route('/auth/login', methods=['POST'])
def auth_login():
    """JWT authentication login endpoint"""
    data = request.get_json()
    username = data.get('username', '')
    password = data.get('password', '')
    
    # Mock authentication - check demo credentials
    valid_users = {
        'admin': {
            'password': 'admin123', 
            'roles': ['admin'], 
            'permissions': ['*']
        },
        'researcher': {
            'password': 'research123', 
            'roles': ['researcher'], 
            'permissions': ['documents:read', 'documents:write', 'chat:use', 'search:use', 'agents:read']
        },
        'user': {
            'password': 'user123', 
            'roles': ['user'], 
            'permissions': ['chat:use', 'search:use']
        },
        'demo': {
            'password': 'demo123', 
            'roles': ['demo'], 
            'permissions': ['chat:use', 'search:use', 'documents:read']
        }
    }
    
    if username in valid_users and valid_users[username]['password'] == password:
        # Generate mock JWT tokens
        access_token = f"mock_access_token_{username}_{int(time.time())}"
        refresh_token = f"mock_refresh_token_{username}_{int(time.time())}"
        
        return jsonify({
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": 3600,  # 1 hour
            "user": {
                "id": str(uuid.uuid4()),
                "username": username,
                "email": f"{username}@example.com",
                "roles": valid_users[username]['roles'],
                "permissions": valid_users[username]['permissions']
            }
        })
    else:
        return jsonify({"detail": "Invalid username or password"}), 401

@app.route('/auth/logout', methods=['POST'])
def auth_logout():
    """Logout endpoint"""
    return jsonify({"message": "Logged out successfully"})

@app.route('/auth/me', methods=['GET'])
def auth_me():
    """Get current user endpoint"""
    # Check for authorization header
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return jsonify({"detail": "Missing or invalid authorization header"}), 401
    
    token = auth_header.replace('Bearer ', '')
    
    # Extract username from mock token
    if 'mock_access_token_' in token:
        try:
            username = token.split('_')[3]  # Extract username from token
            
            # Return mock user data based on username
            valid_users = {
                'admin': {'roles': ['admin'], 'permissions': ['*']},
                'researcher': {'roles': ['researcher'], 'permissions': ['documents:read', 'documents:write', 'chat:use', 'search:use', 'agents:read']},
                'user': {'roles': ['user'], 'permissions': ['chat:use', 'search:use']},
                'demo': {'roles': ['demo'], 'permissions': ['chat:use', 'search:use', 'documents:read']}
            }
            
            if username in valid_users:
                return jsonify({
                    "id": str(uuid.uuid4()),
                    "username": username,
                    "email": f"{username}@example.com",
                    "roles": valid_users[username]['roles'],
                    "permissions": valid_users[username]['permissions']
                })
        except IndexError:
            pass
    
    return jsonify({"detail": "Invalid token"}), 401

@app.route('/auth/refresh', methods=['POST'])
def auth_refresh():
    """Token refresh endpoint"""
    data = request.get_json()
    refresh_token = data.get('refresh_token', '')
    
    if 'mock_refresh_token_' in refresh_token:
        try:
            username = refresh_token.split('_')[3]  # Extract username from token
            
            # Generate new tokens
            new_access_token = f"mock_access_token_{username}_{int(time.time())}"
            new_refresh_token = f"mock_refresh_token_{username}_{int(time.time())}"
            
            valid_users = {
                'admin': {'roles': ['admin'], 'permissions': ['*']},
                'researcher': {'roles': ['researcher'], 'permissions': ['documents:read', 'documents:write', 'chat:use', 'search:use', 'agents:read']},
                'user': {'roles': ['user'], 'permissions': ['chat:use', 'search:use']},
                'demo': {'roles': ['demo'], 'permissions': ['chat:use', 'search:use', 'documents:read']}
            }
            
            if username in valid_users:
                return jsonify({
                    "access_token": new_access_token,
                    "refresh_token": new_refresh_token,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "user": {
                        "id": str(uuid.uuid4()),
                        "username": username,
                        "email": f"{username}@example.com",
                        "roles": valid_users[username]['roles'],
                        "permissions": valid_users[username]['permissions']
                    }
                })
        except IndexError:
            pass
    
    return jsonify({"detail": "Invalid refresh token"}), 401

# Helper function to check authentication
def require_auth():
    """Check if request has valid authentication"""
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return None
    
    token = auth_header.replace('Bearer ', '')
    if 'mock_access_token_' in token:
        try:
            username = token.split('_')[3]
            return username
        except IndexError:
            pass
    return None

# System endpoints
@app.route('/admin/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": int(time.time()),
        "version": "1.0.0-auth",
        "uptime": 2547200,  # ~29.5 days
        "memory_usage": "1.2GB",
        "cpu_usage": "15%",
        "response_time": 145,
        "features": ["jwt_auth", "cors", "secure_tokens"]
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint with authentication"""
    username = require_auth()
    if not username:
        return jsonify({"detail": "Authentication required"}), 401
    
    data = request.get_json()
    query = data.get('query', '')
    
    # Simulate processing time
    time.sleep(random.uniform(1, 3))
    
    # Mock responses based on query content
    if 'hello' in query.lower():
        response_text = f"Hello {username}! I'm your AI research assistant. I can help you analyze documents, answer questions, and provide insights from your research materials."
    elif 'document' in query.lower():
        response_text = "I can help you process and analyze documents. Upload your PDFs, DOCs, or text files and I'll extract key information, create summaries, and answer questions about the content."
    elif 'search' in query.lower():
        response_text = "I can search through your document collection using semantic search. Just describe what you're looking for in natural language and I'll find relevant information."
    else:
        response_text = f"I understand you're asking about: '{query}'. Based on my analysis of your documents, here are some key insights and relevant information I found."
    
    return jsonify({
        "response": response_text,
        "sources": [
            {
                "title": "Research Document 1",
                "content": "This is a sample document excerpt that provides relevant context...",
                "score": 0.89,
                "metadata": {"page": 1, "source": "doc1.pdf"}
            }
        ],
        "metadata": {
            "tokens_used": len(response_text.split()) * 1.3,
            "response_time": random.uniform(2, 5),
            "model_used": "mock-llm-v1",
            "confidence_score": random.uniform(0.8, 0.95),
            "user": username
        }
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """File upload endpoint with authentication"""
    username = require_auth()
    if not username:
        return jsonify({"detail": "Authentication required"}), 401
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Simulate processing time
    time.sleep(random.uniform(2, 4))
    
    return jsonify({
        "message": "File uploaded and processed successfully",
        "document_id": str(uuid.uuid4()),
        "filename": file.filename,
        "pages_processed": random.randint(5, 50),
        "chunks_created": random.randint(20, 200),
        "processing_time": random.uniform(2, 4),
        "status": "completed",
        "uploaded_by": username
    })

@app.route('/search', methods=['POST'])
def search():
    """Search endpoint with authentication"""
    username = require_auth()
    if not username:
        return jsonify({"detail": "Authentication required"}), 401
    
    data = request.get_json()
    query = data.get('query', '')
    max_results = data.get('max_results', 10)
    
    # Simulate processing time
    time.sleep(random.uniform(0.5, 1.5))
    
    # Generate mock results
    results = []
    for i in range(min(max_results, random.randint(3, 8))):
        results.append({
            "title": f"Document {i+1}: {query} Analysis",
            "content": f"This document contains relevant information about {query}. Here's an excerpt that shows the key findings and methodology used in the research...",
            "score": random.uniform(0.6, 0.95),
            "metadata": {
                "page": random.randint(1, 20),
                "source": f"document_{i+1}.pdf",
                "document_id": str(uuid.uuid4())
            }
        })
    
    return jsonify({
        "results": results,
        "total_found": len(results),
        "query": query,
        "processing_time": random.uniform(0.5, 1.5),
        "searched_by": username
    })

@app.route('/agents', methods=['GET'])
def get_agents():
    """Agents status endpoint with authentication"""
    username = require_auth()
    if not username:
        return jsonify({"detail": "Authentication required"}), 401
    
    return jsonify({
        "agents": [
            {
                "type": "planner",
                "status": "active",
                "tasks_completed": 145,
                "average_time": 1.2,
                "last_activity": datetime.now().isoformat()
            },
            {
                "type": "research", 
                "status": "active",
                "tasks_completed": 132,
                "average_time": 3.4,
                "last_activity": datetime.now().isoformat()
            },
            {
                "type": "reasoner",
                "status": "active", 
                "tasks_completed": 156,
                "average_time": 2.1,
                "last_activity": datetime.now().isoformat()
            },
            {
                "type": "executor",
                "status": "active",
                "tasks_completed": 98,
                "average_time": 0.8,
                "last_activity": datetime.now().isoformat()
            }
        ],
        "total_workflows": 2847,
        "successful_workflows": 2734,
        "failed_workflows": 113,
        "requested_by": username
    })

@app.route('/documents', methods=['GET'])
def get_documents():
    """Documents list endpoint with authentication"""
    username = require_auth()
    if not username:
        return jsonify({"detail": "Authentication required"}), 401
    
    # Mock documents list
    documents = []
    for i in range(5):
        documents.append({
            "id": str(uuid.uuid4()),
            "filename": f"document_{i+1}.pdf",
            "title": f"Research Document {i+1}",
            "size": random.randint(100000, 5000000),
            "pages": random.randint(5, 50),
            "uploaded_at": datetime.now().isoformat(),
            "status": "processed"
        })
    
    return jsonify({
        "documents": documents,
        "total": len(documents),
        "requested_by": username
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8008, debug=False, use_reloader=False) 