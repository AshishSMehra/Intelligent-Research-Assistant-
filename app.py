import asyncio
import hashlib
import os
import random
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests
from botocore.exceptions import NoCredentialsError
from datasets import Dataset, DatasetDict
from flasgger import Swagger
from flask import Flask, jsonify, make_response, request
from flask_cors import CORS
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, pipeline

from logging_config import logger

# Import multi-agent system
from src.agents.agent_orchestrator import AgentOrchestrator

# Import pipeline modules
from src.pipeline.pipeline import (
    chunk_text,
    create_collection_if_not_exists,
    delete_document,
    extract_text_from_pdf,
    generate_embeddings,
    get_collection_info,
    get_collection_stats,
    get_model_info,
    process_pdf_with_page_structure,
    search_by_document,
    search_similar_chunks,
    store_embeddings,
)

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# App & routes
# -----------------------------------------------------------------------------
app = Flask(__name__)
swagger = Swagger(app)

# Configure CORS for secure frontend communication
CORS(
    app,
    origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    supports_credentials=True,
)

# Add explicit OPTIONS handling for all routes
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response

# Add a global OPTIONS handler for all routes
@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def handle_options(path):
    response = jsonify({})
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response

# Initialize the vector database at startup (compatible across Flask versions)
logger.info("Initializing application and vector database...")
create_collection_if_not_exists()

# Initialize multi-agent orchestrator
logger.info("Initializing multi-agent orchestrator...")
agent_orchestrator = AgentOrchestrator()


@app.route("/")
def read_root():
    """
    Welcome Endpoint
    This endpoint returns a welcome message.
    ---
    responses:
      200:
        description: A welcome message.
    """
    return jsonify(
        {
            "message": (
                "Welcome to the File Upload API. "
                "Use the /upload endpoint to upload files."
            )
        }
    )


@app.route("/health")
def health():
    """
    Health Check Endpoint
    This endpoint returns the health status of the application.
    ---
    responses:
      200:
        description: The application status.
    """
    return jsonify({"status": "ok"})


# =============================================================================
# Authentication Endpoints
# =============================================================================

@app.route("/auth/login", methods=["POST", "OPTIONS"])
def auth_login():
    """
    Authentication Login Endpoint
    Handles user login with mock authentication.
    ---
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            username:
              type: string
              description: Username
            password:
              type: string
              description: Password
    responses:
      200:
        description: Login successful
      401:
        description: Invalid credentials
    """
    if request.method == "OPTIONS":
        return jsonify({})
    
    data = request.get_json()
    username = data.get("username", "")
    password = data.get("password", "")
    
    # Mock authentication - you can replace this with real authentication
    valid_credentials = {
        "admin": "admin123",
        "researcher": "research123", 
        "user": "user123",
        "demo": "demo123"
    }
    
    if username in valid_credentials and valid_credentials[username] == password:
        # Generate mock tokens
        access_token = f"mock_access_token_{username}_{int(time.time())}"
        refresh_token = f"mock_refresh_token_{username}_{int(time.time())}"
        
        return jsonify({
            "access_token": access_token,
            "expires_in": 3600,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "user": {
                "id": str(uuid.uuid4()),
                "username": username,
                "email": f"{username}@example.com",
                "roles": ["admin"] if username == "admin" else ["user"],
                "permissions": ["*"] if username == "admin" else ["read", "write"]
            }
        }), 200
    else:
        return jsonify({"error": "Invalid username or password"}), 401


@app.route("/auth/me", methods=["GET"])
def auth_me():
    """
    Get Current User Endpoint
    Returns information about the currently authenticated user.
    ---
    responses:
      200:
        description: Current user information
    """
    # Mock user data - in a real app, this would come from the JWT token
    return jsonify({
        "id": str(uuid.uuid4()),
        "username": "admin",
        "email": "admin@example.com",
        "roles": ["admin"],
        "permissions": ["*"]
    })


@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    """
    Logout Endpoint
    Handles user logout.
    ---
    responses:
      200:
        description: Logout successful
    """
    return jsonify({"message": "Logged out successfully"}), 200


@app.route("/auth/refresh", methods=["POST"])
def auth_refresh():
    """
    Refresh Token Endpoint
    Refreshes the access token using a refresh token.
    ---
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            refresh_token:
              type: string
              description: Refresh token
    responses:
      200:
        description: Token refreshed successfully
      401:
        description: Invalid refresh token
    """
    data = request.get_json()
    refresh_token = data.get("refresh_token", "")
    
    # Mock token refresh - in a real app, this would validate the refresh token
    if refresh_token.startswith("mock_refresh_token_"):
        return jsonify({
            "access_token": f"mock_access_token_refreshed_{int(time.time())}",
            "expires_in": 3600,
            "token_type": "Bearer"
        }), 200
    else:
        return jsonify({"error": "Invalid refresh token"}), 401


@app.route("/model-info")
def model_info():
    """
    Model Information Endpoint
    Returns information about the currently loaded embedding model.
    ---
    responses:
      200:
        description: Information about the embedding model.
    """
    try:
        info = get_model_info()
        return jsonify({"status": "success", "model_info": info})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/collection-info")
def collection_info():
    """
    Collection Information Endpoint
    Returns information about the Qdrant vector database collection.
    ---
    responses:
      200:
        description: Information about the vector database collection.
    """
    try:
        info = get_collection_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/collection-stats")
def collection_stats():
    """
    Collection Statistics Endpoint
    Returns detailed statistics about the vector database collection.
    ---
    responses:
      200:
        description: Detailed collection statistics.
    """
    try:
        stats = get_collection_stats()
        return jsonify({"status": "success", "stats": stats})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/search", methods=["POST"])
def search():
    """
    Semantic Search Endpoint
    Search for similar chunks using semantic similarity.
    ---
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            query:
              type: string
              description: The search query text.
            limit:
              type: integer
              description: Maximum number of results (default 10).
            score_threshold:
              type: number
              description: Minimum similarity score 0.0-1.0 (default 0.7).
            include_metadata:
              type: boolean
              description: Include full metadata in results (default true).
    responses:
      200:
        description: Search results with similarity scores.
      400:
        description: Bad request - missing query.
      500:
        description: Internal server error.
    """
    try:
        data = request.get_json()
        if not data or not data.get("query"):
            return jsonify({"status": "error", "error": "Query text is required"}), 400

        query_text = data["query"]
        limit = data.get("limit", 10)
        score_threshold = data.get("score_threshold", 0.7)
        include_metadata = data.get("include_metadata", True)

        results = search_similar_chunks(
            query_text=query_text,
            limit=limit,
            score_threshold=score_threshold,
            include_metadata=include_metadata,
        )

        return jsonify(
            {
                "status": "success",
                "query": query_text,
                "results_count": len(results),
                "results": results,
            }
        )

    except Exception as e:
        logger.error("Search endpoint error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/documents/<document_id>")
def get_document(document_id):
    """
    Get Document Chunks Endpoint
    Retrieve all chunks for a specific document.
    ---
    parameters:
      - name: document_id
        in: path
        type: string
        required: true
        description: The document ID to retrieve.
      - name: include_text
        in: query
        type: boolean
        description: Include chunk text in results (default true).
    responses:
      200:
        description: Document chunks.
      404:
        description: Document not found.
      500:
        description: Internal server error.
    """
    try:
        include_text = request.args.get("include_text", "true").lower() == "true"

        results = search_by_document(document_id=document_id, include_text=include_text)

        if not results:
            return (
                jsonify(
                    {"status": "error", "error": "Document {document_id} not found"}
                ),
                404,
            )

        return jsonify(
            {
                "status": "success",
                "document_id": document_id,
                "chunks_count": len(results),
                "chunks": results,
            }
        )

    except Exception as e:
        logger.error("Get document endpoint error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/documents/<document_id>", methods=["DELETE"])
def delete_document_endpoint(document_id):
    """
    Delete Document Endpoint
    Delete all chunks for a specific document.
    ---
    parameters:
      - name: document_id
        in: path
        type: string
        required: true
        description: The document ID to delete.
    responses:
      200:
        description: Document deleted successfully.
      404:
        description: Document not found.
      500:
        description: Internal server error.
    """
    try:
        success = delete_document(document_id)

        if success:
            return jsonify(
                {
                    "status": "success",
                    "message": "Document {document_id} deleted successfully",
                }
            )
        else:
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": "Failed to delete document {document_id}",
                    }
                ),
                500,
            )

    except Exception as e:
        logger.error("Delete document endpoint error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


# =============================================================================
# Multi-Agent System Endpoints
# =============================================================================


@app.route("/agents", methods=["GET"])
def get_agents_status():
    """
    Get Agents Status Endpoint
    Returns the status of all agents in the multi-agent system.
    ---
    responses:
      200:
        description: Status of all agents.
      500:
        description: Internal server error.
    """
    try:
        metrics = agent_orchestrator.get_agent_metrics()
        return jsonify({"status": "success", "agents": metrics})
    except Exception as e:
        logger.error("Error getting agents status: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/agents/<agent_type>", methods=["GET"])
def get_agent_details(agent_type):
    """
    Get Agent Details Endpoint
    Returns detailed information about a specific agent.
    ---
    parameters:
      - name: agent_type
        in: path
        type: string
        required: true
        description: Type of agent (planner, research, reasoner, executor)
    responses:
      200:
        description: Agent details.
      404:
        description: Agent not found.
      500:
        description: Internal server error.
    """
    try:
        if agent_type not in agent_orchestrator.agents:
            return (
                jsonify(
                    {"status": "error", "error": "Agent type '{agent_type}' not found"}
                ),
                404,
            )

        agent = agent_orchestrator.agents[agent_type]
        details = agent.get_metrics()

        return jsonify(
            {"status": "success", "agent_type": agent_type, "details": details}
        )
    except Exception as e:
        logger.error("Error getting agent details: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/agents/<agent_type>/activate", methods=["POST"])
def activate_agent(agent_type):
    """
    Activate Agent Endpoint
    Activates a specific agent.
    ---
    parameters:
      - name: agent_type
        in: path
        type: string
        required: true
        description: Type of agent to activate
    responses:
      200:
        description: Agent activated successfully.
      500:
        description: Internal server error.
    """
    try:
        agent_orchestrator.activate_agent(agent_type)
        return jsonify(
            {
                "status": "success",
                "message": "Agent {agent_type} activated successfully",
            }
        )
    except Exception as e:
        logger.error("Error activating agent: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/agents/<agent_type>/deactivate", methods=["POST"])
def deactivate_agent(agent_type):
    """
    Deactivate Agent Endpoint
    Deactivates a specific agent.
    ---
    parameters:
      - name: agent_type
        in: path
        type: string
        required: true
        description: Type of agent to deactivate
    responses:
      200:
        description: Agent deactivated successfully.
      500:
        description: Internal server error.
    """
    try:
        agent_orchestrator.deactivate_agent(agent_type)
        return jsonify(
            {
                "status": "success",
                "message": "Agent {agent_type} deactivated successfully",
            }
        )
    except Exception as e:
        logger.error("Error deactivating agent: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/agents/reset", methods=["POST"])
def reset_all_agents():
    """
    Reset All Agents Endpoint
    Resets all agents to initial state.
    ---
    responses:
      200:
        description: All agents reset successfully.
      500:
        description: Internal server error.
    """
    try:
        agent_orchestrator.reset_agents()
        return jsonify(
            {"status": "success", "message": "All agents reset successfully"}
        )
    except Exception as e:
        logger.error("Error resetting agents: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/workflows", methods=["GET"])
def get_workflow_history():
    """
    Get Workflow History Endpoint
    Returns the execution history of workflows.
    ---
    parameters:
      - name: limit
        in: query
        type: integer
        description: Maximum number of workflows to return (default 10)
    responses:
      200:
        description: Workflow history.
      500:
        description: Internal server error.
    """
    try:
        limit = request.args.get("limit", 10, type=int)
        history = agent_orchestrator.get_workflow_history(limit)

        return jsonify(
            {"status": "success", "workflows": history, "total_workflows": len(history)}
        )
    except Exception as e:
        logger.error("Error getting workflow history: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/workflows/<workflow_id>", methods=["GET"])
def get_workflow_details(workflow_id):
    """
    Get Workflow Details Endpoint
    Returns detailed information about a specific workflow.
    ---
    parameters:
      - name: workflow_id
        in: path
        type: string
        required: true
        description: ID of the workflow
    responses:
      200:
        description: Workflow details.
      404:
        description: Workflow not found.
      500:
        description: Internal server error.
    """
    try:
        workflow = agent_orchestrator.get_workflow(workflow_id)

        if not workflow:
            return (
                jsonify(
                    {"status": "error", "error": "Workflow '{workflow_id}' not found"}
                ),
                404,
            )

        return jsonify({"status": "success", "workflow": workflow})
    except Exception as e:
        logger.error("Error getting workflow details: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/capabilities", methods=["GET"])
def get_agent_capabilities():
    """
    Get Agent Capabilities Endpoint
    Returns the capabilities of all agents.
    ---
    responses:
      200:
        description: Agent capabilities.
      500:
        description: Internal server error.
    """
    try:
        capabilities = agent_orchestrator.get_agent_capabilities()

        return jsonify({"status": "success", "capabilities": capabilities})
    except Exception as e:
        logger.error("Error getting agent capabilities: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    Chat Endpoint
    Process a user query using the multi-agent system.
    ---
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            query:
              type: string
              description: The user's query
            context:
              type: object
              description: Additional context for the query
    responses:
      200:
        description: Chat response from multi-agent system.
      400:
        description: Bad request - missing query.
      500:
        description: Internal server error.
    """
    try:
        data = request.get_json()
        if not data or not data.get("query"):
            return jsonify({"status": "error", "error": "Query is required"}), 400

        query = data["query"]

        logger.info("Processing chat query: {query[:50]}...")

        # Execute workflow using multi-agent system (synchronous wrapper)

        result = asyncio.run(agent_orchestrator.execute_workflow(query, {}))

        return jsonify({"status": "success", "result": result})

    except Exception as e:
        logger.error("Error processing chat query: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/test/workflow", methods=["POST"])
def test_workflow():
    """
    Test Workflow Endpoint
    Test a complete workflow execution.
    ---
    parameters:
      - name: query
        in: query
        type: string
        required: true
        description: Test query to execute
    responses:
      200:
        description: Workflow test result.
      400:
        description: Bad request - missing query.
      500:
        description: Internal server error.
    """
    try:
        query = request.args.get("query")
        if not query:
            return (
                jsonify({"status": "error", "error": "Query parameter is required"}),
                400,
            )

        logger.info("Testing workflow with query: {query}")

        # Execute workflow using multi-agent system (synchronous wrapper)

        result = asyncio.run(agent_orchestrator.execute_workflow(query))

        return jsonify({"status": "success", "test_result": result})

    except Exception as e:
        logger.error("Error testing workflow: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/admin/health", methods=["GET"])
def admin_health_check():
    """
    Admin Health Check Endpoint
    Returns health status of the multi-agent system.
    ---
    responses:
      200:
        description: Health status.
      500:
        description: Internal server error.
    """
    try:
        available_agents = agent_orchestrator.get_available_agents()
        agent_metrics = agent_orchestrator.get_agent_metrics()

        health_status = {
            "status": "healthy",
            "available_agents": available_agents,
            "total_agents": len(available_agents),
            "active_agents": agent_metrics.get("active_agents", 0),
            "total_workflows": agent_metrics.get("total_workflows", 0),
        }

        return jsonify(health_status)

    except Exception as e:
        logger.error("Admin health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    """
    Upload a File or a URL to start the ingestion pipeline.
    ---
    consumes:
      - multipart/form-data
      - application/json
    parameters:
      - name: file
        in: formData
        type: file
        description: The PDF file to upload.
      - name: body
        in: body
        required: false
        schema:
          type: object
          properties:
            url:
              type: string
              description: The URL of the PDF to download.
    responses:
      200:
        description: Document ingestion started successfully.
      400:
        description: Bad request. Please provide either a file or a URL.
      500:
        description: Internal server error during ingestion.
    """
    file = None
    url = None
    file_path = None
    filename = None

    if "multipart/form-data" in request.content_type:
        file = request.files.get("file")
        if not file:
            return make_response(
                jsonify({"detail": "No file part in multipart/form-data request."}),
                400,
            )
        filename = file.filename
        file_path = os.path.join(UPLOADS_DIR, filename)
        file.save(file_path)

    elif request.is_json:
        data = request.get_json()
        if data:
            url = data.get("url")
        if not url:
            return make_response(
                jsonify({"detail": "No URL provided in JSON body."}),
                400,
            )
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            filename = url.split("/")[-1] or "{uuid.uuid4()}.pdf"
            file_path = os.path.join(UPLOADS_DIR, filename)
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            logger.error("Failed to download file from URL {url}. Error: {e}")
            return make_response(
                jsonify({"detail": "Failed to download file from URL. {e}"}), 500
            )

    if not file_path:
        return make_response(
            jsonify(
                {
                    "detail": (
                        "Please provide a file (multipart/form-data) or "
                        "a URL (application/json)."
                    )
                }
            ),
            400,
        )

    # --- Start Enhanced Ingestion Pipeline ---
    try:
        logger.info("Starting enhanced ingestion for {filename}...")
        document_id = str(uuid.uuid4())

        # Use enhanced processing with page structure preservation
        success = process_pdf_with_page_structure(file_path, document_id)

        if not success:
            logger.warning(
                "Enhanced processing failed for {filename}, falling back to basic processing..."
            )

            # Fallback to basic processing
            # 1. Extract Text
            text = extract_text_from_pdf(file_path)
            if not text:
                return make_response(
                    jsonify({"detail": "Could not extract text from {filename}."}), 500
                )

            # 2. Chunk Text
            chunks = chunk_text(text)

            # 3. Generate Embeddings
            embeddings = generate_embeddings(chunks)

            # 4. Store in Vector DB
            store_embeddings(embeddings, chunks, document_id)

        logger.info("Successfully completed ingestion for document_id: {document_id}")
        return jsonify(
            {
                "message": (
                    "File '{filename}' processed successfully with "
                    "page structure preservation."
                ),
                "document_id": document_id,
            }
        )

    except Exception as e:
        logger.error(
            "An error occurred during the ingestion pipeline for {filename}. Error: {e}"
        )
        return make_response(
            jsonify({"detail": "An error occurred during processing."}), 500
        )


if __name__ == "__main__":
    app.run(debug=False, port=8008, use_reloader=False)
