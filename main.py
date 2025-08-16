"""
Main entry point for the Intelligent Research Assistant.

This file provides both FastAPI and Flask applications for different use cases.
"""

import uvicorn
from src.main import app as fastapi_app

if __name__ == "__main__":
    # Run FastAPI application
    uvicorn.run(
        "src.main:app",
        host="127.0.0.1",
        port=8008,
        reload=True,
        log_level="info"
    ) 