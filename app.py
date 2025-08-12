import os
import uuid
import requests
from flask import Flask, request, jsonify, make_response
from flasgger import Swagger
from logging_config import logger

# Import pipeline modules
from src.pipeline.pipeline import (
    extract_text_from_pdf,
    chunk_text,
    generate_embeddings,
    store_embeddings,
    create_collection_if_not_exists,
    process_pdf_with_page_structure,
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

# Initialize the vector database at startup (compatible across Flask versions)
logger.info("Initializing application and vector database...")
create_collection_if_not_exists()

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
    return jsonify({"message": "Welcome to the File Upload API. Use the /upload endpoint to upload files."})

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


@app.route('/upload', methods=['POST'])
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

    if 'multipart/form-data' in request.content_type:
        file = request.files.get('file')
        if not file:
            return make_response(jsonify({"detail": "No file part in multipart/form-data request."}), 400)
        filename = file.filename
        file_path = os.path.join(UPLOADS_DIR, filename)
        file.save(file_path)

    elif request.is_json:
        data = request.get_json()
        if data:
            url = data.get('url')
        if not url:
             return make_response(jsonify({"detail": "No URL provided in JSON body."}), 400)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            filename = url.split("/")[-1] or f"{uuid.uuid4()}.pdf"
            file_path = os.path.join(UPLOADS_DIR, filename)
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download file from URL {url}. Error: {e}")
            return make_response(jsonify({"detail": f"Failed to download file from URL. {e}"}), 500)

    if not file_path:
        return make_response(jsonify({"detail": "Please provide a file (multipart/form-data) or a URL (application/json)."}), 400)

    # --- Start Enhanced Ingestion Pipeline ---
    try:
        logger.info(f"Starting enhanced ingestion for {filename}...")
        document_id = str(uuid.uuid4())

        # Use enhanced processing with page structure preservation
        success = process_pdf_with_page_structure(file_path, document_id)
        
        if not success:
            logger.warning(f"Enhanced processing failed for {filename}, falling back to basic processing...")
            
            # Fallback to basic processing
            # 1. Extract Text
            text = extract_text_from_pdf(file_path)
            if not text:
                return make_response(jsonify({"detail": f"Could not extract text from {filename}."}), 500)

            # 2. Chunk Text
            chunks = chunk_text(text)

            # 3. Generate Embeddings
            embeddings = generate_embeddings(chunks)

            # 4. Store in Vector DB
            store_embeddings(embeddings, chunks, document_id)

        logger.info(f"Successfully completed ingestion for document_id: {document_id}")
        return jsonify({
            "message": f"File '{filename}' processed successfully with page structure preservation.",
            "document_id": document_id
        })

    except Exception as e:
        logger.error(f"An error occurred during the ingestion pipeline for {filename}. Error: {e}")
        return make_response(jsonify({"detail": "An error occurred during processing."}), 500)

if __name__ == '__main__':
    app.run(debug=True, port=8008)
