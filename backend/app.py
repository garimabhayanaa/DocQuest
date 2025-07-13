from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import traceback
import logging
from processor import process_document
from qa_engine import build_vector_store, retrieve_relevant_chunks, generate_answer_with_sources
from logic_quiz import generate_challenge_questions, evaluate_user_answer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
load_dotenv()

# Global variable to store document chunks for challenge questions
document_chunks = []


def validate_environment():
    """Validate required environment variables and dependencies"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/uploads", exist_ok=True)

    logger.info("Environment validation successful")


# Validate on startup
try:
    validate_environment()
except Exception as e:
    logger.error(f"Environment validation failed: {e}")
    raise


@app.route('/process', methods=['POST'])
def process():
    """Process uploaded document (PDF or TXT)"""
    global document_chunks

    logger.info("Flask received file upload request.")

    file = request.files.get('file')
    if not file:
        logger.error("No file uploaded.")
        return jsonify({"error": "No file uploaded"}), 400

    if not file.filename:
        logger.error("No filename provided.")
        return jsonify({"error": "No filename provided"}), 400

    # Support both PDF and TXT files
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        logger.error("File format not supported.")
        return jsonify({"error": "Only PDF and TXT files are supported"}), 400

    try:
        logger.info(f"Processing file: {file.filename}")
        file.seek(0)  # Reset file pointer

        result = process_document(file)
        logger.info("Document processed successfully.")

        # Store chunks globally for challenge questions
        document_chunks = result.get("chunks", [])
        logger.info(f"Stored {len(document_chunks)} chunks for challenge questions")

        # Build vector store
        try:
            if result.get("chunks"):
                build_vector_store(result["chunks"])
                logger.info("Vector store built successfully.")
                result["vector_store_status"] = "success"
            else:
                logger.warning("No chunks to build vector store")
                result["vector_store_status"] = "no_chunks"
        except Exception as vs_error:
            logger.error(f"Vector store build failed: {vs_error}")
            result["vector_store_error"] = str(vs_error)
            result["vector_store_status"] = "failed"

        # Add document metadata
        result["document_info"] = {
            "filename": file.filename,
            "total_chunks": len(document_chunks),
            "processing_status": "success"
        }

        return jsonify(result)

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in /process: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route('/ask', methods=['POST'])
def ask():
    """Handle Q&A requests with improved error handling"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        query = data.get("query")
        if not query or not query.strip():
            return jsonify({"error": "Query is required and cannot be empty"}), 400

        logger.info(f"Processing Q&A query: {query[:100]}...")

        result = generate_answer_with_sources(query)

        # Add metadata to response
        result["query"] = query
        result["response_type"] = "qa"

        logger.info("Q&A response generated successfully")
        return jsonify(result)

    except FileNotFoundError:
        logger.error("Vector store not found")
        return jsonify({
            "error": "No document has been processed yet. Please upload a document first.",
            "suggestion": "Use the /process endpoint to upload a PDF or TXT file"
        }), 400
    except Exception as e:
        logger.error(f"Error in /ask: {e}")
        return jsonify({
            "error": f"Failed to generate answer: {str(e)}",
            "response_type": "error"
        }), 500


@app.route('/challenge/init', methods=['POST'])
def challenge_init():
    """Initialize challenge questions with improved error handling"""
    try:
        logger.info("Challenge init called")

        # Use global document chunks if available
        if document_chunks:
            chunks = document_chunks
            logger.info(f"Using stored document chunks: {len(chunks)} chunks")
        else:
            # Fallback to request data
            data = request.json
            chunks = data.get("chunks") if data else None

            if not chunks:
                logger.error("No chunks found in request or stored globally.")
                return jsonify({
                    "error": "No document chunks available. Please process a document first.",
                    "suggestion": "Use the /process endpoint to upload a document"
                }), 400

        # Validate chunks structure
        if not isinstance(chunks, list) or not chunks:
            return jsonify({"error": "Invalid chunks format"}), 400

        # Generate questions
        questions = generate_challenge_questions(chunks)

        if not questions:
            return jsonify({
                "error": "Failed to generate challenge questions",
                "fallback_message": "Try uploading a different document or check if the document has sufficient content"
            }), 500

        logger.info(f"Generated {len(questions)} challenge questions")

        return jsonify({
            "questions": questions,
            "total_questions": len(questions),
            "document_chunks": len(chunks),
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error in /challenge/init: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to initialize challenge: {str(e)}",
            "status": "error"
        }), 500


@app.route('/challenge/evaluate', methods=['POST'])
def challenge_evaluate():
    """Evaluate user's answer to challenge question with improved validation"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        question = data.get("question")
        answer = data.get("answer")

        if not question or not question.strip():
            return jsonify({"error": "Question is required and cannot be empty"}), 400

        if not answer or not answer.strip():
            return jsonify({"error": "Answer is required and cannot be empty"}), 400

        logger.info(f"Evaluating answer for question: {question[:50]}...")

        feedback = evaluate_user_answer(question, answer)

        if not feedback:
            return jsonify({
                "error": "Failed to generate evaluation feedback",
                "fallback_message": "Your answer has been received but cannot be evaluated at this time."
            }), 500

        logger.info("Answer evaluation completed successfully")

        return jsonify({
            "feedback": feedback,
            "question": question,
            "user_answer": answer,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error in /challenge/evaluate: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to evaluate answer: {str(e)}",
            "status": "error"
        }), 500


@app.route('/document/info', methods=['GET'])
def document_info():
    """Get information about the currently processed document"""
    try:
        if not document_chunks:
            return jsonify({
                "error": "No document currently processed",
                "suggestion": "Upload a document using the /process endpoint"
            }), 400

        # Extract document statistics
        sections = set()
        total_chars = 0

        for chunk in document_chunks:
            section = chunk.get('section', 'Unknown')
            sections.add(section)
            total_chars += len(chunk.get('text', ''))

        return jsonify({
            "total_chunks": len(document_chunks),
            "sections": list(sections),
            "total_sections": len(sections),
            "total_characters": total_chars,
            "average_chunk_size": total_chars // len(document_chunks) if document_chunks else 0,
            "status": "available"
        })

    except Exception as e:
        logger.error(f"Error in /document/info: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    try:
        # Check if vector store exists
        vector_store_status = "not_built"
        if os.path.exists("data/faiss_index"):
            vector_store_status = "available"

        # Check document status
        document_status = "none"
        if document_chunks:
            document_status = f"loaded_{len(document_chunks)}_chunks"

        return jsonify({
            "status": "healthy",
            "message": "Backend is running",
            "vector_store": vector_store_status,
            "document": document_status,
            "endpoints": {
                "process": "/process - Upload PDF/TXT documents",
                "ask": "/ask - Ask questions about the document",
                "challenge_init": "/challenge/init - Generate challenge questions",
                "challenge_evaluate": "/challenge/evaluate - Evaluate answers",
                "document_info": "/document/info - Get document statistics"
            }
        })

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/process", "/ask", "/challenge/init", "/challenge/evaluate", "/document/info",
                                "/health"]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again or contact support."
    }), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)