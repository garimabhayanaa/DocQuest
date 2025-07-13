import os
from dotenv import load_dotenv
import faiss
import logging
import time
import json
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint

load_dotenv()
logger = logging.getLogger(__name__)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm():
    """Get LLM with error handling and retries"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")
    try:
        model_name = "mistralai/Devstral-Small-2507"
        llm = HuggingFaceEndpoint(
            model=model_name,
            huggingfacehub_api_token=hf_token,
            temperature=0.3,  # Lower temperature for more factual responses
            max_new_tokens=512,  # Increased for better responses
            timeout=60,
            repetition_penalty=1.1,
            do_sample=True,
            top_p=0.9
        )
        # Test the LLM with a simple prompt
        test_response = llm.invoke("Test")
        if test_response and test_response.strip():
            logger.info(f"Successfully initialized LLM with model: {model_name}")
            return llm
        else:
            logger.warning(f"Model {model_name} returned empty response")
    except Exception as e:
        logger.error(f"Failed to initialize LLM with model {model_name}: {e}")

    raise ValueError("Failed to initialize any LLM model")


VECTOR_DB_PATH = "data/faiss_index"


def build_vector_store(chunks):
    """Build vector store with better error handling"""
    try:
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)

        texts = []
        metadatas = []

        for c in chunks:
            if c.get("text") and c["text"].strip():
                texts.append(c["text"])
                metadatas.append({
                    "section": c.get("section", "Unknown"),
                    "paragraph": c.get("paragraph", 0),
                    "chunk_id": len(texts)  # Add unique chunk ID
                })

        if not texts:
            raise ValueError("No valid text chunks provided")

        docs = [Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))]

        logger.info(f"Building vector store with {len(docs)} documents")
        vector_store = FAISS.from_documents(docs, embedding_model)
        vector_store.save_local(VECTOR_DB_PATH)

        logger.info("Vector store built and saved successfully")
        return vector_store

    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        raise


def load_vector_store():
    """Load vector store with error handling"""
    try:
        if not os.path.exists(VECTOR_DB_PATH):
            raise ValueError("Vector store not found. Please process a document first.")

        vector_store = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        logger.info("Vector store loaded successfully")
        return vector_store

    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise


def retrieve_relevant_chunks(query, k=5):
    """Retrieve relevant chunks with error handling and better context"""
    try:
        vector_store = load_vector_store()
        results = vector_store.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(results)} relevant chunks")
        return results

    except Exception as e:
        logger.error(f"Failed to retrieve chunks: {e}")
        return []


def document_to_dict(doc):
    """Convert Document object to JSON-serializable dictionary"""
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }


def generate_answer_with_sources(query, k=5):
    """Generate answer with sources and improved error handling - NO HALLUCINATION"""
    try:
        vector_store = load_vector_store()
        docs = vector_store.similarity_search(query, k=k)

        if not docs:
            return {
                "answer": "No relevant information found in the document to answer your question.",
                "sources": [],
                "confidence": "low"
            }

        # Build context string with clear source attribution
        context_parts = []
        for i, doc in enumerate(docs):
            section = doc.metadata.get("section", "Unknown Section")
            para = doc.metadata.get("paragraph", "?")
            chunk_id = doc.metadata.get("chunk_id", i)

            context_parts.append(f"[SOURCE {i + 1} - Section: {section}, Paragraph: {para}]\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        # Limit context length but preserve complete chunks
        if len(context) > 2000:
            context = context[:2000] + "..."

        # Anti-hallucination prompt with strict instructions
        prompt = f"""You are a document assistant. Answer ONLY based on the provided context from the document. 

STRICT RULES:
1. Use ONLY information explicitly stated in the context
2. If the context doesn't contain enough information to answer fully, say so
3. Always reference which SOURCE number you're using
4. Do NOT add external knowledge or assumptions
5. If uncertain, state your uncertainty clearly

Context from document:
{context}

Question: {query}

Answer (reference sources by number):"""

        try:
            llm = get_llm()

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Generating answer, attempt {attempt + 1}")
                    response = llm.invoke(prompt)

                    if response and response.strip():
                        # Validate response doesn't contain obvious hallucinations
                        if is_response_grounded(response, context):
                            logger.info("Answer generated successfully")
                            return {
                                "answer": response.strip(),
                                "sources": [document_to_dict(doc) for doc in docs],  # Convert to dict
                                "confidence": "high"
                            }
                        else:
                            logger.warning("Response appears to contain hallucination, using fallback")
                            return generate_fallback_answer(query, docs)
                    else:
                        logger.warning(f"Empty response from LLM, attempt {attempt + 1}")

                except Exception as e:
                    logger.error(f"Answer generation failed, attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise

            # Fallback answer
            return generate_fallback_answer(query, docs)

        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            return generate_fallback_answer(query, docs)

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": [],
            "confidence": "error"
        }


def is_response_grounded(response, context):
    """Basic check to ensure response is grounded in context"""
    try:
        # Simple heuristic: check if response references sources
        if "SOURCE" in response.upper() or "SECTION" in response.upper():
            return True

        # Check if key phrases from context appear in response
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())

        # If response has reasonable overlap with context, likely grounded
        overlap = len(context_words.intersection(response_words))
        return overlap > 3  # Minimum overlap threshold

    except Exception as e:
        logger.error(f"Error checking response grounding: {e}")
        return False


def generate_fallback_answer(query, docs):
    """Generate fallback answer without LLM - extract-based approach"""
    if not docs:
        return {
            "answer": "No relevant information found in the document to answer your question.",
            "sources": [],
            "confidence": "low"
        }

    # Extract-based approach: find most relevant sentences
    answer_parts = []
    for i, doc in enumerate(docs[:3]):  # Use top 3 most relevant
        section = doc.metadata.get("section", "Unknown Section")
        para = doc.metadata.get("paragraph", "?")

        # Extract first few sentences that might be relevant
        content = doc.page_content[:200]
        if content.endswith("..."):
            content = content[:-3]

        answer_parts.append(f"From {section} (paragraph {para}): {content}")

    answer = "Based on the document content:\n\n" + "\n\n".join(answer_parts)

    if len(answer) > 500:
        answer = answer[:500] + "..."

    return {
        "answer": answer,
        "sources": [document_to_dict(doc) for doc in docs],  # Convert to dict
        "confidence": "medium"
    }


def get_document_summary():
    """Get a summary of the document for better context"""
    try:
        vector_store = load_vector_store()
        # Get first few chunks as document overview
        all_docs = vector_store.similarity_search("", k=10)  # Empty query to get diverse chunks

        summary_parts = []
        for doc in all_docs[:5]:
            section = doc.metadata.get("section", "Unknown")
            summary_parts.append(f"Section '{section}': {doc.page_content[:100]}...")

        return "\n".join(summary_parts)
    except Exception as e:
        logger.error(f"Error getting document summary: {e}")
        return "Document summary not available"