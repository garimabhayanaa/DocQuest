import re
import fitz  # PyMuPDF
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
import logging
import os
from dotenv import load_dotenv
from typing import Dict, List, Any
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Section headers for structured documents
SECTION_HEADERS = [
    "Abstract", "Introduction", "Background", "Methodology", "Methods",
    "Results", "Discussion", "Conclusion", "References", "Literature Review",
    "Related Work", "Experiments", "Analysis", "Findings", "Summary",
    "Overview", "Executive Summary", "Objectives", "Scope", "Definitions"
]

embedding_model = None


def get_embedding_model():
    """Lazy initialization of embedding model"""
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    return embedding_model


def get_llm():
    """Get LLM with error handling"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")

    try:
        llm = HuggingFaceEndpoint(
            model="mistralai/Devstral-Small-2507",
            huggingfacehub_api_token=hf_token,
            temperature=0.3,
            max_new_tokens=150,
            repetition_penalty=1.1,
            do_sample=True,
            top_p=0.9
        )
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise


def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file"""
    try:
        file.seek(0)
        content = file.read()
        if not content:
            raise ValueError("PDF file is empty")

        pdf_stream = io.BytesIO(content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")

        if doc.page_count == 0:
            raise ValueError("PDF has no pages")

        text_parts = []
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")
                continue

        doc.close()

        if not text_parts:
            raise ValueError("No extractable text found in PDF")

        full_text = "\n".join(text_parts)
        logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
        return full_text

    except Exception as e:
        logger.error(f"PDF parsing error: {e}")
        raise ValueError(f"PDF parsing error: {str(e)}")


def extract_text_from_txt(file) -> str:
    """Extract text from TXT file"""
    try:
        file.seek(0)
        content = file.read()

        # Try to decode as UTF-8 first, then fall back to latin-1
        try:
            if isinstance(content, bytes):
                text = content.decode('utf-8')
            else:
                text = content
        except UnicodeDecodeError:
            if isinstance(content, bytes):
                text = content.decode('latin-1', errors='ignore')
            else:
                text = str(content)

        if not text.strip():
            raise ValueError("TXT file is empty or contains no readable text")

        logger.info(f"Successfully extracted {len(text)} characters from TXT")
        return text

    except Exception as e:
        logger.error(f"TXT parsing error: {e}")
        raise ValueError(f"TXT parsing error: {str(e)}")


def extract_text(file) -> str:
    """Extract text from either PDF or TXT file"""
    filename = file.filename.lower()

    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif filename.endswith('.txt'):
        return extract_text_from_txt(file)
    else:
        raise ValueError("Unsupported file format. Only PDF and TXT files are supported.")


def split_into_sections(text: str) -> Dict[str, str]:
    """Split text into logical sections"""
    sections = {}
    current_section = "Main Content"
    buffer = []
    lines = text.split("\n")

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Check if line is a section header
        is_section_header = False
        for header in SECTION_HEADERS:
            patterns = [
                rf"^{header}$",
                rf"^{header}\s*$",
                rf"^\d+\.\s*{header}$",
                rf"^\d+\.\d+\s*{header}$",
                rf"^{header.upper()}$",
                rf"^{header.lower()}$",
            ]

            for pattern in patterns:
                if re.match(pattern, line_clean, re.IGNORECASE):
                    is_section_header = True
                    # Save previous section
                    if buffer:
                        sections[current_section] = "\n".join(buffer).strip()
                    current_section = header
                    buffer = []
                    break

            if is_section_header:
                break

        if not is_section_header:
            buffer.append(line)

    # Add the last section
    if buffer:
        sections[current_section] = "\n".join(buffer).strip()

    # Remove empty sections
    sections = {k: v for k, v in sections.items() if v.strip()}
    logger.info(f"Found sections: {list(sections.keys())}")
    return sections


def chunk_section(section_text: str) -> List[str]:
    """Chunk long sections into manageable pieces"""
    if not section_text.strip():
        return []

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_text(section_text)

        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 30]
        return chunks

    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        # Fallback: simple splitting
        return [section_text[i:i + 500] for i in range(0, len(section_text), 400)]


def generate_summary(sections: Dict[str, str]) -> str:
    """Generate document summary"""
    try:
        llm = get_llm()

        # Combine sections for summary generation
        combined = ""
        for section_name, section_text in list(sections.items())[:4]:
            section_snippet = section_text[:400]  # Limit section length
            combined += f"\n{section_name}: {section_snippet}\n"

        combined = combined[:1800]  # Limit total length

        prompt = f"""Create a concise summary of this document in exactly 150 words or less. Focus on the main purpose, key findings, and conclusions:

{combined}

Summary (150 words max):"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm.invoke(prompt)
                if response and response.strip():
                    # Ensure summary is within word limit
                    words = response.strip().split()
                    if len(words) > 150:
                        summary = " ".join(words[:150]) + "..."
                    else:
                        summary = response.strip()

                    logger.info("Summary generated successfully")
                    return summary

            except Exception as e:
                logger.error(f"Summary generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        # Fallback summary
        return generate_fallback_summary(sections)

    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return generate_fallback_summary(sections)


def generate_fallback_summary(sections: Dict[str, str]) -> str:
    """Generate fallback summary without LLM"""
    logger.info("Generating fallback summary")

    summary_parts = []
    for section_name, section_text in list(sections.items())[:3]:
        sentences = section_text.split('.')[:2]
        if sentences:
            clean_text = '. '.join(sentences).strip()
            if clean_text and len(clean_text) > 20:
                summary_parts.append(f"{section_name}: {clean_text}")

    if summary_parts:
        return "Document Summary:\n" + "\n".join(summary_parts)
    else:
        return f"Document contains {len(sections)} sections with processed content ready for analysis."


def process_document(file) -> Dict[str, Any]:
    """Main document processing function"""
    logger.info("Starting document processing")

    try:
        # Extract text
        text = extract_text(file)
        logger.info("Text extraction successful")

        # Split into sections
        sections_raw = split_into_sections(text)
        logger.info(f"Section splitting successful: {list(sections_raw.keys())}")

        if not sections_raw:
            raise ValueError("No sections found in document")

        # Chunk sections
        sections = {}
        chunks = []

        for sec_name, sec_text in sections_raw.items():
            try:
                chunked = chunk_section(sec_text)
                logger.info(f"Chunked section '{sec_name}' into {len(chunked)} chunks")

                sections[sec_name] = chunked

                for idx, chunk_text in enumerate(chunked):
                    chunks.append({
                        "section": sec_name,
                        "paragraph": idx + 1,
                        "text": chunk_text
                    })
            except Exception as e:
                logger.error(f"Failed to chunk section '{sec_name}': {e}")
                continue

        if not chunks:
            raise ValueError("No chunks generated from document")

        logger.info(f"Generated {len(chunks)} total chunks")

        # Generate summary
        try:
            summary = generate_summary(sections_raw)
            logger.info("Summary generated successfully")
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            summary = "Summary generation failed. Document processed successfully."

        result = {
            "sections": list(sections.keys()),
            "chunks": chunks,
            "summary": summary,
            "stats": {
                "total_chunks": len(chunks),
                "total_sections": len(sections),
                "total_text_length": len(text)
            }
        }

        logger.info("Document processing completed successfully")
        return result

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise
