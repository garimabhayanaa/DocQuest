from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from qa_engine import retrieve_relevant_chunks, get_document_summary
import os
from dotenv import load_dotenv
import logging
import time
import re

logger = logging.getLogger(__name__)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
load_dotenv()


def get_llm():
    """Get LLM with error handling and retries"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")
    try:
        llm = HuggingFaceEndpoint(
            model="mistralai/Devstral-Small-2507",
            huggingfacehub_api_token=hf_token,
            temperature=0.4,  # Slightly higher for creative questions
            max_new_tokens=400,
            repetition_penalty=1.1,
            do_sample=True,
            top_p=0.9
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM with model 'mistralai/Devstral-Small-2507': {e}")
        raise ValueError("Failed to initialize any LLM model")


def extract_text_from_response(response):
    """Extract text from various response formats"""
    if isinstance(response, str):
        return response
    elif hasattr(response, 'choices') and response.choices:
        # Handle OpenAI-style response format
        if hasattr(response.choices[0], 'message'):
            return response.choices[0].message.content
        elif hasattr(response.choices[0], 'text'):
            return response.choices[0].text
        else:
            return str(response.choices[0])
    elif hasattr(response, 'content'):
        return response.content
    elif hasattr(response, 'text'):
        return response.text
    else:
        return str(response)


def document_to_dict(doc):
    """Convert Document object to JSON-serializable dictionary"""
    if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
        return {
            "text": doc.page_content,
            "section": doc.metadata.get("section", "Unknown"),
            "paragraph": doc.metadata.get("paragraph", 0),
            "chunk_id": doc.metadata.get("chunk_id", 0)
        }
    return doc  # Already a dict


def generate_challenge_questions(doc_chunks, num=3):
    """Generate challenge questions with improved error handling and anti-hallucination"""
    try:
        llm = get_llm()

        # Convert Document objects to dictionaries if needed
        processed_chunks = []
        for chunk in doc_chunks:
            if hasattr(chunk, 'page_content'):  # It's a Document object
                processed_chunks.append(document_to_dict(chunk))
            else:  # It's already a dict
                processed_chunks.append(chunk)

        # Select diverse chunks for question generation
        selected_chunks = []
        sections_seen = set()

        for chunk in processed_chunks:
            section = chunk.get('section', 'Unknown')
            if section not in sections_seen and len(selected_chunks) < 8:
                selected_chunks.append(chunk)
                sections_seen.add(section)

        if not selected_chunks:
            selected_chunks = processed_chunks[:5]

        # Build context with clear section markers
        context_parts = []
        for i, chunk in enumerate(selected_chunks):
            section = chunk.get('section', 'Unknown')
            text = chunk.get('text', '')[:300]  # Limit chunk length
            context_parts.append(f"[SECTION {i + 1}: {section}]\n{text}")

        full_context = "\n\n".join(context_parts)

        # Strict prompt for document-based questions
        prompt = f"""Generate {num} analytical questions based ONLY on the provided document content.

REQUIREMENTS:
1. Questions must be answerable using ONLY the document content
2. Focus on comprehension, analysis, and connections between sections
3. Avoid questions requiring external knowledge
4. Make questions thought-provoking but document-grounded
5. Format as numbered list

Document Content:
{full_context}

Generate {num} questions:"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating questions, attempt {attempt + 1}")
                response = llm.invoke(prompt)

                # Extract text from response regardless of format
                response_text = extract_text_from_response(response)

                if response_text and response_text.strip():
                    # Parse and validate questions
                    questions = parse_and_validate_questions(response_text, selected_chunks)

                    if len(questions) >= num:
                        logger.info(f"Generated {len(questions)} questions successfully")
                        return questions[:num]
                    else:
                        logger.warning(f"Only {len(questions)} valid questions generated")
                        if questions:  # Return what we have if some valid questions
                            return questions + generate_fallback_questions(selected_chunks, num - len(questions))

                logger.warning(f"Empty or invalid response from LLM, attempt {attempt + 1}")

            except Exception as e:
                logger.error(f"Question generation failed, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        # Fallback questions
        logger.info("Using fallback questions")
        return generate_fallback_questions(selected_chunks, num)

    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        return generate_fallback_questions(doc_chunks, num)


def parse_and_validate_questions(response, doc_chunks):
    """Parse and validate questions from LLM response"""
    questions = []

    # Extract questions using multiple patterns
    patterns = [
        r'^\d+\.\s*(.+\?)\s*$',  # 1. Question?
        r'^\d+\)\s*(.+\?)\s*$',  # 1) Question?
        r'^\-\s*(.+\?)\s*$',  # - Question?
        r'^\•\s*(.+\?)\s*$',  # • Question?
        r'^(.+\?)\s*$'  # Just Question?
    ]

    lines = response.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        for pattern in patterns:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                question = match.group(1).strip()
                if is_valid_question(question, doc_chunks):
                    questions.append(question)
                break

    return questions


def is_valid_question(question, doc_chunks):
    """Validate if question is appropriate and answerable from document"""
    if not question or len(question) < 10:
        return False

    if not question.endswith('?'):
        return False

    # Check for common issues
    invalid_patterns = [
        r'external\s+knowledge',
        r'outside\s+the\s+document',
        r'research\s+further',
        r'google\s+search',
        r'additional\s+sources'
    ]

    question_lower = question.lower()
    for pattern in invalid_patterns:
        if re.search(pattern, question_lower):
            return False

    # Check if question has some connection to document content
    doc_text = " ".join([chunk.get('text', '') for chunk in doc_chunks[:5]]).lower()
    question_words = set(question_lower.split())
    doc_words = set(doc_text.split())

    # Should have some overlap with document content
    overlap = len(question_words.intersection(doc_words))
    return overlap > 2


def generate_fallback_questions(doc_chunks, num=3):
    """Generate fallback questions based on document structure"""
    if not doc_chunks:
        return [
                   "What are the main points discussed in this document?",
                   "How are the different sections related to each other?",
                   "What conclusions can be drawn from the content?"
               ][:num]

    # Convert Document objects to dictionaries if needed
    processed_chunks = []
    for chunk in doc_chunks:
        if hasattr(chunk, 'page_content'):  # It's a Document object
            processed_chunks.append(document_to_dict(chunk))
        else:  # It's already a dict
            processed_chunks.append(chunk)

    # Analyze document structure
    sections = set()
    key_terms = set()

    for chunk in processed_chunks[:10]:
        section = chunk.get('section', '')
        if section:
            sections.add(section)

        # Extract potential key terms (simple approach)
        text = chunk.get('text', '').lower()
        words = text.split()
        for word in words:
            if len(word) > 6 and word.isalpha():  # Longer words likely to be key terms
                key_terms.add(word)

    questions = []

    # Section-based questions
    if len(sections) > 1:
        questions.append(f"How do the different sections ({', '.join(list(sections)[:3])}) relate to each other?")

    # Content-based questions
    if key_terms:
        sample_terms = list(key_terms)[:3]
        questions.append(f"What is the significance of the key concepts discussed in this document?")

    # General analytical questions
    fallback_questions = [
        "What are the main themes or arguments presented in this document?",
        "What evidence or examples are provided to support the main points?",
        "What questions or implications does this document raise for further consideration?",
        "How might the information in this document be applied or interpreted?",
        "What connections can be made between different parts of this document?"
    ]

    # Combine and return
    questions.extend(fallback_questions)
    return questions[:num]


def evaluate_user_answer(question, user_answer):
    """Evaluate user answer with improved error handling and grounding"""
    try:
        # Retrieve relevant chunks for grounding
        chunks = retrieve_relevant_chunks(question, k=5)

        if not chunks:
            return "I cannot evaluate your answer as no relevant document content was found for this question."

        # Build context with clear source attribution
        context_parts = []
        for i, chunk in enumerate(chunks):
            # Handle both Document objects and dictionaries
            if hasattr(chunk, 'page_content'):  # Document object
                section = chunk.metadata.get("section", "Unknown Section")
                para = chunk.metadata.get("paragraph", "?")
                content = chunk.page_content[:300]
            else:  # Dictionary
                section = chunk.get("section", "Unknown Section")
                para = chunk.get("paragraph", "?")
                content = chunk.get("text", "")[:300]

            context_parts.append(f"[SOURCE {i + 1} - {section}, Para {para}]\n{content}")

        context = "\n\n".join(context_parts)

        # Limit context length
        if len(context) > 1800:
            context = context[:1800] + "..."

        llm = get_llm()

        # Detailed evaluation prompt with anti-hallucination measures
        eval_prompt = f"""You are evaluating a user's answer based ONLY on the provided document content.

EVALUATION CRITERIA:
1. Accuracy: Does the answer align with the document content?
2. Completeness: Does it address the question adequately?
3. Use of evidence: Does it reference or reflect document content?
4. Reasoning: Is the logic sound based on the document?

Document Content:
{context}

Question: {question}

User's Answer: {user_answer}

Provide evaluation feedback focusing on:
- What the user got right (reference specific sources)
- What could be improved
- How well they used the document content
- Specific suggestions for better answers

Evaluation:"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Evaluating answer, attempt {attempt + 1}")
                response = llm.invoke(eval_prompt)

                # Extract text from response regardless of format
                response_text = extract_text_from_response(response)

                if response_text and response_text.strip():
                    # Validate the evaluation is grounded
                    if is_evaluation_grounded(response_text, context, user_answer):
                        logger.info("Answer evaluation successful")
                        return response_text.strip()
                    else:
                        logger.warning("Evaluation appears ungrounded, using fallback")
                        return generate_fallback_evaluation(question, user_answer, chunks)
                else:
                    logger.warning(f"Empty evaluation response, attempt {attempt + 1}")

            except Exception as e:
                logger.error(f"Answer evaluation failed, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        return generate_fallback_evaluation(question, user_answer, chunks)

    except Exception as e:
        logger.error(f"Answer evaluation failed: {e}")
        return generate_fallback_evaluation(question, user_answer, [])


def is_evaluation_grounded(evaluation, context, user_answer):
    """Check if evaluation is grounded in document content"""
    try:
        # Check if evaluation mentions sources or document content
        if "SOURCE" in evaluation.upper() or "SECTION" in evaluation.upper():
            return True

        # Check for reasonable overlap with context and user answer
        eval_words = set(evaluation.lower().split())
        context_words = set(context.lower().split())
        answer_words = set(user_answer.lower().split())

        context_overlap = len(eval_words.intersection(context_words))
        answer_overlap = len(eval_words.intersection(answer_words))

        return context_overlap > 5 and answer_overlap > 3

    except Exception as e:
        logger.error(f"Error checking evaluation grounding: {e}")
        return False


def generate_fallback_evaluation(question, user_answer, chunks):
    """Generate fallback evaluation without LLM"""
    if not user_answer or len(user_answer.strip()) < 5:
        return "Your answer is very brief. Please provide more detail and explanation based on the document content."

    word_count = len(user_answer.split())

    evaluation = f"Thank you for your {word_count}-word response. "

    if chunks:
        # Try to find connections between answer and document
        doc_content = ""
        for chunk in chunks[:3]:
            if hasattr(chunk, 'page_content'):  # Document object
                doc_content += chunk.page_content + " "
            else:  # Dictionary
                doc_content += chunk.get('text', '') + " "

        answer_words = set(user_answer.lower().split())
        doc_words = set(doc_content.lower().split())
        overlap = len(answer_words.intersection(doc_words))

        if overlap > 10:
            evaluation += "Your answer shows good engagement with the document content. "
        elif overlap > 5:
            evaluation += "Your answer has some connection to the document content. "
        else:
            evaluation += "Consider referencing more specific details from the document. "

    if word_count < 20:
        evaluation += "Try to provide more detailed explanations and examples from the document."
    elif word_count > 150:
        evaluation += "Good detail in your response - you've provided comprehensive coverage."
    else:
        evaluation += "Good length for your response."

    return evaluation
