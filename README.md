# DocQuest

**DocQuest** is an intelligent document understanding system that allows users to upload academic or technical documents (e.g., lecture notes, PDFs) and automatically:
- Extract relevant content  
- Generate summaries for each section  
- Create comprehension-based quizzes  
- Enable question answering via a vector store  
It leverages state-of-the-art NLP models (from Hugging Face and LangChain) and provides a RESTful Flask backend for interaction.
---

## ⚙️ Features
- 📄 PDF upload and parsing (via PyMuPDF)  
- 🔍 Intelligent section splitting  
- 🧠 Chunk-wise embedding with FAISS  
- 📝 Summary generation via LLM APIs  
- 🧪 Quiz generation with context reasoning  
- ❓ Ask-anything Q&A over document chunks  
- 🔗 RAG-style (Retrieval-Augmented Generation) architecture  
---

## Architecture & Reasoning Flow
### 📊 Flow Diagram (Textual Representation)
```text
        [PDF Upload]
             |
       [Text Extraction]
             |
     [Section-wise Splitting]
             |
      [Chunk Generation]
             |
   ┌─────────┴─────────┐
   |                   |
[Summarizer]     [Embeddings + FAISS]
   |                   |
[Quiz Generator]       |
   |                   |
[Question & Answer Pipeline (LLM)]
             |
       [Final Output JSON]
```

## 📦 Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/garimabhayana/DocQuest.git
cd DocQuest
```
### 2. Create & Activate Virtual Environment (Python 3.11+)
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Dependencies
``` bash
pip install -r requirements.txt
```
### 4. Add Environment Variables
Create a .env file in the root directory with:
```env
HF_TOKEN=your_huggingface_token
```
### 5. Run the Flask Backend
``` bash
python backend/app.py
```
You should see:
✅ Flask received file upload request.

## 🚀 API Endpoints
Endpoint	Method	Description
/process	POST	Uploads a document & processes it
/challenge/init	POST	Generates quiz questions from chunks
/ask	POST	Asks question over processed doc

## ✅ Example Usage
- Upload a PDF using /process
- Quiz and summary are auto-generated
- Call /ask with user question to get answers with references

## 📁 Project Structure
``` bash
DocQuest
├── backend/
    ├── app.py                 # Main Flask app
    ├── processor.py           # PDF extraction + section chunking
    ├── logic_quiz.py          # Quiz generation logic
    ├── qa_engine.py           # Embedding, vector store, Q&A
├── frontend/
    ├── app.py                 # Main frontend app
├── requirements.txt       # Python dependencies
└── .env                   # Environment secrets (not committed)
```
## 🧪 Models Used
- Embedding : sentence-transformers/all-MiniLM-L6-v2
- Question Generation	Same / Custom Prompt via LangChain
- Q&A (RAG)	Vector store + LLM-based inference

## 💡 Notes
This project is designed for offline academic PDFs or technical documentation
Ideal for educators, learners, and content creators
Built with extensibility in mind – supports easy model swapping
