import streamlit as st
import requests
import os
import time

# Ensure upload directory exists
upload_dir = "data/uploads"
os.makedirs(upload_dir, exist_ok=True)

st.set_page_config(
    page_title="Document Reasoning Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_URL = "http://127.0.0.1:5000"

# Professional minimal styling with high contrast
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles with strong contrast */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1a202c !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main content styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Force high contrast text for main content */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stWrite, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #1a202c !important;
    }

    /* Ensure tab content is visible */
    .stTabs .stTabPanel {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel p, .stTabs .stTabPanel div, .stTabs .stTabPanel span {
        color: #1a202c !important;
    }

    /* Custom card styling */
    .custom-card {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        color: #1a202c !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }

    .summary-card {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        color: #1a202c !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        font-weight: 400 !important;
    }

    .question-card {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        color: #1a202c !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        font-weight: 400 !important;
    }

    .feedback-card {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        color: #1a202c !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        font-weight: 400 !important;
    }

    .metric-container {
        background: #ffffff;
        border: 2px solid #d1d5db;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a202c !important;
        margin: 0;
    }

    .metric-label {
        font-size: 1rem;
        color: #374151 !important;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    .status-success {
        color: #16a34a !important;
        font-weight: 600;
        padding: 0.5rem 0;
        font-size: 16px !important;
    }

    .status-error {
        color: #dc2626 !important;
        font-weight: 600;
        padding: 0.5rem 0;
        font-size: 16px !important;
    }

    /* Welcome section with dark background for white title */
    .welcome-container {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        border-radius: 12px;
        padding: 3rem 2rem;
        margin: 2rem 0;
        text-align: center;
        border: 2px solid #1d4ed8;
    }

    .welcome-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .welcome-subtitle {
        font-size: 1.2rem;
        color: #e2e8f0 !important;
        margin-bottom: 2rem;
        line-height: 1.6;
        font-weight: 400;
    }

    .feature-item {
        background: #ffffff;
        border: 2px solid #d1d5db;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        text-align: left;
    }

    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }

    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a202c !important;
        margin-bottom: 0.5rem;
    }

    .feature-description {
        color: #374151 !important;
        line-height: 1.5;
        font-size: 1rem;
    }

    /* Button styling */
    .stButton > button {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        background: #1d4ed8 !important;
        transform: translateY(-1px) !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 1rem 0;
        font-weight: 600;
        color: #374151 !important;
        font-size: 1.1rem !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb !important;
        border-bottom-color: #2563eb;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 6px !important;
        border: 2px solid #d1d5db !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
        color: #1a202c !important;
    }

    .stTextArea > div > div > textarea {
        border-radius: 6px !important;
        border: 2px solid #d1d5db !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
        color: #1a202c !important;
    }

    /* Enhanced sidebar styling with high contrast */
    .sidebar .sidebar-content {
        background: #f9fafb;
        color: #1a202c !important;
    }

    /* Sidebar text visibility boost */
    .sidebar h1, .sidebar h2, .sidebar h3, .sidebar h4, .sidebar h5, .sidebar h6 {
        color: #1a202c !important;
        font-weight: 700 !important;
    }

    .sidebar p, .sidebar div, .sidebar span, .sidebar label {
        color: #1a202c !important;
        font-weight: 500 !important;
    }

    .sidebar .stMarkdown {
        color: #1a202c !important;
    }

    .sidebar .stFileUploader label {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    .sidebar .stFileUploader div {
        color: #1a202c !important;
    }

    /* Section headers with high contrast */
    h1, h2, h3, h4, h5, h6 {
        color: #1a202c !important;
        font-weight: 700 !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #1a202c !important;
        font-size: 1.1rem !important;
    }

    .streamlit-expanderContent {
        color: #1a202c !important;
    }

    .streamlit-expanderContent p, .streamlit-expanderContent div, .streamlit-expanderContent span {
        color: #1a202c !important;
    }

    /* Override any low contrast text everywhere */
    p, span, div, label, .stText {
        color: #1a202c !important;
    }

    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        font-weight: 500 !important;
        font-size: 1rem !important;
    }

    .stSuccess > div, .stError > div, .stWarning > div, .stInfo > div {
        color: #1a202c !important;
    }

    /* File uploader styling */
    .stFileUploader label {
        color: #1a202c !important;
        font-weight: 600 !important;
    }

    /* Ensure all text areas and inputs have proper contrast */
    .stTextInput label, .stTextArea label, .stSelectbox label {
        color: #1a202c !important;
        font-weight: 600 !important;
    }

    /* Force visibility for all interactive elements */
    .stButton, .stDownloadButton, .stFileUploader, .stTextInput, .stTextArea, .stSelectbox {
        color: #1a202c !important;
    }

    /* AGGRESSIVE TAB CONTENT VISIBILITY FIXES */
    .stTabs .stTabPanel {
        color: #1a202c !important;
        background: #ffffff !important;
        padding: 1rem !important;
    }

    .stTabs .stTabPanel > div {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel * {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel .stMarkdown {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel .stMarkdown * {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel .stWrite {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel .stWrite * {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel h1, .stTabs .stTabPanel h2, .stTabs .stTabPanel h3, 
    .stTabs .stTabPanel h4, .stTabs .stTabPanel h5, .stTabs .stTabPanel h6 {
        color: #1a202c !important;
        font-weight: 700 !important;
    }

    .stTabs .stTabPanel p, .stTabs .stTabPanel div, .stTabs .stTabPanel span {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel .stTextInput, .stTabs .stTabPanel .stTextArea, 
    .stTabs .stTabPanel .stButton, .stTabs .stTabPanel .stSelectbox {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel .stTextInput label, .stTabs .stTabPanel .stTextArea label {
        color: #1a202c !important;
        font-weight: 600 !important;
    }

    .stTabs .stTabPanel .stExpander {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel .stExpander * {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel .stSuccess, .stTabs .stTabPanel .stError, 
    .stTabs .stTabPanel .stWarning, .stTabs .stTabPanel .stInfo {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel .stSuccess *, .stTabs .stTabPanel .stError *, 
    .stTabs .stTabPanel .stWarning *, .stTabs .stTabPanel .stInfo * {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel .stContainer {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel .stContainer * {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel .stColumns {
        color: #1a202c !important;
    }

    .stTabs .stTabPanel .stColumns * {
        color: #1a202c !important;
    }

    /* Ensure spinners and other elements are visible */
    .stSpinner > div {
        color: #1a202c !important;
    }

    /* NUCLEAR OPTION - Force ALL elements to be white */
    * {
        color: #ffffff !important;
    }

    /* But keep specific elements with their intended colors */
    .welcome-title {
        color: #ffffff !important;
    }

    .welcome-subtitle {
        color: #e2e8f0 !important;
    }

    .metric-value {
        color: #ffffff !important;
    }

    .metric-label {
        color: #ffffff !important;
    }

    .feature-description {
        color: #ffffff !important;
    }

    .stButton > button {
        color: white !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #374151 !important;
    }

    .status-success {
        color: #16a34a !important;
    }

    .status-error {
        color: #dc2626 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📄 Document Reasoning Assistant")

# Initialize session state
session_keys = {
    "challenge_questions": [],
    "chunks": [],
    "summary": "",
    "sections": [],
    "document_processed": False,
    "challenge_evaluations": {}
}

for key, default in session_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a document file",
        type=['pdf', 'txt'],
        help="Upload a PDF or TXT file for analysis"
    )

    if uploaded_file:
        # Save uploaded file
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✓ {uploaded_file.name} uploaded successfully!")

        # Process document button
        if st.button("🔄 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    # Send file to backend
                    with open(file_path, "rb") as f:
                        files = {"file": f}
                        response = requests.post(f"{BACKEND_URL}/process", files=files)

                    if response.status_code == 200:
                        doc_data = response.json()
                        st.session_state["chunks"] = doc_data["chunks"]
                        st.session_state["sections"] = doc_data["sections"]
                        st.session_state["summary"] = doc_data["summary"]
                        st.session_state["document_processed"] = True
                        st.session_state["challenge_questions"] = []
                        st.session_state["challenge_evaluations"] = {}

                        st.success("✓ Document processed successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Processing failed: {response.json().get('error', 'Unknown error')}")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Main content area
if st.session_state["document_processed"]:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📄 Summary", "❓ Ask Anything", "🧠 Challenge Me"])

    with tab1:
        st.header("Document Summary")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Summary in a container
            with st.container():
                st.markdown(f'<div class="summary-card">{st.session_state["summary"]}</div>', unsafe_allow_html=True)

        with col2:
            # Metrics in containers
            with st.container():
                st.markdown(f'''
                <div class="metric-container">
                    <div class="metric-value">{len(st.session_state["sections"])}</div>
                    <div class="metric-label">Total Sections</div>
                </div>
                ''', unsafe_allow_html=True)

                st.markdown(f'''
                <div class="metric-container">
                    <div class="metric-value">{len(st.session_state["chunks"])}</div>
                    <div class="metric-label">Total Chunks</div>
                </div>
                ''', unsafe_allow_html=True)

        # Section details
        with st.expander("📚 View Document Sections"):
            for i, section in enumerate(st.session_state["sections"], 1):
                st.write(f"**{i}.** {section}")

    with tab2:
        st.header("Ask Anything")
        st.write("Ask questions about the document and get contextual answers with source references.")

        # Question input
        query = st.text_input("💭 Enter your question:", placeholder="What is the main topic of this document?")

        if st.button("🔍 Get Answer", type="primary") and query:
            with st.spinner("Generating answer..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/ask", json={"query": query})

                    if response.status_code == 200:
                        result = response.json()

                        # Display answer
                        st.subheader("💡 Answer")
                        st.markdown(f'<div class="custom-card">{result["answer"]}</div>', unsafe_allow_html=True)

                        # Display sources
                        if result.get("sources"):
                            st.subheader("📚 Supporting Sources")
                            for i, source in enumerate(result["sources"], 1):
                                section = source["metadata"].get("section", "Unknown")
                                paragraph = source["metadata"].get("paragraph", "?")
                                content = source["page_content"]

                                with st.expander(f"Source {i}: {section} (Paragraph {paragraph})"):
                                    st.write(content)
                    else:
                        st.error(f"❌ Error: {response.json().get('error', 'Unknown error')}")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    with tab3:
        st.header("Challenge Me")
        st.write("Test your understanding with AI-generated comprehension questions.")

        # Generate questions if not already done
        if not st.session_state["challenge_questions"]:
            if st.button("🎯 Generate Challenge Questions", type="primary"):
                with st.spinner("Generating logic-based questions..."):
                    try:
                        response = requests.post(f"{BACKEND_URL}/challenge/init", json={
                            "chunks": st.session_state["chunks"]
                        })

                        if response.status_code == 200:
                            st.session_state["challenge_questions"] = response.json()["questions"]
                            st.rerun()
                        else:
                            st.error(f"❌ Question generation failed: {response.json().get('error', 'Unknown error')}")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        # Display questions and handle answers
        if st.session_state["challenge_questions"]:
            st.success("✅ Challenge questions generated successfully!")

            for idx, question in enumerate(st.session_state["challenge_questions"]):
                st.markdown(f'<div class="question-card"><strong>Question {idx + 1}:</strong><br><br>{question}</div>',
                            unsafe_allow_html=True)

                # Answer input
                answer_key = f"answer_{idx}"
                user_answer = st.text_area(
                    f"Your answer to Question {idx + 1}:",
                    key=answer_key,
                    height=100,
                    placeholder="Type your answer here..."
                )

                # Evaluate button
                if st.button(f"📝 Evaluate Answer {idx + 1}", key=f"eval_{idx}"):
                    if user_answer.strip():
                        with st.spinner("Evaluating your answer..."):
                            try:
                                response = requests.post(f"{BACKEND_URL}/challenge/evaluate", json={
                                    "question": question,
                                    "answer": user_answer
                                })

                                if response.status_code == 200:
                                    feedback = response.json()["feedback"]
                                    st.session_state["challenge_evaluations"][idx] = feedback
                                    st.markdown(
                                        f'<div class="feedback-card"><strong>Feedback:</strong><br><br>{feedback}</div>',
                                        unsafe_allow_html=True)
                                else:
                                    st.error(f"❌ Evaluation failed: {response.json().get('error', 'Unknown error')}")

                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                    else:
                        st.warning("⚠️ Please provide an answer before evaluation.")

                # Show previous evaluation if exists
                if idx in st.session_state["challenge_evaluations"]:
                    st.markdown(
                        f'<div class="feedback-card"><strong>Previous Feedback:</strong><br><br>{st.session_state["challenge_evaluations"][idx]}</div>',
                        unsafe_allow_html=True)

                st.markdown("---")

else:
    # Welcome message
    st.markdown('''
    <div class="welcome-container">
        <div class="welcome-title">Welcome to Document Reasoning Assistant</div>
        <div class="welcome-subtitle">
            Upload your documents and unlock intelligent analysis with AI-powered insights
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Features in columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('''
        <div class="feature-item">
            <div class="feature-icon">📄</div>
            <div class="feature-title">Document Upload</div>
            <div class="feature-description">Support for PDF and TXT files with intelligent processing</div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown('''
        <div class="feature-item">
            <div class="feature-icon">❓</div>
            <div class="feature-title">Ask Anything</div>
            <div class="feature-description">Query your document with contextual answers</div>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown('''
        <div class="feature-item">
            <div class="feature-icon">📝</div>
            <div class="feature-title">Auto Summary</div>
            <div class="feature-description">Get concise 150-word summaries instantly</div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown('''
        <div class="feature-item">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Challenge Mode</div>
            <div class="feature-description">Test comprehension with AI-generated questions</div>
        </div>
        ''', unsafe_allow_html=True)

    # Instructions
    st.markdown('''
    <div class="custom-card">
        <h4>How to get started:</h4>
        <ol>
            <li>Upload a document using the sidebar</li>
            <li>Click "Process Document" to analyze it</li>
            <li>Explore the three interaction modes in the tabs above</li>
        </ol>
    </div>
    ''', unsafe_allow_html=True)

    # Backend status check
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ Backend server is running")
        else:
            st.error("❌ Backend server is not responding properly")
    except:
        st.error("❌ Backend server is not running. Please start the Flask server first.")
