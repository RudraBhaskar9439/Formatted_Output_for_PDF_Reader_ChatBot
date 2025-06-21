import streamlit as st
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import google.generativeai as genai
import json
from dataclasses import dataclass
from typing import List, Dict
from io import BytesIO

# ----- Data Classes -----
@dataclass
class QuestionResponse:
    correct_option: str
    explanation: str
    source: str

class ChunkWithSource:
    def __init__(self, text: str, source: str):
        self.text = text
        self.source = source

# ----- Utilities -----
def extract_text_from_pdf_bytes(file_bytes):
    text = ""
    with fitz.open("pdf", file_bytes) as doc:
        for page in doc:
            text += page.get_text()
    return text

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_text(text)

def sentence_encode(sentences):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(sentences)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----- Conversation Memory -----
class ConversationMemory:
    def __init__(self, max_history: int = 5):
        self.history: List[Dict] = []
        self.max_history = max_history

    def add_interaction(self, query: str, response: str, context: str):
        self.history.append({"query": query, "response": response, "context": context})
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_formatted_history(self) -> str:
        return "\n".join(
            f"Question: {h['query']}\nAnswer: {h['response']}\n"
            for h in self.history
        )

# ----- Streamlit App -----
st.set_page_config(page_title="📘 Gemini PDF QA", layout="wide")
st.title("📘 Gemini PDF QA System")
st.markdown("Upload PDFs and ask questions to get structured answers in JSON format.")

uploaded_files = st.file_uploader("📂 Upload one or more PDF files", type="pdf", accept_multiple_files=True)

# Initialize session state
if "chunks_with_sources" not in st.session_state:
    st.session_state.chunks_with_sources = []
    st.session_state.chunk_vectors = []
    st.session_state.text_chunks = []
    st.session_state.memory = ConversationMemory()

# Process uploaded PDFs
if uploaded_files and st.button("📄 Process PDFs"):
    chunks = []
    for file in uploaded_files:
        file_bytes = file.read()
        text = extract_text_from_pdf_bytes(file_bytes)
        for chunk in split_text_into_chunks(text):
            chunks.append(ChunkWithSource(chunk, file.name))

    st.session_state.chunks_with_sources = chunks
    st.session_state.text_chunks = [c.text for c in chunks]
    st.session_state.chunk_vectors = sentence_encode(st.session_state.text_chunks)
    st.success(f"✅ Loaded {len(uploaded_files)} PDFs and created {len(chunks)} chunks.")

# Ask a question
query = st.text_input("❓ Ask your question")
generate = st.button("🧠 Generate Answer")

if generate and query:
    if "chunk_vectors" not in st.session_state or len(st.session_state.chunk_vectors) == 0:
        st.error("⚠️ Please upload and process PDFs first.")
    else:
        query_vector = sentence_encode([query])[0]
        similarities = [
            (cosine_similarity(v, query_vector), idx)
            for idx, v in enumerate(st.session_state.chunk_vectors)
        ]
        top_k = sorted(similarities, reverse=True)[:3]
        top_indices = [idx for _, idx in top_k]
        new_context = "\n".join(st.session_state.text_chunks[i] for i in top_indices)

        # Gemini API key
        GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')

        prompt = f"""You are an assistant that provides answers in JSON format.

Current Context (from {st.session_state.chunks_with_sources[top_indices[0]].source}):
{new_context}

Question: {query}

Respond ONLY with a JSON object in this exact format:
{{
    "correct_option": "The correct answer",
    "explanation": "Detailed explanation of why this is correct",
    "source": "{st.session_state.chunks_with_sources[top_indices[0]].source}"
}}"""

        try:
            response = model.generate_content(prompt)
            json_response = json.loads(response.text)

            st.subheader("📦 JSON Response")
            st.json(json_response)

            st.session_state.memory.add_interaction(query, json.dumps(json_response), new_context)
        except json.JSONDecodeError:
            st.error("❌ Failed to parse JSON response.")
            st.text(response.text)
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")

# Show conversation history
with st.expander("📜 Conversation History"):
    st.text(st.session_state.memory.get_formatted_history())
