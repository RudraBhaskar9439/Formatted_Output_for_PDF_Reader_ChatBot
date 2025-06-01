from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import google.generativeai as genai
import os
from typing import List, Dict
from dataclasses import dataclass
import json

@dataclass
class QuestionResponse:
    correct_option: str
    explanation: str
    source: str


class ChunkWithSource:
    def __init__(self, text: str, source: str):
        self.text = text
        self.source = source

def process_pdfs(folder_path: str) -> List[ChunkWithSource]:
    chunks_with_sources = []
    pdf_files = get_pdf_files(folder_path)
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(text)
        
        for chunk in chunks:
            chunks_with_sources.append(ChunkWithSource(chunk, filename))
    
    return chunks_with_sources



def get_pdf_files(folder_path: str) -> List[str]:
    """Get all PDF files from the specified folder."""
    pdf_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.pdf'):
            pdf_files.append(os.path.join(folder_path, file))
    return pdf_files

class ConversationMemory:
    def __init__(self, max_history: int = 5):
        self.history: List[Dict] = []
        self.max_history = max_history

    def add_interaction(self, query: str, response: str, context: str):
        self.history.append({
            "query": query,
            "response": response,
            "context": context
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_formatted_history(self) -> str:
        formatted = ""
        for interaction in self.history:
            formatted += f"Question: {interaction['query']}\n"
            formatted += f"Answer: {interaction['response']}\n"
        return formatted

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
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
    embeddings = model.encode(sentences)
    return embeddings

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
if __name__ == "__main__":
    # Specify your PDFs folder path
    folder_path = "/Users/rudrabhaskar/Desktop/ML-main/ML/Folder_with_pdfs"
    
    # Process all PDFs with source tracking
    chunks_with_sources = process_pdfs(folder_path)
    if not chunks_with_sources:
        print("No PDF files found in the specified folder!")
        exit()
    
    # Extract just the text for embeddings
    all_chunks = [chunk.text for chunk in chunks_with_sources]
    print(f"Total chunks created: {len(all_chunks)}")
    
    # Create embeddings for all chunks
    chunk_vectors = sentence_encode(all_chunks)
    
    # Initialize conversation memory
    memory = ConversationMemory()

    while True:
        # Get user input
        query = input("\nEnter your question (or 'quit' to exit): ")
        
        if query.lower() == 'quit':
            break
            
        query_vector = sentence_encode([query])
        top_k = 3
        
        similarities = []
        for idx, chunk_vec in enumerate(chunk_vectors):
            sim = cosine_similarity(chunk_vec, query_vector[0])
            similarities.append((sim, idx))
        
        print("Similarities:", similarities)

        print("==" * 20)

        # Sort by similarity descending and get top_k indices
        top_chunks = sorted(similarities, reverse=True)[:top_k]
        top_indices = [idx for _, idx in top_chunks]

        print("Top chunk indices:", top_indices)

        new_context = ""
        for i in top_indices:
            new_context += all_chunks[i] + "\n"

        GOOGLE_API_KEY = "AIzaSyBqqT1xFUv4iMDViJ8dIJHlD_kM7_T0fE4"

          # Create history-aware prompt
        conversation_history = memory.get_formatted_history()
        prompt_template = f"""You are an assistant that provides answers in JSON format.

Current Context (from {chunks_with_sources[top_indices[0]].source}):
{new_context}

Question: {query}

Respond ONLY with a JSON object in this exact format:
{{
    "correct_option": "The correct answer",
    "explanation": "Detailed explanation of why this is correct",
    "source": "{chunks_with_sources[top_indices[0]].source}"
}}"""

        try:
            # Configure the API
            genai.configure(api_key=GOOGLE_API_KEY)
            
            # Initialize the model correctly
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Generate response with the actual prompt
            response = model.generate_content(prompt_template)
            
            # Parse JSON response
            try:
                json_response = json.loads(response.text)
                print("\nFormatted Response:")
                print(json.dumps(json_response, indent=2))
                
                # Store interaction in memory
                memory.add_interaction(query, json.dumps(json_response), new_context)
                
            except json.JSONDecodeError as je:
                print(f"Error parsing JSON response: {je}")
                print("Raw response:", response.text)
                
        except Exception as e:
            print(f"Error generating response: {str(e)}")