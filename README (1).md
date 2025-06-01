# Formatted Ouput from Chatting with Multiple PDFs using Gemini AI
A Python-based application that enables intelligent conversation with PDF documents using Google's Gemini AI and semantic search.

# Features
PDF text extraction and processing
Semantic chunking and embeddings
Intelligent context retrieval using cosine similarity
Natural language interaction with documents
Formats the Output
Integration with Google's Gemini AI model
Prerequisites
Python 3.8+
Google Gemini API key
# Installation
Clone the repository:
```python
git clone <your-repo-url>
cd <your-repo-directory>
```
Install required packages:
```python
git clone <your-repo-url>
cd <your-repo-directory>
```
Create a .env file in the project root:
```python
git clone <your-repo-url>
cd <your-repo-directory>
```
Usage
1.Place your PDF file in the project directory

2.Run the script:
```python
git clone <your-repo-url>
cd <your-repo-directory>
```
3.Enter your questions when prompted. Type 'quit' to exit.
Example interaction:
```python
git clone <your-repo-url>
cd <your-repo-directory>
```
How It Works

1.PDF Processing:

Extracts text from PDF documents
Splits text into manageable chunks
Maintains context between segments

2.Semantic Search:

Uses SentenceTransformers for text embeddings
Implements cosine similarity for relevant context retrieval
Selects top-k most relevant chunks

3.AI Integration:

Leverages Google's Gemini AI model
Provides context-aware responses
Maintains conversation coherence

# Technical Components

1.sentence-transformers/all-MiniLM-L6-v2 for embeddings
2.RecursiveCharacterTextSplitter for text chunking
3.Google Gemini AI for natural language understanding
4.PyMuPDF for PDF processing

# Configuration
Customize these parameters in the code:

1.chunk_size: Size of text chunks (default: 1000)
2.chunk_overlap: Overlap between chunks (default: 200)
3.top_k: Number of relevant chunks to consider (default: 3)

# Error Handling
The application includes comprehensive error handling for:

*Missing API keys
*PDF processing errors
*Model generation issues
*Invalid inputs


# MIT License

Copyright (c) 2024 [Rudra Bhaskar]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# Author
[Rudra Bhaskar]

