# PaperMind AI — Mini Project Report Content

## 1. Title
PaperMind AI: Intelligent Research Paper Analyzer with Semantic Search

## 2. Abstract
PaperMind AI is an AI-assisted application for analyzing research papers and text-based PDF documents. It extracts text and page numbers, splits the content into overlapping chunks, and creates semantic embeddings using a pretrained sentence-transformer model. When a user asks a question, the system compares the query embedding with document chunk embeddings using cosine similarity and retrieves the most relevant passages. An optional language-model API can generate an answer using only the retrieved context. The application also offers a lightweight extractive summary and displays source page references. The system aims to make academic information retrieval faster and more accessible.

## 3. Problem Statement
Research papers and technical documents contain extensive information, making manual searching time-consuming. Keyword-based search can fail when the query uses different wording from the source. A system is therefore needed to extract PDF text, retrieve information according to semantic meaning, and present relevant passages with page references. This project develops such a system using Python, NLP, embeddings, semantic search, and optional Retrieval-Augmented Generation.

## 4. Objectives
1. Upload and process text-based PDF files.
2. Extract text and preserve page references.
3. Split document text into overlapping chunks.
4. Create embeddings for semantic representation.
5. Retrieve relevant passages for natural-language questions.
6. Optionally generate context-grounded answers using an LLM.
7. Provide a quick extractive summary and source verification.
8. Evaluate retrieval relevance and application usability.

## 5. Existing System
Users commonly read a complete document or use keyword search. This can take time and may miss semantically related passages. Manual source verification is also required.

## 6. Proposed System
The proposed system extracts text from an uploaded PDF, creates overlapping chunks, generates embeddings, ranks chunks by cosine similarity, and displays relevant passages with page references. If an API key is configured, an LLM can formulate a concise answer from the retrieved context.

## 7. System Requirements
### Hardware
- Computer with 8 GB RAM recommended
- Internet access for initial embedding-model download
- Additional GPU not required, though it may improve speed

### Software
- Python 3.10 or 3.11 recommended
- Streamlit
- PyMuPDF
- Sentence Transformers
- scikit-learn
- NumPy
- python-dotenv
- Optional Google Gemini API

## 8. Modules
1. PDF Upload Module
2. Text Extraction Module
3. Text Chunking Module
4. Embedding Generation Module
5. Semantic Retrieval Module
6. Answer Generation Module (optional)
7. Summary Module
8. Source Display and Download Module

## 9. Algorithm
1. Start.
2. Upload a text-based PDF.
3. Extract text from each page.
4. Split the text into overlapping chunks.
5. Generate an embedding for each chunk.
6. Accept a natural-language query.
7. Generate an embedding for the query.
8. Calculate cosine similarity between the query and all chunk embeddings.
9. Sort chunks by similarity and select the top-k results.
10. Display relevant passages and page numbers.
11. If an LLM is configured, generate an answer using retrieved passages as context.
12. Verify the answer against the original PDF.
13. Stop.

## 10. Evaluation
Use a small set of questions with known answers from the selected PDFs. For each question, manually check whether the top retrieved passages support the answer. Record top-1 and top-3 relevance, response time, and whether the answer is supported by its sources. Do not report unmeasured accuracy values.

## 11. Advantages
- Reduces manual searching time
- Searches by meaning, not only exact keywords
- Displays page references for source verification
- Can work without a paid answer-generation API
- Easy to use through a web interface

## 12. Limitations
- Scanned PDFs require OCR
- Tables, equations, and complex layouts may not extract perfectly
- Semantic retrieval can return irrelevant passages
- Generated answers may still be incorrect and must be verified
- Initial model download requires internet access

## 13. Future Scope
- OCR for scanned documents
- Search across multiple PDFs
- Export answers and citations into a report
- Persistent vector database
- Improved citation highlighting
- Support for additional languages

## 14. Conclusion
PaperMind AI demonstrates how semantic embeddings and retrieval can improve access to information in long PDF documents. It combines PDF text extraction, chunking, semantic search, source references, and optional context-grounded answer generation in a Python web application. Its results should always be checked against the original source.
