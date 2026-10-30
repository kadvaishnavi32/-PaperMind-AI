# PaperMind AI — AI Mini Project

**Title:** PaperMind AI: Intelligent Research Paper Analyzer with Semantic Search

## Problem statement
Research papers and technical PDFs can be long and difficult to search manually. Keyword search may also miss relevant information when a question uses different words from the document. This project extracts PDF text, creates semantic embeddings, retrieves relevant passages, and can optionally generate answers grounded in retrieved sources.

## Features
- Upload a text-based PDF
- Extract text with page numbers
- Create overlapping text chunks
- Semantic search using sentence embeddings and cosine similarity
- Ask questions and view relevant source passages
- Optional Gemini-generated answers grounded in retrieved passages
- Lightweight extractive summary
- Download summary as a text file

## Technology stack
- Python
- Streamlit
- PyMuPDF
- Sentence Transformers (`all-MiniLM-L6-v2`)
- scikit-learn and NumPy
- Optional Google Gemini API via `google-genai`

## Project structure
```text
PaperMind_AI_Mini_Project/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Run locally

### 1. Install Python
Use Python 3.10 or 3.11 if possible.

### 2. Open terminal in this folder and create a virtual environment
Windows:
```bash
python -m venv .venv
.venv\\Scripts\\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Optional: enable generated answers
Copy `.env.example` to `.env` and add your Google Gemini API key:
```text
GOOGLE_API_KEY=your_api_key_here
```
Do not share or commit `.env`. If no key is configured, semantic search still works and the app displays relevant source passages.

### 5. Run the app
```bash
streamlit run app.py
```
Streamlit will print a local URL, usually `http://localhost:8501`.

## How the AI works
1. Extract PDF text and page numbers.
2. Split text into overlapping chunks.
3. Encode each chunk into a dense vector using `all-MiniLM-L6-v2`.
4. Encode the user's query with the same model.
5. Rank chunks by cosine similarity.
6. Display the best matching passages and source pages.
7. If an API key is configured, pass the retrieved context to Gemini with instructions to answer only from that context.

## Testing checklist
- Upload a text-based PDF.
- Ask a question whose answer is clearly present.
- Verify that at least one returned source passage contains supporting information.
- Ask an unrelated question and confirm that the retrieved passages do not provide a reliable answer.
- Test a PDF with no selectable text and confirm that the app explains the OCR limitation.
- Compare answers with the original PDF before using them in a report.

## Limitations and future enhancements
- OCR for scanned PDFs
- Multi-PDF search
- Persistent vector index
- Citation highlighting
- Better abstractive summaries
- Downloadable report
- Authentication and user-specific document history

## Academic note
This is an educational prototype, not a guarantee of factual correctness. Semantic similarity is not a confidence probability. Generated answers must be verified against the cited source passages.
