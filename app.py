
import os
import re
import hashlib
import streamlit as st
import fitz
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from google import genai

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="PaperMind AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PaperMind AI")
st.subheader(
    "Intelligent Research Paper Analyzer with Semantic Search"
)

st.write(
    "Upload a research paper in PDF format, "
    "ask questions, search relevant information "
    "and generate a concise summary."
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("Project Information")

    st.markdown("### AI Techniques")
    st.markdown("""
    - Text Embeddings
    - Semantic Search
    - Cosine Similarity
    - Retrieval-Augmented Generation (RAG)
    - TF-IDF Summarization
    """)

    st.info(
        "The first launch downloads the sentence "
        "embedding model. Internet access is required."
    )

    top_k = st.slider(
        "Number of source passages",
        min_value=1,
        max_value=8,
        value=4
    )

    st.markdown("---")
    st.caption("PaperMind AI | Python Mini Project")


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_pdf_text(pdf_file):

    pages = []

    pdf_bytes = pdf_file.getvalue()
    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text")

        if text and text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    document.close()
    return pages


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def create_chunks(pages, chunk_size=180, overlap=35):

    chunks = []

    for page_data in pages:

        words = page_data["text"].split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(words[start:end])

            if len(chunk_text.strip()) > 30:
                chunks.append({
                    "text": chunk_text,
                    "page": page_data["page"]
                })

            start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

def create_embeddings(chunks, model):

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return embeddings


# --------------------------------------------------
# SEMANTIC SEARCH
# --------------------------------------------------

def search_relevant_passages(
    question,
    chunks,
    embeddings,
    model,
    number_of_results
):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    scores = cosine_similarity(
        question_embedding,
        embeddings
    )[0]

    ranked_indices = np.argsort(scores)[::-1]

    results = []

    for index in ranked_indices[:number_of_results]:

        results.append({
            "text": chunks[index]["text"],
            "page": chunks[index]["page"],
            "score": float(scores[index])
        })

    return results


# --------------------------------------------------
# EXTRACTIVE SUMMARIZATION
# --------------------------------------------------

def generate_summary(pages, sentence_count=8):

    full_text = " ".join(
        page["text"] for page in pages
    )

    sentences = re.split(
        r'(?<=[.!?])\s+',
        full_text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 30
    ]

    if not sentences:
        return "No readable text found in this PDF."

    if len(sentences) <= sentence_count:
        return " ".join(sentences)

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(sentences)

        document_vector = matrix.mean(axis=0)

        scores = cosine_similarity(
            matrix,
            np.asarray(document_vector)
        ).flatten()

        selected_indices = np.argsort(scores)[
            -sentence_count:
        ]

        selected_indices = sorted(selected_indices)

        summary = [
            sentences[index]
            for index in selected_indices
        ]

        return " ".join(summary)

    except ValueError:

        return " ".join(sentences[:sentence_count])


# --------------------------------------------------
# GEMINI AI ANSWER GENERATION
# --------------------------------------------------

def generate_ai_answer(question, results):

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return None

    context = "\n\n".join(
        f"PDF Page {item['page']}: {item['text']}"
        for item in results
    )

    prompt = f"""
You are PaperMind AI, an assistant for analyzing
research papers.

Answer the user's question using ONLY the provided
PDF passages.

IMPORTANT RULES:

1. Answer the exact question directly.
2. For questions about the main objective, explain
   the overall purpose of the research or system.
3. Do not confuse the main objective with technical
   implementation steps.
4. Do not list all objectives unless specifically asked.
5. Use simple and clear English.
6. Keep normal answers between 3 and 6 sentences.
7. Use bullet points only when necessary.
8. Do not invent information.
9. If the PDF does not contain the answer, clearly say
   that the information is not available in the provided PDF.
10. Mention the relevant PDF page number when possible.

USER QUESTION:
{question}

PDF CONTEXT:
{context}

ANSWER:
"""

    try:

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return response.text

    except Exception as error:

        st.warning(
            "Gemini AI could not generate an answer. "
            "Relevant PDF passages are shown below."
        )

        st.caption(f"AI service message: {error}")

        return None


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

st.markdown("---")
st.header("📄 Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is None:

    st.info(
        "Please upload a PDF research paper to start."
    )

    st.markdown("### How PaperMind AI works")

    st.markdown("""
    1. Upload a research paper.
    2. Extract text from PDF pages.
    3. Divide text into overlapping chunks.
    4. Generate text embeddings.
    5. Search relevant passages using cosine similarity.
    6. Generate answers using retrieved PDF context.
    """)

    st.stop()


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

file_hash = hashlib.md5(
    uploaded_file.getvalue()
).hexdigest()

if st.session_state.get("file_hash") != file_hash:

    with st.spinner("Extracting PDF text..."):

        try:

            pages = extract_pdf_text(uploaded_file)

            if not pages:
                st.error(
                    "No readable text found. "
                    "This PDF may be scanned."
                )
                st.stop()

            st.session_state["pages"] = pages
            st.session_state["file_hash"] = file_hash
            st.session_state.pop("chunks", None)
            st.session_state.pop("embeddings", None)

        except Exception as error:

            st.error(f"PDF extraction failed: {error}")
            st.stop()


pages = st.session_state["pages"]

st.success(
    f"PDF uploaded successfully! Total pages with readable text: {len(pages)}"
)


# --------------------------------------------------
# LOAD MODEL AND PROCESS CHUNKS
# --------------------------------------------------

if "chunks" not in st.session_state:

    with st.spinner(
        "Loading AI embedding model and processing document..."
    ):

        try:

            model = load_embedding_model()

            chunks = create_chunks(pages)

            if not chunks:
                st.error("Could not create text chunks.")
                st.stop()

            embeddings = create_embeddings(
                chunks,
                model
            )

            st.session_state["chunks"] = chunks
            st.session_state["embeddings"] = embeddings

        except Exception as error:

            st.error(
                f"Embedding model processing failed: {error}"
            )
            st.stop()


chunks = st.session_state["chunks"]
embeddings = st.session_state["embeddings"]

st.success(
    f"Document processed successfully! "
    f"{len(chunks)} text chunks created."
)


# --------------------------------------------------
# DOCUMENT INFORMATION
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.metric("PDF Pages", len(pages))

with col2:
    st.metric("Text Chunks", len(chunks))


# --------------------------------------------------
# APPLICATION TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📖 PDF Summary",
    "🤖 Ask the PDF",
    "🔍 Semantic Search",
    "ℹ️ About Project"
])


# --------------------------------------------------
# TAB 1: SUMMARY
# --------------------------------------------------

with tab1:

    st.header("📖 Research Paper Summary")

    if st.button("Generate Summary"):

        with st.spinner("Generating summary..."):

            summary = generate_summary(pages)

            st.session_state["summary"] = summary

    if "summary" in st.session_state:

        st.markdown("### Extractive Summary")

        st.write(st.session_state["summary"])

        st.caption(
            "This summary selects important sentences "
            "from the original PDF using TF-IDF."
        )


# --------------------------------------------------
# TAB 2: ASK THE PDF
# --------------------------------------------------

with tab2:

    st.header("🤖 Ask Questions About Your PDF")

    question = st.text_input(
        "Enter your question",
        placeholder="What is the main objective of this paper?"
    )

    if st.button("Find Answer"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            model = load_embedding_model()

            with st.spinner(
                "Searching relevant passages..."
            ):

                results = search_relevant_passages(
                    question,
                    chunks,
                    embeddings,
                    model,
                    top_k
                )

            st.markdown("## AI-generated answer")

            answer = generate_ai_answer(
                question,
                results
            )

            if answer:

                st.write(answer)

            else:

                st.info(
                    "AI-generated answer is unavailable. "
                    "Please read the relevant PDF passages below."
                )

            st.markdown("## Relevant Source Passages")

            for index, item in enumerate(results, start=1):

                with st.expander(
                    f"Source {index} — PDF page "
                    f"{item['page']} — similarity "
                    f"{item['score']:.3f}"
                ):

                    st.write(item["text"])


# --------------------------------------------------
# TAB 3: SEMANTIC SEARCH
# --------------------------------------------------

with tab3:

    st.header("🔍 Semantic Search")

    search_query = st.text_input(
        "Enter a topic or concept to search"
    )

    if st.button("Search Document"):

        if not search_query.strip():

            st.warning("Please enter a search query.")

        else:

            model = load_embedding_model()

            search_results = search_relevant_passages(
                search_query,
                chunks,
                embeddings,
                model,
                top_k
            )

            for index, item in enumerate(
                search_results,
                start=1
            ):

                st.markdown(
                    f"### Result {index} "
                    f"(PDF Page {item['page']})"
                )

                st.write(item["text"])

                st.caption(
                    f"Cosine similarity: {item['score']:.3f}"
                )

                st.divider()


# --------------------------------------------------
# TAB 4: ABOUT PROJECT
# --------------------------------------------------

with tab4:

    st.header("About PaperMind AI")

    st.write("""
    PaperMind AI is an intelligent research paper
    analyzer that helps users understand PDF documents
    through natural-language questions.

    The system extracts text from PDF documents,
    creates overlapping text chunks, converts them
    into embeddings and uses semantic search to find
    relevant information.

    Cosine similarity is used to rank the passages.
    An optional Gemini language model generates
    concise answers using the retrieved passages.
    """)

    st.markdown("### Technologies Used")

    st.markdown("""
    - Python
    - Streamlit
    - PyMuPDF
    - Sentence Transformers
    - Scikit-learn
    - NumPy
    - Google Gemini API
    - Retrieval-Augmented Generation (RAG)
    """)

    st.markdown("### Main Algorithms")

    st.markdown("""
    1. Text Embedding
    2. Cosine Similarity
    3. TF-IDF Extractive Summarization
    4. Top-K Retrieval
    """)

    st.info(
        "AI-generated answers depend on the quality "
        "of PDF extraction, retrieved passages and "
        "the availability of the Gemini API."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "PaperMind AI | Intelligent Research Paper Analyzer "
    "with Semantic Search"
)
