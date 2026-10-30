import streamlit as st

st.title("PaperMind AI - Dependency Test")

packages = [
    "fitz",
    "numpy",
    "sklearn",
    "sentence_transformers",
    "dotenv",
    "google.genai"
]

for package in packages:
    try:
        __import__(package)
        st.success(f"{package} - Installed")
    except Exception as e:
        st.error(f"{package} - Error: {e}")