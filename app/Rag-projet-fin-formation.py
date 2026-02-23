import streamlit as st
from rag import rag_local, rag_openai

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("📄 Assistant IA - Analyse de documents")

model_choice = st.radio(
    "Choisissez le modèle :",
    ["Local (Ollama)", "OpenAI (GPT‑4o)"]
)

query = st.text_input("Votre question :")

if query:
    with st.spinner("Analyse en cours..."):
        if model_choice == "Local (Ollama)":
            response = rag_local(query)
        else:
            response = rag_openai(query)

    st.subheader("Réponse :")
    st.write(response)
