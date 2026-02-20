import streamlit as st
from rag import rag

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("📄 Assistant IA - Analyse de documents")

st.write("Posez une question basée sur vos documents PDF.")

query = st.text_input("Votre question :")

if query:
    with st.spinner("Analyse en cours..."):
        response = rag(query)

    st.subheader("Réponse :")
    st.write(response)