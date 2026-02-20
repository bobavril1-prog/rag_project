import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM


# ----------------------------
# 1. Charger les PDF
# ----------------------------

docs = []
for file in os.listdir("data"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"data/{file}")
        docs.extend(loader.load())


# ----------------------------
# 2. Découpage (chunking)
# ----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)


# ----------------------------
# 3. Embeddings (local Ollama)
# ----------------------------

embeddings = OllamaEmbeddings(model="nomic-embed-text")


# ----------------------------
# 4. Base vectorielle FAISS
# ----------------------------

if os.path.exists("faiss_index"):
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")


retriever = db.as_retriever()


# ----------------------------
# 5. LLM local (Ollama)
# ----------------------------

llm = OllamaLLM(model="llama3")


# ----------------------------
# 6. Fonction RAG
# ----------------------------

def rag(query):

    context = retriever.invoke(query)

    context_text = "\n".join([doc.page_content for doc in context])

    prompt = f"""
Tu es un assistant IA.
Tu réponds uniquement à partir du contexte fourni.
Si l'information n'existe pas dans le contexte, dis "Information non trouvée dans les documents."

Contexte :
{context_text}

Question : {query}

Réponse :
"""

    response = llm.invoke(prompt)

    return response


# ----------------------------
# 7. Test local
# ----------------------------

if __name__ == "__main__":
    print(rag("Quels sont les points clés abordés dans ces documents ?"))



   