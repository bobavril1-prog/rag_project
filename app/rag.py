import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM


# 1. Charger tous les PDF du dossier data/
docs = []
for file in os.listdir("data"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"data/{file}")
        docs.extend(loader.load())

# 2. Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# 3. Embeddings locaux via Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Vector store FAISS
db = FAISS.from_documents(chunks, embeddings)

# 5. Retriever
retriever = db.as_retriever()

# 6. LLM local via Ollama
llm = OllamaLLM(model="llama3")

# 7. Pipeline RAG
def rag(query):
    context = retriever.invoke(query)
    context_text = "\n".join([c.page_content for c in context])

    prompt = f"""
Tu es un assistant qui répond uniquement à partir du contexte fourni.

Contexte :
{context_text}

Question : {query}

Réponse :
"""

    return llm.invoke(prompt)

# 8. Test
print(rag("Quels sont les points clés abordés dans ces documents ?"))
