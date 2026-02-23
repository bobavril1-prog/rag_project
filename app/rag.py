import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from openai import OpenAI

# Charger .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

retriever = db.as_retriever(search_kwargs={"k": 10})


# ----------------------------
# 5. LLM local (Ollama)
# ----------------------------

llm_local = OllamaLLM(model="llama3")

# ----------------------------
# 6. Fonction RAG locale (Ollama)
# ----------------------------

def rag_local(query):
    context = retriever.invoke(query)
    context_text = "\n".join([doc.page_content for doc in context])

    prompt = f"""
Tu es mon assistant IA.
Tu réponds uniquement à partir du contexte que je t'ai fourni.
Si l'information n'existe pas dans le contexte, dis "Je suis navré Bob, je ne trouve pas l'information demandée dans les documents."

Contexte :
{context_text}

Question : {query}

Réponse :
"""

    return llm_local.invoke(prompt)

# ----------------------------
# 7. Fonction RAG OpenAI (GPT‑4o)
# ----------------------------

def rag_openai(query):
    client = OpenAI(api_key=OPENAI_API_KEY)

    context = retriever.invoke(query)
    context_text = "\n".join([doc.page_content for doc in context])

    prompt = f"""
Tu es mon assistant IA.
Tu réponds uniquement à partir du contexte que je t'ai fourni.
Si l'information n'existe pas dans le contexte, dis "Je suis navré Bob, je ne trouve pas l'information demandée dans les documents."

Contexte :
{context_text}

Question : {query}

Réponse :
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
