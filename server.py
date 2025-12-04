import chromadb
import requests
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_collection("catalogo")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

class ChatRequest(BaseModel):
    query: str
        
def ask_ollama(prompt):
    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=data)
    return r.json()["response"]

@app.post("/chat")
def chat(request: ChatRequest):
    query = request.query

    # Embedding
    q_embed = embedder.encode([query]).tolist()[0]

    # Buscar en Chroma
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=3
    )

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
Eres un asistente experto en tornería y metalmecánica.
Usa SOLO la información del catálogo para responder.

Si la pregunta NO está en el catálogo, responde:
"Este producto no aparece en el catálogo. Por favor contacte a la empresa al correo: ventas@miempresa.cl"

Pregunta del usuario:
{query}

Información relevante del catálogo:
{context}

Respuesta:
"""

    answer = ask_ollama(prompt)
    return {"respuesta": answer}
