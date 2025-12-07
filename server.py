import os
from dotenv import load_dotenv
import chromadb
import requests
from requests.exceptions import RequestException
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Cargar variables de entorno desde .env
load_dotenv()

app = FastAPI()

chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("catalogo")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")

class ChatRequest(BaseModel):
    query: str
        
def ask_ollama(prompt):
    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_URL, json=data, timeout=120)  # 2 minutos de timeout
        r.raise_for_status()
        return r.json().get("response", "")
    except RequestException as e:
        raise HTTPException(status_code=503, detail=f"Ollama no disponible en {OLLAMA_URL}: {str(e)}")

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
