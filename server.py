import os
from dotenv import load_dotenv
import chromadb
import requests
from requests.exceptions import RequestException
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from uuid import uuid4

# Cargar variables de entorno desde .env
load_dotenv()

app = FastAPI()

chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("catalogo")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Almacén de conversaciones en memoria (en producción usar Redis o DB)
conversations: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY = 10  # Máximo de mensajes a recordar por conversación

class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None  # ID para mantener historial

class ChatResponse(BaseModel):
    respuesta: str
    conversation_id: str
        
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

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    query = request.query
    
    # Gestionar conversation_id
    conv_id = request.conversation_id or str(uuid4())
    if conv_id not in conversations:
        conversations[conv_id] = []
    
    # Obtener historial de la conversación
    history = conversations[conv_id]

    # Embedding para búsqueda en catálogo
    q_embed = embedder.encode([query]).tolist()[0]

    # Buscar en Chroma - 5 resultados para contexto
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=5
    )

    catalog_context = "\n\n".join(results["documents"][0])
    
    # Construir historial de conversación para el prompt
    history_text = ""
    if history:
        history_text = "HISTORIAL DE CONVERSACIÓN:\n"
        for msg in history[-MAX_HISTORY:]:
            history_text += f"Usuario: {msg['user']}\nAsistente: {msg['assistant']}\n\n"

    prompt = f"""Eres un asistente virtual amigable de la empresa Montecristo, experto en tornería, metalmecánica y suministros industriales.

PERSONALIDAD:
- Eres amable, profesional y cercano
- Respondes saludos como "hola", "buenos días", etc. de forma natural y cálida
- Agradeces cuando el usuario dice "gracias" o similar
- Si te despiden, despídete amablemente e invita a volver

TU ROL TÉCNICO:
1. ANALIZAR el problema o necesidad que plantea el usuario
2. PROPONER soluciones técnicas basadas en tu conocimiento
3. RECOMENDAR productos del catálogo cuando sean útiles
4. Dar consejos prácticos sobre uso, instalación o selección de materiales

INSTRUCCIONES:
- Para saludos/despedidas/agradecimientos: responde de forma natural y breve, sin buscar en el catálogo
- Para consultas técnicas: analiza, sugiere soluciones y recomienda productos con código, nombre y precio
- Puedes combinar varios productos si es necesario
- Si no hay productos relevantes pero puedes dar consejos técnicos, hazlo
- Si la consulta está fuera de tu área, indica amablemente que asesoras en tornería y metalmecánica
- Mantén coherencia con la conversación previa si existe

{history_text}
CATÁLOGO DISPONIBLE:
{catalog_context}

MENSAJE ACTUAL DEL USUARIO:
{query}

RESPUESTA:"""

    answer = ask_ollama(prompt)
    
    # Guardar en historial
    conversations[conv_id].append({
        "user": query,
        "assistant": answer
    })
    
    # Limitar tamaño del historial
    if len(conversations[conv_id]) > MAX_HISTORY:
        conversations[conv_id] = conversations[conv_id][-MAX_HISTORY:]
    
    return ChatResponse(respuesta=answer, conversation_id=conv_id)


@app.delete("/chat/{conversation_id}")
def clear_conversation(conversation_id: str):
    """Limpiar historial de una conversación"""
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"message": "Conversación eliminada"}
    return {"message": "Conversación no encontrada"}
