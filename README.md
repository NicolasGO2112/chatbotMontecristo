# chatbotMontecristo

Este repositorio contiene una API en FastAPI que consulta una base vectorial (Chroma) con embeddings
generados por `sentence-transformers` y genera respuestas usando Ollama.

## Requisitos
- Python 3.9+ (recomendado 3.10 o 3.11)
- Git
- (Opcional) Ollama corriendo localmente en `http://localhost:11434` con el modelo `llama3.1:8b` si deseas usar la integración por defecto.

## Pasos para levantar el proyecto (desde 0)

1. Clonar el repositorio

```powershell
git clone https://github.com/NicolasGO2112/chatbotMontecristo.git
cd chatbotMontecristo
```

2. Crear y activar un entorno virtual (Windows PowerShell)

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
```

3. Instalar dependencias básicas

```powershell
pip install --upgrade pip
pip install fastapi uvicorn chromadb sentence-transformers requests pydantic pandas openpyxl
```

4. (Recomendado) Generar `requirements.txt` para reproducibilidad

```powershell
pip freeze > requirements.txt
```

5. (Si necesitas poblar o regenerar la base vectorial) Ejecutar la ingestión

- Asegúrate de tener `catalogo.xlsx` en la raíz del proyecto con las columnas esperadas (ej.: `codigo`, `nombre`, `descripcion`, `material`, `categoria`, `dimensiones`, `stock`, `precio`, `proveedor`).

```powershell
python ingest.py
```

Esto creará/actualizará la colección `catalogo` dentro de la carpeta `db/`.

6. Arrancar la API (FastAPI)

```powershell
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

El endpoint para consultas es `POST http://127.0.0.1:8000/chat`.

7. Probar el endpoint (ejemplo PowerShell)

```powershell
$body = @{ query = "¿Tienen tornillos de acero?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/chat -Method Post -Body $body -ContentType 'application/json'
```

## Notas importantes
- Ollama: el `server.py` llama a Ollama en `http://localhost:11434`. Si no tienes Ollama, o prefieres usar OpenAI u otro servicio, edita la función `ask_ollama()` en `server.py`.
- Recursos: `sentence-transformers` puede instalar PyTorch u otro backend. En máquinas sin GPU la inferencia puede ser lenta.
- Chroma DB: el proyecto ya puede contener `db/chroma.sqlite3`. Si quieres regenerarla, puedes eliminar la carpeta `db/` antes de ejecutar `ingest.py`.
- `.gitignore`: se añadió un `.gitignore` para excluir `venv/`, cachés y la base de datos local. Si ya tienes esos archivos en el historial, quítalos con:

```powershell
git rm --cached db/chroma.sqlite3
git rm -r --cached venv
git commit -m "Remove tracked local files; add .gitignore"
```

