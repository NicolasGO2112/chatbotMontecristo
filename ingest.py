import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# Cargar modelo de embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Cargar Excel
df = pd.read_excel("catalogo.xlsx")

# Unir datos como texto
def row_to_text(row):
    return f"""
    Codigo: {row['codigo']}
    Nombre: {row['nombre']}
    Descripción: {row['descripcion']}
    Material: {row['material']}
    Categoría: {row['categoria']}
    Dimensiones: {row['dimensiones']}
    Stock: {row['stock']}
    Precio: {row['precio']}
    Proveedor: {row['proveedor']}

    """

texts = [row_to_text(row) for _, row in df.iterrows()]

# Crear base vectorial
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection(name="catalogo")

embeddings = embedder.encode(texts).tolist()

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=[str(i) for i in range(len(texts))]
)

print("Ingestión completada ✔")
