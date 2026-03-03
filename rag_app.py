import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- CONFIGURACIÓN ---
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)

DB_PATH = "data/vectors_v2"
CONTEXT_CSV_FILE = "data/raw/Explicacion_contexto_programa.csv"

# --- CARGA DE RECURSOS ---
def ingest_context_csv_to_chroma(v_db, csv_path: str):
    df_ctx = pd.read_csv(csv_path)
    full_text = ""
    
    # Recorrer cada fila y columna para extraer el texto
    for _, row in df_ctx.iterrows():
        for col in df_ctx.columns:
            value = str(row[col]).strip()
            if value and value.lower() != "nan":
                full_text += value + "\n"
        full_text += "\n"
        
    doc = Document(page_content=full_text, metadata={"source": "context_csv_full"})
    v_db.add_documents([doc])

def load_all():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    v_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # Solo ingerimos si la colección está vacía para no duplicar datos
    try:
        existing = v_db._collection.count()
    except Exception:
        existing = 0

    if existing == 0:
        ingest_context_csv_to_chroma(v_db, CONTEXT_CSV_FILE)

    return v_db

print("Cargando base de conocimientos...")
v_db = load_all()
print("✅ Listo.")

# --- FUNCIÓN DE IA ---
def get_ai_response(prompt, context=""):
    sys_msg = (
        "Eres el Analista Principal de Kinesis.\n\n"
        "ESTILO (OBLIGATORIO):\n"
        "1. Sé breve y directo.\n"
        "2. Usa 1 párrafo. Sin listas. Sin enlaces. Sin cierre tipo '¿te gustaría...?'.\n\n"
        "REGLAS IMPORTANTES:\n"
        "1. Si la respuesta está explícitamente en el CONTEXTO proporcionado, debes usarlo como fuente principal.\n"
        "2. No inventes ni infieras si el contexto contiene la respuesta.\n"
        "3. Si la pregunta es conceptual o programática, responde usando el CONTEXTO.\n"
        "4. Si el usuario saluda (hola, buenos días, hey, etc.), responde con un saludo amigable y ofrece ayuda con el proyecto Kinesis.\n"
        "5. Si la pregunta NO está relacionada con el proyecto Kinesis y NO es un saludo, responde: 'Mi función está enfocada en el proyecto Kinesis. ¿En qué puedo ayudarte sobre el programa?'\n\n"
        "CONTEXTO DISPONIBLE:\n"
        f"{context}\n"
    )

    res = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return res.choices[0].message.content


# --- API ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar por tu dominio en producción
    allow_methods=["POST"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    texto: str

@app.post("/chat")
def chat(req: ChatRequest):
    user_input = req.texto

    # RAG: buscar contexto en vectores
    docs = v_db.similarity_search(user_input, k=2)
    
    # Unir todo el contexto encontrado (sin límite de caracteres)
    context_text = "\n\n".join([d.page_content for d in docs])

    # Generar respuesta usando el contexto
    response = get_ai_response(user_input, context_text)

    # TEXTO normal
    return {
        "tipo": "texto",
        "respuesta": response
    }