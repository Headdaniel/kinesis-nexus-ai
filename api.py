import os
import re
import duckdb
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
CSV_FILE = "data/raw/Base_maestra_kinesis.csv"
CONTEXT_CSV_FILE = "data/raw/Explicacion_contexto_programa.csv"

# --- CARGA DE RECURSOS ---
def ingest_context_csv_to_chroma(v_db, csv_path: str):
    df_ctx = pd.read_csv(csv_path)
    full_text = ""
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
    ingest_context_csv_to_chroma(v_db, CONTEXT_CSV_FILE)

    con = duckdb.connect(database=':memory:')
    df = pd.read_csv(CSV_FILE)
    df.columns = [
        re.sub(r'[^\w]', '_', c.lower().strip()
            .replace('á','a').replace('é','e')
            .replace('í','i').replace('ó','o').replace('ú','u'))
        for c in df.columns
    ]
    con.execute("CREATE TABLE kinesis AS SELECT * FROM df")
    esquema = con.execute("DESCRIBE kinesis").df()[['column_name', 'column_type']].to_string()
    return v_db, con, esquema

print("Cargando base de conocimientos...")
v_db, sql_db, esquema_cols = load_all()
print("✅ Listo.")

# --- FUNCIÓN DE IA (idéntica a app.py) ---
def get_ai_response(prompt, context="", df_data=None):
    if df_data is not None:
        sys_msg = "Eres un analista experto. Resume los datos en una frase natural y breve. Sé directo."
        content = f"Datos obtenidos: {df_data.to_string()}\nPregunta: {prompt}"
    else:
        sys_msg = (
            "Eres el Analista Principal de Kinesis.\n\n"
            "REGLAS IMPORTANTES:\n"
            "1. Si la respuesta está explícitamente en el CONTEXTO proporcionado, debes usarlo como fuente principal.\n"
            "2. No inventes ni infieras si el contexto contiene la respuesta.\n"
            "3. Si la pregunta menciona ingresos, ventas, montos, totales, sumas, años o países, DEBES generar una consulta SQL válida usando la tabla 'kinesis'. Siempre intenta generar el SQL antes de decir que no existe información. No expliques. Si es cálculo, responde SOLO con el SQL en ```sql```.\n"
            "4. Si la pregunta es conceptual o programática, responde usando el CONTEXTO.\n"
            "5. Si el usuario saluda (hola, buenos días, hey, etc.), responde con un saludo amigable y ofrece ayuda con el proyecto Kinesis.\n"
            "6. Si la pregunta NO está relacionada con el proyecto Kinesis y NO es un saludo, responde: 'Mi función está enfocada en el proyecto Kinesis. ¿En qué puedo ayudarte sobre el programa?'\n\n"
            "TABLA DISPONIBLE: 'kinesis'\n"
            f"Esquema: {esquema_cols}\n\n"
            "CONTEXTO DISPONIBLE:\n"
            f"{context}\n"
        )
        content = prompt

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": content}
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

# El HTML envía "texto"
class ChatRequest(BaseModel):
    texto: str

@app.post("/chat")
def chat(req: ChatRequest):
    user_input = req.texto

    # RAG: buscar contexto en vectores
    docs = v_db.similarity_search(user_input, k=2)
    context_text = "\n".join([d.page_content for d in docs])

    # Respuesta inicial
    initial_res = get_ai_response(user_input, context_text)

    # Si la IA generó SQL, ejecutarlo
    if "SELECT" in initial_res.upper():
        try:
            sql_match = re.search(r'SELECT.*', initial_res.replace('\n', ' '), re.IGNORECASE)
            query = sql_match.group(0).split('```')[0].strip()
            query = query.replace(" tu_tabla", " kinesis").replace(" tabla", " kinesis")
            if not query.endswith(';'):
                query += ';'

            df_res = sql_db.execute(query).df()
            narrativa = get_ai_response(user_input, df_data=df_res)

            # KPI: una sola cifra
            if len(df_res) == 1 and len(df_res.columns) == 1:
                return {
                    "tipo": "kpi",
                    "valor": str(df_res.iloc[0, 0]),
                    "interpretacion": narrativa
                }

            # GRÁFICO: múltiples filas
            elif len(df_res) > 0:
                col_x = df_res.columns[0]
                col_y = df_res.columns[1]
                plotly_data = {
                    "data": [{
                        "type": "bar",
                        "x": df_res[col_x].tolist(),
                        "y": df_res[col_y].tolist(),
                        "marker": {"color": "#d50262"}
                    }],
                    "layout": {
                        "template": "plotly_white",
                        "title": {
                            "text": user_input,
                            "font": {"family": "Plus Jakarta Sans, sans-serif", "size": 13, "color": "#0d0d55"},
                            "x": 0.02
                        },
                        "margin": {"t": 45, "b": 40, "l": 40, "r": 20},
                        "xaxis": {
                            "title": col_x,
                            "tickfont": {"family": "Plus Jakarta Sans, sans-serif", "size": 11, "color": "#595959"}
                        },
                        "yaxis": {
                            "title": col_y,
                            "tickfont": {"family": "Plus Jakarta Sans, sans-serif", "size": 11, "color": "#595959"}
                        },
                        "font": {"family": "Plus Jakarta Sans, sans-serif"}
                    }
                }
                return {
                    "tipo": "grafico",
                    "datos_del_grafico": plotly_data,
                    "interpretacion": narrativa
                }

        except Exception as e:
            return {
                "tipo": "texto",
                "respuesta": f"Tuve un problema técnico procesando esos datos: {str(e)}"
            }

    # TEXTO normal
    return {
        "tipo": "texto",
        "respuesta": initial_res
    }
