import streamlit as st
import os
import duckdb
import pandas as pd
import plotly.express as px
import re
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# --- 1. CONFIGURACIÓN DE PÁGINA (DEBE SER LO PRIMERO) ---
st.set_page_config(page_title="Kinesis AI Pro", page_icon="🧠", layout="wide")

# --- 2. SEGURIDAD: CONTROL DE ACCESO ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "Kinesis2026": # Tu clave
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<h1 style='text-align: center; color: #58a6ff;'>Kinesis Nexus</h1>", unsafe_allow_html=True)
        st.text_input("Introduce la clave de acceso", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Clave incorrecta. Intenta de nuevo", type="password", on_change=password_entered, key="password")
        st.error("😕 Ups, esa no es la clave.")
        return False
    return True

if not check_password():
    st.stop()

# --- 3. CONFIGURACIÓN DE RECURSOS ---
load_dotenv()
API_KEY = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)
DB_PATH = "data/vectors_v2"
CSV_FILE = "data/raw/Base_maestra_kinesis.csv"
CONTEXT_CSV_FILE = "data/raw/Explicacion_contexto_programa.csv"

def ingest_context_csv_to_chroma(v_db, csv_path: str):
    df_ctx = pd.read_csv(csv_path)

    full_text = ""

    for _, row in df_ctx.iterrows():
        for col in df_ctx.columns:
            value = str(row[col]).strip()
            if value and value.lower() != "nan":
                full_text += value + "\n"
        full_text += "\n"

    doc = Document(
        page_content=full_text,
        metadata={"source": "context_csv_full"}
    )


    v_db.add_documents([doc])


# Estilo CSS para modo oscuro total y centrado
st.markdown("""
    <style>
    .stApp { 
        background-color: #002f6cff !important; 
    }

    .block-container { 
        max-width: 800px; 
        padding-top: 2rem; 
    }

    footer {display: none;}

    [data-testid="stBottom"], 
    [data-testid="stBottomBlockContainer"] { 
        background-color: #002f6cff !important; 
    }

    [data-testid="stChatInput"] { 
        max-width: 800px; 
        margin: 0 auto; 
        background-color: #002f6cff !important; 
    }

    [data-testid="stChatInput"] textarea { 
        background-color: #002f6cff !important; 
        color: #c9e0ffff !important; 
        border: 1px solid #c9e0ffff !important; 
    }

    .stChatMessage { 
        background-color: #002f6cff !important; 
        border-radius: 15px !important; 
        border: 1px solid #c9e0ffff !important; 
    }

    .stMarkdown, 
    p, li, span, 
    h1, h2, h3, 
    label, 
    div, 
    textarea, 
    input { 
        color: #c9e0ffff !important; 
    }

    .kpi-box {
    background-color: #001f4d !important;  /* Más oscuro que el texto */
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #c9e0ffff;
    }


    .kpi-value { 
        font-size: 3.8rem; 
        font-weight: 800; 
        color: #c9e0ffff; 
    }
    pre, code {
    background-color: #001f4d !important;
    color: #c9e0ffff !important;
    }

    </style>
    """, unsafe_allow_html=True)


st.markdown(
    """
    <div style="text-align: center; margin-top: 120px;">
        <img src="data:image/png;base64,{}" width="60" style="margin-bottom: 20px;">
        <h1 style="margin-bottom: 10px;">
            ¡Hola, ChangeLab!
        </h1>
        <h3 style="font-weight: normal; margin-top: 0;">
            Soy el experto en el proyecto Kinesis ¿En qué te puedo ayudar hoy?
        </h3>
    </div>
    """.format(
        __import__("base64").b64encode(open("cabezarobot.png", "rb").read()).decode()
    ),
    unsafe_allow_html=True
)



# --- 4. CARGA DE DATOS (CON CACHÉ Y SPINNER) ---
@st.cache_resource
def load_all():
    with st.spinner("Cargando base de conocimientos..."):
        # Modelos para PDF
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        v_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        
        ingest_context_csv_to_chroma(v_db, CONTEXT_CSV_FILE)

        # Base de datos SQL
        con = duckdb.connect(database=':memory:')
        df = pd.read_csv(CSV_FILE)
        df.columns = [re.sub(r'[^\w]', '_', c.lower().strip().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')) for c in df.columns]
        con.execute("CREATE TABLE kinesis AS SELECT * FROM df")
        esquema = con.execute("DESCRIBE kinesis").df()[['column_name', 'column_type']].to_string()
        return v_db, con, esquema

# Intentar cargar todo
try:
    v_db, sql_db, esquema_cols = load_all()
    
except Exception as e:
    st.error(f"Error crítico al iniciar: {e}")
    st.stop()


# --- 5. FUNCIONES DE IA ---
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
            "4. Si la pregunta es conceptual o programática, responde usando el CONTEXTO.\n
            "5. Si la pregunta NO está relacionada con el proyecto Kinesis, su base de datos, su contexto programático o sus resultados, debes responder lo siguiente EXACTAMENTE:\n"
                "'Sí, estoy en la capacidad de responder esa pregunta porque soy una IA general. Sin embargo, mi función aquí está enfocada exclusivamente en el proyecto Kinesis. Si quieres, puedo ayudarte con algo relacionado con el proyecto, por ejemplo, explicarte qué es una teoría de cambio o analizar algún resultado del programa.'\n\n"
            "TABLA DISPONIBLE: 'kinesis'\n"
            f"Esquema: {esquema_cols}\n\n"
            "CONTEXTO DISPONIBLE:\n"
            f"{context}\n"
        )
        content = prompt

    res = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": content}
        ],
        temperature=0.1
    )
    return res.choices[0].message.content


# --- 6. INTERFAZ DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "viz" in m:
            if m["viz_type"] == "chart": st.plotly_chart(m["viz"], use_container_width=True)
            elif m["viz_type"] == "kpi": st.markdown(f'<div class="kpi-box"><div class="kpi-value">{m["viz"]}</div></div>', unsafe_allow_html=True)

# Entrada del usuario
if user_input := st.chat_input("¿Qué quieres consultar hoy?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Buscar en PDF
        docs = v_db.similarity_search(user_input, k=6)

        context_text = "\n".join([d.page_content for d in docs])
        
        # Obtener respuesta inicial
        initial_res = get_ai_response(user_input, context_text)
        
        if "SELECT" in initial_res.upper():
            try:
                # Extraer y limpiar SQL
                sql_match = re.search(r'SELECT.*', initial_res.replace('\n', ' '), re.IGNORECASE)
                query = sql_match.group(0).split('```')[0].strip()
                query = query.replace(" tu_tabla", " kinesis").replace(" tabla", " kinesis")
                if not query.endswith(';'): query += ';'
                
                # Ejecutar
                df_res = sql_db.execute(query).df()
                
                # Narrativa natural
                narrativa = get_ai_response(user_input, df_data=df_res)
                st.markdown(narrativa)
                
                msg_data = {"role": "assistant", "content": narrativa}
                
                # Visualización
                if len(df_res) == 1 and len(df_res.columns) == 1:
                    val = df_res.iloc[0,0]
                    st.markdown(f'<div class="kpi-box"><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)
                    msg_data.update({"viz": val, "viz_type": "kpi"})
                elif len(df_res) > 0:
                    fig = px.bar(df_res, x=df_res.columns[0], y=df_res.columns[1], template="plotly_dark", color_discrete_sequence=['#58a6ff'])
                    st.plotly_chart(fig, use_container_width=True)
                    msg_data.update({"viz": fig, "viz_type": "chart"})
                
                st.session_state.messages.append(msg_data)
                
            except Exception as e:
                st.error("Tuve un problema técnico procesando esos datos.")
        else:
            st.markdown(initial_res)
            st.session_state.messages.append({"role": "assistant", "content": initial_res})