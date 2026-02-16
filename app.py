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

# --- 1. CONFIGURACIÓN DE SEGURIDAD ---
load_dotenv()
API_KEY = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)
DB_PATH = "data/vectors"
CSV_FILE = "data/raw/Base_maestra_kinesis.csv"

# --- 2. INTERFAZ Y ESTILO (ESTRICTO) ---
st.set_page_config(page_title="Kinesis AI", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    /* Centrar todo el contenedor principal */
    .block-container {
        max-width: 800px;
        padding-top: 2rem;
        color: #ffffff !important;
    }
    
    /* Fondo general */
    .stApp { background: #0f1116; }

    /* Forzar texto blanco en mensajes de chat y labels */
    .stMarkdown, p, li, span, h1, h2, h3 {
        color: #ffffff !important;
    }

    /* Burbujas de chat */
    .stChatMessage { 
        background: #161b22; 
        border-radius: 15px; 
        border: 1px solid #30363d;
        color: #ffffff !important;
    }

    /* AJUSTE DEL CAMPO DE ENTRADA (Mismo ancho que chat) */
    [data-testid="stChatInput"] {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Estilo interno del input */
    [data-testid="stChatInput"] textarea {
        background-color: #21262d !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
    }

    /* Estilo para KPIs */
    .kpi-box {
        background: #1f242c;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #58a6ff;
        margin: 10px 0;
    }
    .kpi-value { font-size: 3.5rem; font-weight: bold; color: #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Kinesis: Sistema de Inteligencia Generativa")

# --- 3. CARGA DE DATOS ---
@st.cache_resource
def load_all():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    v_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    con = duckdb.connect(database=':memory:')
    df = pd.read_csv(CSV_FILE)
    # Limpieza estricta de columnas para SQL
    df.columns = [re.sub(r'[^\w]', '_', c.lower().strip().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')) for c in df.columns]
    con.execute("CREATE TABLE kinesis AS SELECT * FROM df")
    esquema = con.execute("DESCRIBE kinesis").df()[['column_name', 'column_type']].to_string()
    
    return v_db, con, esquema

v_db, sql_db, esquema_cols = load_all()

# --- 4. FUNCIONES DE INTELIGENCIA ---
def get_ai_response(prompt, context="", df_data=None):
    if df_data is not None:
        sys_msg = "Eres un analista. Resume los datos en una frase breve y natural. No menciones el nombre de la tabla ni SQL."
        content = f"Datos obtenidos: {df_data.to_string()}\nPregunta: {prompt}"
    else:
        sys_msg = f"""Eres el Analista Principal de Kinesis. 
        REGLA DE ORO: La tabla se llama 'kinesis'. NUNCA uses 'tabla' o 'tu_tabla'.
        
        Si la pregunta requiere cálculos, responde SOLO con el SQL dentro de ```sql.
        Esquema: {esquema_cols}
        
        Contexto PDF: {context}"""
        content = prompt

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": content}],
        temperature=0.1
    )
    return res.choices[0].message.content

# --- 5. LÓGICA DEL CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "viz" in m:
            if m["viz_type"] == "chart": st.plotly_chart(m["viz"], use_container_width=True)
            elif m["viz_type"] == "kpi": 
                st.markdown(f'<div class="kpi-box"><div class="kpi-value">{m["viz"]}</div></div>', unsafe_allow_html=True)

if user_input := st.chat_input("Escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        docs = v_db.similarity_search(user_input, k=2)
        context_text = "\n".join([d.page_content for d in docs])
        
        initial_res = get_ai_response(user_input, context_text)
        
        if "SELECT" in initial_res.upper():
            try:
                # Limpiar y ejecutar SQL
                query = re.search(r'SELECT.*?;', initial_res.replace('\n', ' '), re.IGNORECASE).group(0)
                # Forzar el nombre de la tabla si la IA se equivoca
                query = query.replace(" tu_tabla", " kinesis").replace(" tabla", " kinesis")
                
                df_res = sql_db.execute(query).df()
                narrativa = get_ai_response(user_input, df_data=df_res)
                st.markdown(narrativa)
                
                msg_data = {"role": "assistant", "content": narrativa}
                
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
                st.error(f"Error: {e}")
        else:
            st.markdown(initial_res)
            st.session_state.messages.append({"role": "assistant", "content": initial_res})