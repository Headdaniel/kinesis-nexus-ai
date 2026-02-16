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

# --- 1. CONFIGURACIÓN DE SEGURIDAD (Local vs Nube) ---
load_dotenv()
# En la nube usará st.secrets, en local os.getenv
API_KEY = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")

client = Groq(api_key=API_KEY)
DB_PATH = "data/vectors"
CSV_FILE = "data/raw/Base_maestra_kinesis.csv"

# --- 2. INTERFAZ Y ESTILO RESPONSIVE ---
st.set_page_config(page_title="Kinesis AI", page_icon="🧠", layout="wide")

# CSS para centrar el chat y hacerlo responsive
st.markdown("""
    <style>
    /* Centrar el contenido y limitar ancho */
    .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }
    .stApp { background: #0f1116; color: #e6edf3; }
    
    /* Burbujas de chat */
    .stChatMessage { 
        background: #161b22; 
        border-radius: 15px; 
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #30363d;
    }
    
    /* Estilo para números grandes (KPIs) */
    .kpi-box {
        background: #1f242c;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #58a6ff;
        margin: 10px 0;
    }
    .kpi-value { font-size: 3rem; font-weight: bold; color: #58a6ff; }
    .kpi-label { font-size: 1rem; color: #8b949e; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Kinesis: Sistema de Inteligencia Generativa")

# --- 3. CARGA DE DATOS ---
@st.cache_resource
def load_all():
    # PDF Memory
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    v_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # CSV / DuckDB
    con = duckdb.connect(database=':memory:')
    df = pd.read_csv(CSV_FILE)
    df.columns = [re.sub(r'[^\w]', '_', c.lower().strip().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')) for c in df.columns]
    con.execute("CREATE TABLE kinesis AS SELECT * FROM df")
    esquema = con.execute("DESCRIBE kinesis").df()[['column_name', 'column_type']].to_string()
    
    return v_db, con, esquema

v_db, sql_db, esquema_cols = load_all()

# --- 4. FUNCIONES DE INTELIGENCIA ---
def get_ai_response(prompt, context="", df_data=None):
    # Si tenemos datos de un DataFrame, pedimos a la IA que los resuma
    if df_data is not None:
        sys_msg = "Eres un analista. Resume los siguientes datos en una frase breve y natural para el usuario."
        content = f"Pregunta original: {prompt}\nDatos obtenidos: {df_data.to_string()}\nRespuesta:"
    else:
        # Si no, es una pregunta normal de PDF o decisión SQL
        sys_msg = f"Eres un analista senior. Contexto PDF: {context}\nEsquema SQL: {esquema_cols}\nSi la pregunta requiere datos, responde SOLO con el código SQL en un bloque ```sql."
        content = prompt

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": content}],
        temperature=0.1
    )
    return res.choices[0].message.content

# --- 5. CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "viz" in m:
            if m["viz_type"] == "df": st.dataframe(m["viz"])
            elif m["viz_type"] == "chart": st.plotly_chart(m["viz"], use_container_width=True)
            elif m["viz_type"] == "kpi": 
                st.markdown(f'<div class="kpi-box"><div class="kpi-value">{m["viz"]}</div><div class="kpi-label">Resultado del Análisis</div></div>', unsafe_allow_html=True)

if user_input := st.chat_input("Escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        # Buscar contexto PDF
        docs = v_db.similarity_search(user_input, k=2)
        context_text = "\n".join([d.page_content for d in docs])
        
        # 1. Decisión de la IA
        initial_res = get_ai_response(user_input, context_text)
        
        if "SELECT" in initial_res.upper():
            try:
                # 2. Ejecutar SQL
                query = re.search(r'SELECT.*?;', initial_res.replace('\n', ' '), re.IGNORECASE).group(0)
                df_res = sql_db.execute(query).df()
                
                # 3. Generar Narrativa Natural basada en los datos
                narrativa = get_ai_response(user_input, df_data=df_res)
                st.markdown(narrativa)
                
                # 4. Visualización Inteligente
                msg_data = {"role": "assistant", "content": narrativa}
                
                if len(df_res) == 1 and len(df_res.columns) == 1:
                    # Es un solo número (KPI)
                    val = df_res.iloc[0,0]
                    st.markdown(f'<div class="kpi-box"><div class="kpi-value">{val}</div><div class="kpi-label">Dato Calculado</div></div>', unsafe_allow_html=True)
                    msg_data.update({"viz": val, "viz_type": "kpi"})
                elif len(df_res) > 1:
                    # Gráfico de barras
                    fig = px.bar(df_res, x=df_res.columns[0], y=df_res.columns[1], template="plotly_dark", color_discrete_sequence=['#58a6ff'])
                    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    msg_data.update({"viz": fig, "viz_type": "chart"})
                else:
                    st.dataframe(df_res)
                    msg_data.update({"viz": df_res, "viz_type": "df"})
                
                st.session_state.messages.append(msg_data)
                
            except Exception as e:
                st.error(f"Lo siento, hubo un error analizando los datos: {e}")
        else:
            st.markdown(initial_res)
            st.session_state.messages.append({"role": "assistant", "content": initial_res})