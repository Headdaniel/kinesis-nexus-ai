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

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Kinesis AI Pro", page_icon="🧠", layout="wide")

# --- 2. SEGURIDAD ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "Kinesis2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<h1 style='text-align: center; color: #c9e0ffff;'>Kinesis Nexus</h1>", unsafe_allow_html=True)
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
DB_PATH = "data/vectors"
CSV_FILE = "data/raw/Base_maestra_kinesis.csv"

# --- ESTILO CSS (Sora, Manrope y Color Celeste) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;700&family=Sora:wght@700&display=swap');

    /* Color de fondo y texto general */
    .stApp { background-color: #0f1116 !important; }
    
    html, body, [class*="css"], .stMarkdown, p, li, span, label {
        font-family: 'Manrope', sans-serif !important;
        color: #c9e0ffff !important;
    }

    /* Títulos y bienvenida con Sora en negrita */
    .sora-bold {
        font-family: 'Sora', sans-serif !important;
        font-weight: 700 !important;
        color: #c9e0ffff !important;
    }
    
    h1, h2, h3 {
        font-family: 'Sora', sans-serif !important;
        font-weight: 700 !important;
        color: #c9e0ffff !important;
    }

    /* Ajustes de contenedores */
    .block-container { max-width: 800px; padding-top: 1rem; }
    footer {display: none;}
    
    /* Input de chat */
    [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] { background-color: #0f1116 !important; }
    [data-testid="stChatInput"] { max-width: 800px; margin: 0 auto; background-color: #0f1116 !important; }
    [data-testid="stChatInput"] textarea { background-color: #21262d !important; color: #c9e0ffff !important; border: 1px solid #30363d !important; }
    
    /* Burbujas de chat */
    .stChatMessage { background-color: #161b22 !important; border-radius: 15px !important; border: 1px solid #30363d !important; }
    
    /* KPIs */
    .kpi-box { background: #1f242c; padding: 25px; border-radius: 12px; text-align: center; border: 2px solid #58a6ff; margin: 15px 0; }
    .kpi-value { font-family: 'Sora', sans-serif; font-size: 3.8rem; font-weight: 800; color: #c9e0ffff; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGO (Posicionamiento adaptativo) ---
if os.path.exists("Logo Kinesis_Negativo.png"):
    st.image("Logo Kinesis_Negativo.png", width=120)

# --- 4. CARGA DE DATOS ---
@st.cache_resource
def load_all():
    with st.spinner("Cargando base de conocimientos..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        v_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        con = duckdb.connect(database=':memory:')
        df = pd.read_csv(CSV_FILE)
        df.columns = [re.sub(r'[^\w]', '_', c.lower().strip().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')) for c in df.columns]
        con.execute("CREATE TABLE kinesis AS SELECT * FROM df")
        esquema = con.execute("DESCRIBE kinesis").df()[['column_name', 'column_type']].to_string()
        return v_db, con, esquema

try:
    v_db, sql_db, esquema_cols = load_all()
except Exception as e:
    st.error(f"Error crítico al iniciar: {e}")
    st.stop()

# --- 5. FUNCIONES DE IA ---
def get_ai_response(prompt, context="", df_data=None):
    if df_data is not None:
        sys_msg = "Eres un analista experto. Resume los datos en una frase natural y breve. Sé directo. No menciones código técnico."
        content = f"Datos obtenidos: {df_data.to_string()}\nPregunta: {prompt}"
    else:
        sys_msg = f"""Analista Principal de Kinesis. 
        TABLA: 'kinesis'. NUNCA uses otro nombre.
        Si hay cálculos, responde SOLO con el SQL en ```sql.
        Esquema: {esquema_cols}
        Contexto PDF: {context}"""
        content = prompt

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": content}],
        temperature=0.1
    )
    return res.choices[0].message.content

# --- 6. INTERFAZ DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Saludo de bienvenida central (ChatGPT/Gemini Style)
if len(st.session_state.messages) == 0:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center;'>
            <h2 class='sora-bold' style='font-size: 2.2rem;'>¡Hola, ChangeLabiano!</h2>
            <p style='font-size: 1.1rem;'>Soy el experto en el proyecto Kinesis. ¿En qué te puedo ayudar hoy?</p>
        </div>
    """, unsafe_allow_html=True)

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
        docs = v_db.similarity_search(user_input, k=2)
        context_text = "\n".join([d.page_content for d in docs])
        
        initial_res = get_ai_response(user_input, context_text)
        
        # Identificar fuente
        fuente = "Base_maestra_kinesis.csv" if "SELECT" in initial_res.upper() else "Documentación_Kinesis.pdf"
        texto_fuente = f"\n\n--- \n*Esta respuesta te la doy revisando esta fuente: {fuente}*"
        
        if "SELECT" in initial_res.upper():
            try:
                sql_match = re.search(r'SELECT.*', initial_res.replace('\n', ' '), re.IGNORECASE)
                query = sql_match.group(0).split('```')[0].strip()
                query = query.replace(" tu_tabla", " kinesis").replace(" tabla", " kinesis")
                if not query.endswith(';'): query += ';'
                
                df_res = sql_db.execute(query).df()
                narrativa = get_ai_response(user_input, df_data=df_res)
                
                # Unir narrativa con fuente
                full_narrativa = narrativa + texto_fuente
                st.markdown(full_narrativa)
                
                msg_data = {"role": "assistant", "content": full_narrativa}
                
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
            full_res = initial_res + texto_fuente
            st.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})