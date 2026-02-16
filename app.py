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

# --- 1. CONFIGURACIÓN DE PÁGINA (ESTO NO SE TOCA) ---
st.set_page_config(page_title="Kinesis AI Pro", page_icon="🧠", layout="wide")

# ==========================================
# SECCIÓN DE ESTILO Y FORMA (LO ÚNICO QUE CAMBIÉ)
# ==========================================
COLOR_FONDO = "#002f6cff"      # Azul Opaco
COLOR_TEXTO = "#c9e0ffff"      # Celeste
COLOR_USUARIO = "#ba0c2fff"    # Fucsia
COLOR_IA = "#002060ff"         # Azul Encendido

st.markdown(f"""
    <style>
    /* Importar Fuentes */
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;700&family=Sora:wght@400;700&display=swap');

    /* 1. FONDO Y TIPOGRAFÍA GLOBAL */
    .stApp {{ background-color: {COLOR_FONDO} !important; }}
    html, body, p, li, span, label, h1, h2, h3, div, .stMarkdown {{
        font-family: 'Manrope', sans-serif !important;
        color: {COLOR_TEXTO} !important;
    }}
    h1, h2, h3 {{ font-family: 'Sora', sans-serif !important; }}

    /* 2. BARRA DE INPUT (ANCHO REDUCIDO) */
    [data-testid="stChatInput"] {{
        max-width: 800px !important;
        margin: 0 auto !important;
        background-color: {COLOR_FONDO} !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background-color: {COLOR_IA} !important;
        color: {COLOR_TEXTO} !important;
        border: 1px solid {COLOR_TEXTO} !important;
    }}
    [data-testid="stBottom"] {{ background-color: {COLOR_FONDO} !important; border-top: 1px solid {COLOR_IA}; }}

    /* 3. ALINEACIÓN DEL CHAT */
    .stChatMessage {{ background-color: transparent !important; border: none !important; }}
    
    /* USUARIO (Derecha - Fucsia) */
    [data-testid="stChatMessage"]:nth-child(odd) {{
        flex-direction: row-reverse !important;
        text-align: right !important;
    }}
    [data-testid="stChatMessage"]:nth-child(odd) .stMarkdown {{
        background-color: {COLOR_USUARIO} !important;
        text-align: right !important;
        border-radius: 15px 15px 0 15px;
        padding: 10px 15px;
        display: inline-block;
    }}

    /* IA (Izquierda - Azul) */
    [data-testid="stChatMessage"]:nth-child(even) {{
        flex-direction: row !important;
        text-align: left !important;
    }}
    [data-testid="stChatMessage"]:nth-child(even) .stMarkdown {{
        background-color: {COLOR_IA} !important;
        text-align: left !important;
        border-radius: 15px 15px 15px 0;
        padding: 10px 15px;
        display: inline-block;
    }}

    /* 4. ELEMENTOS EXTRA (KPIs y Tablas) */
    .kpi-box {{ background: {COLOR_IA}; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid {COLOR_TEXTO}; margin: 10px 0; }}
    .kpi-value {{ font-family: 'Sora'; font-size: 3rem; font-weight: bold; color: {COLOR_TEXTO}; }}
    
    /* Ocultar footer */
    footer {{display: none;}}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# LÓGICA DEL SISTEMA (INTACTA)
# ==========================================

# --- 2. SEGURIDAD ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "Kinesis2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Puse el logo también aquí para que se vea bonito al entrar
        if os.path.exists("Logo Kinesis_Negativo.png"):
            st.image("Logo Kinesis_Negativo.png", width=120)
        st.markdown("<h1 style='text-align: center;'>Kinesis Nexus</h1>", unsafe_allow_html=True)
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

# --- HEADER CON LOGO Y TÍTULO ---
col1, col2 = st.columns([1, 8])
with col1:
    if os.path.exists("Logo Kinesis_Negativo.png"):
        st.image("Logo Kinesis_Negativo.png", width=80)
with col2:
    st.markdown('<div style="margin-top: 15px; font-size: 20px; font-family: Sora; font-weight: bold;">Sistema de Inteligencia Generativa</div>', unsafe_allow_html=True)

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

# --- 5. FUNCIONES DE IA (INTACTAS) ---
def get_ai_response(prompt, context="", df_data=None):
    if df_data is not None:
        sys_msg = "Eres un analista experto. Resume los datos en una frase natural y breve. Sé directo."
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

# MENSAJE DE BIENVENIDA (SOLO SI EL CHAT ESTÁ VACÍO)
if len(st.session_state.messages) == 0:
    st.markdown(f"""
    <div style="text-align: center; margin-top: 40px; font-size: 1.2rem;">
        <b>¡Hola, ChangeLabiano!</b><br>
        Soy el experto en el proyecto Kinesis. ¿En qué te puedo ayudar hoy?
    </div>
    """, unsafe_allow_html=True)

# MOSTRAR HISTORIAL
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "viz" in m:
            if m["viz_type"] == "chart": st.plotly_chart(m["viz"], use_container_width=True)
            elif m["viz_type"] == "kpi": st.markdown(f'<div class="kpi-box"><div class="kpi-value">{m["viz"]}</div></div>', unsafe_allow_html=True)

# ENTRADA DE USUARIO
if user_input := st.chat_input("¿Qué quieres consultar hoy?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Buscar en PDF
        docs = v_db.similarity_search(user_input, k=2)
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
                    # AQUÍ SOLO CAMBIÉ LA PALETA DE COLORES AL GRAFICO PARA QUE COINCIDA
                    fig = px.bar(df_res, x=df_res.columns[0], y=df_res.columns[1], 
                                 template="plotly_dark", 
                                 color_discrete_sequence=[COLOR_USUARIO, COLOR_IA, COLOR_TEXTO])
                    # Fondo del gráfico transparente para que se vea el azul de la app
                    fig.update_layout(paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO, font_family="Manrope")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    msg_data.update({"viz": fig, "viz_type": "chart"})
                
                st.session_state.messages.append(msg_data)
                
            except Exception as e:
                st.error("Tuve un problema técnico procesando esos datos.")
        else:
            st.markdown(initial_res)
            st.session_state.messages.append({"role": "assistant", "content": initial_res})