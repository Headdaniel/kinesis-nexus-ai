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
DB_PATH = "data/vectors"
CSV_FILE = "data/raw/Base_maestra_kinesis.csv"

# Estilo CSS para modo oscuro total y centrado
# Reemplaza tu bloque de estilo CSS por este:
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;700&family=Sora:wght@600;800&display=swap');

    /* Fondo y texto general */
    .stApp {{ 
        background-color: #002f6cff !important; 
    }}
    
    html, body, [class*="css"] {{
        font-family: 'Manrope', sans-serif !important;
        color: #c9e0ffff !important;
    }}

    /* Títulos y Bienvenida en Sora */
    h1, h2, h3, .sora-font {{
        font-family: 'Sora', sans-serif !important;
        font-weight: 800 !important;
        color: #c9e0ffff !important;
    }}

    /* Burbujas de chat: Alineación */
    .stChatMessage {{
        background-color: transparent !important;
    }}

    /* Mensajes del Usuario (Derecha) */
    [data-testid="stChatMessage"]:nth-child(odd) {{
        flex-direction: row-reverse !important;
        text-align: right !important;
    }}
    [data-testid="stChatMessage"]:nth-child(odd) .stMarkdown {{
        background-color: #ba0c2fff !important; /* Fucsia */
        padding: 10px 15px;
        border-radius: 15px 15px 0px 15px;
        display: inline-block;
    }}

    /* Mensajes de la IA (Izquierda) */
    [data-testid="stChatMessage"]:nth-child(even) {{
        text-align: left !important;
    }}
    [data-testid="stChatMessage"]:nth-child(even) .stMarkdown {{
        background-color: #002060ff !important; /* Azul encendido */
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0px;
        display: inline-block;
    }}

    /* Ajuste del logo y entrada de texto */
    [data-testid="stChatInput"] {{
        max-width: 800px;
        margin: 0 auto;
    }}
    </style>
    """, unsafe_allow_html=True)

# Header: Logo y Título
col_logo, col_vacio = st.columns([1, 4])
with col_logo:
    if os.path.exists("Logo Kinesis_Negativo.png"):
        st.image("Logo Kinesis_Negativo.png", width=120)

# Saludo de Bienvenida (Solo aparece si no hay mensajes)
if len(st.session_state.messages) == 0:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="text-align: center;">
            <h1 class="sora-font" style="color: #c9e0ffff; font-size: 2.5rem;">
                ¡Hola, ChangeLabiano!
            </h1>
            <p style="font-size: 1.2rem; font-family: 'Manrope'; color: #c9e0ffff;">
                Soy el experto en el proyecto Kinesis. ¿En qué te puedo ayudar hoy?
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- 4. CARGA DE DATOS (CON CACHÉ Y SPINNER) ---
@st.cache_resource
def load_all():
    with st.spinner("Cargando base de conocimientos..."):
        # Modelos para PDF
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        v_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        
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
        # 1. El Spinner crea el efecto de "IA escribiendo/pensando"
        with st.spinner("Consultando fuentes de Kinesis..."):
            # Buscar en PDF
            docs = v_db.similarity_search(user_input, k=2)
            context_text = "\n".join([d.page_content for d in docs])
            
            # Fuente por defecto (Metodología)
            fuente = "Documentación Metodológica (PDF)"
            
            # Obtener respuesta inicial
            initial_res = get_ai_response(user_input, context_text)
            
            if "SELECT" in initial_res.upper():
                try:
                    # Si entra aquí, la fuente cambia al CSV
                    fuente = "Base_maestra_kinesis.csv"
                    
                    sql_match = re.search(r'SELECT.*', initial_res.replace('\n', ' '), re.IGNORECASE)
                    query = sql_match.group(0).split('```')[0].strip().replace(" tu_tabla", " kinesis").replace(" tabla", " kinesis")
                    if not query.endswith(';'): query += ';'
                    
                    df_res = sql_db.execute(query).df()
                    narrativa = get_ai_response(user_input, df_data=df_res)
                    
                    # Mostramos respuesta + la fuente
                    st.markdown(narrativa)
                    st.markdown(f"<small>📚 **Fuente:** {fuente}</small>", unsafe_allow_html=True)
                    
                    msg_data = {"role": "assistant", "content": f"{narrativa}\n\n📚 **Fuente:** {fuente}"}
                    
                    if len(df_res) == 1 and len(df_res.columns) == 1:
                        val = df_res.iloc[0,0]
                        st.markdown(f'<div class="kpi-box"><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)
                        msg_data.update({"viz": val, "viz_type": "kpi"})
                    elif len(df_res) > 0:
                        # Gráfico con tus colores corporativos
                        fig = px.bar(df_res, x=df_res.columns[0], y=df_res.columns[1], 
                                     template="plotly_dark", 
                                     color_discrete_sequence=['#ba0c2fff', '#002060ff', '#002f6cff'])
                        
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                        msg_data.update({"viz": fig, "viz_type": "chart"})
                    
                    st.session_state.messages.append(msg_data)
                    
                except Exception as e:
                    st.error("Tuve un problema técnico procesando esos datos.")
            else:
                # Respuesta de texto (PDF) + su fuente
                st.markdown(initial_res)
                st.markdown(f"<small>📚 **Fuente:** {fuente}</small>", unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": f"{initial_res}\n\n📚 **Fuente:** {fuente}"})