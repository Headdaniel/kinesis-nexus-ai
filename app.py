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

# --- 1. CONFIGURACIÓN DE PÁGINA Y FUENTES ---
st.set_page_config(page_title="Kinesis AI Pro", page_icon="🧠", layout="wide")

# Colores Corporativos Kinesis
COLOR_FUCSIA = "#ba0c2fff"
COLOR_AZUL_ENCENDIDO = "#002060ff"
COLOR_AZUL_OPACO = "#002f6cff"
COLOR_CELESTE = "#c9e0ffff"

# Importar Google Fonts (Sora y Manrope) y CSS Personalizado
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;700&family=Sora:wght@400;700&display=swap');

    /* Aplicar tipografías */
    html, body, [class*="css"] {{
        font-family: 'Manrope', sans-serif;
        color: {COLOR_CELESTE} !important;
    }}
    
    h1, h2, h3, .title-text {{
        font-family: 'Sora', sans-serif !important;
        font-weight: 700 !important;
        color: {COLOR_CELESTE} !important;
    }}

    /* Fondo general */
    .stApp {{ background-color: #0f1116 !important; }}
    
    /* Área del chat */
    .block-container {{ max-width: 800px; padding-top: 1rem; }}
    
    /* Ocultar elementos extra */
    footer {{display: none;}}
    header {{visibility: hidden;}}

    /* Barra inferior (Input) */
    [data-testid="stBottom"] {{ background-color: #0f1116 !important; border-top: 1px solid {COLOR_AZUL_OPACO}; }}
    [data-testid="stChatInput"] {{ background-color: #0f1116 !important; }}
    [data-testid="stChatInput"] textarea {{ 
        background-color: #161b22 !important; 
        color: white !important; 
        border: 1px solid {COLOR_AZUL_OPACO} !important;
        font-family: 'Manrope', sans-serif;
    }}

    /* Burbujas de Chat - Usuario (Derecha) */
    [data-testid="stChatMessage"]:nth-child(odd) {{
        flex-direction: row-reverse;
        background-color: {COLOR_AZUL_ENCENDIDO} !important;
        border: none;
    }}
    
    /* Burbujas de Chat - IA (Izquierda) */
    [data-testid="stChatMessage"]:nth-child(even) {{
        background-color: #161b22 !important;
        border: 1px solid {COLOR_AZUL_OPACO};
    }}

    /* KPIs */
    .kpi-box {{
        background: {COLOR_AZUL_ENCENDIDO};
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        border-left: 5px solid {COLOR_FUCSIA};
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    .kpi-value {{ font-family: 'Sora', sans-serif; font-size: 3.5rem; font-weight: 800; color: white; }}
    
    /* Mensaje de bienvenida */
    .welcome-text {{
        text-align: center;
        color: {COLOR_CELESTE};
        margin-top: 50px;
        margin-bottom: 30px;
    }}
    .welcome-title {{
        font-size: 2.5rem;
        font-weight: bold;
        background: -webkit-linear-gradient(45deg, {COLOR_CELESTE}, {COLOR_FUCSIA});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. GENERADOR DEL MAPA DEL SISTEMA ---
def crear_mapa_sistema():
    contenido_mapa = """
    RADIOGRAFÍA DEL SISTEMA KINESIS NEXUS
    =====================================
    PROPÓSITO:
    Sistema de Inteligencia Generativa para el análisis de datos empresariales (CSV) 
    y consulta de metodología teórica (PDF/Vectores).

    ESTRUCTURA DE ARCHIVOS:
    1. app.py: Cerebro principal. Interfaz Streamlit, lógica de IA y conexión a datos.
    2. requirements.txt: Lista de ingredientes (librerías) para que funcione en la nube.
    3. .env: (Local) Guarda las llaves secretas como la API KEY de Groq.
    4. data/raw/Base_maestra_kinesis.csv: La base de datos cruda con información de empresas.
    5. data/vectors/: Memoria a largo plazo (ChromaDB) donde vive el conocimiento del PDF.
    
    FLUJO DE DATOS:
    Usuario -> Pregunta -> Identificación de Intención (SQL vs Texto) -> 
    Si es Dato -> DuckDB ejecuta SQL -> Gráfico Plotly -> Respuesta Natural.
    Si es Teoría -> Búsqueda Vectorial -> LLM Llama-3 -> Respuesta Natural.
    """
    try:
        with open("Mapa del sistema.txt", "w", encoding="utf-8") as f:
            f.write(contenido_mapa)
    except Exception:
        pass # Si falla por permisos en la nube, no rompemos la app

crear_mapa_sistema()

# --- 3. SEGURIDAD ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "Kinesis2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Logo en login
        if os.path.exists("Logo Kinesis_Negativo.png"):
            st.image("Logo Kinesis_Negativo.png", width=200)
        st.markdown("<h1 style='text-align: center;'>Kinesis Nexus</h1>", unsafe_allow_html=True)
        st.text_input("Introduce la clave de acceso", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Clave incorrecta", type="password", on_change=password_entered, key="password")
        st.error("😕 Clave errónea.")
        return False
    return True

if not check_password():
    st.stop()

# --- 4. CARGA DE RECURSOS ---
load_dotenv()
API_KEY = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)
DB_PATH = "data/vectors"
CSV_FILE = "data/raw/Base_maestra_kinesis.csv"

@st.cache_resource
def load_all():
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
    st.error(f"Error de carga: {e}")
    st.stop()

# --- 5. LOGO HEADER ---
col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists("Logo Kinesis_Negativo.png"):
        st.image("Logo Kinesis_Negativo.png", use_container_width=True)
with col2:
    st.markdown('<div class="title-text" style="margin-top: 10px; font-size: 24px;">Sistema de Inteligencia Generativa</div>', unsafe_allow_html=True)

# --- 6. BIENVENIDA ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Detectar nombre de usuario (Experimental) o usar default
user_name = "ChangeLabiano"
# En Streamlit Cloud a veces se puede obtener del header, pero por privacidad usaremos el default o una variable de entorno si existiera.

if len(st.session_state.messages) == 0:
    st.markdown(f"""
    <div class="welcome-text">
        <div class="welcome-title">¡Hola, {user_name}!</div>
        <p style="font-size: 1.2rem;">Soy el experto en el proyecto Kinesis. ¿En qué te puedo ayudar hoy?</p>
    </div>
    """, unsafe_allow_html=True)

# --- 7. MOTOR DE IA ---
def get_ai_response(prompt, context="", df_data=None):
    if df_data is not None:
        sys_msg = "Eres un analista experto de Kinesis. Resume los datos en una frase natural y breve. NO muestres código Python ni SQL."
        content = f"Datos: {df_data.to_string()}\nPregunta: {prompt}"
    else:
        sys_msg = f"""Analista Kinesis. Tabla: 'kinesis'. 
        Si piden gráfico/cálculo: Responde SOLO SQL en ```sql.
        Si es teoría: Usa el contexto. NO inventes.
        Esquema: {esquema_cols}
        Contexto PDF: {context}"""
        content = prompt

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": content}],
        temperature=0.1
    )
    return res.choices[0].message.content

# --- 8. CHAT INTERACTIVO ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "viz" in m:
            if m["viz_type"] == "chart": st.plotly_chart(m["viz"], use_container_width=True)
            elif m["viz_type"] == "kpi": st.markdown(f'<div class="kpi-box"><div class="kpi-value">{m["viz"]}</div></div>', unsafe_allow_html=True)

if user_input := st.chat_input("Escribe tu consulta..."):
    # Mensaje usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    # Respuesta IA
    with st.chat_message("assistant"):
        with st.spinner("Analizando datos y fuentes..."): # Efecto de "escribiendo"
            
            # Buscar contexto PDF
            docs = v_db.similarity_search(user_input, k=2)
            context_text = "\n".join([d.page_content for d in docs])
            
            # Decisión inicial
            initial_res = get_ai_response(user_input, context_text)
            
            final_response = ""
            viz_obj = None
            viz_type = None
            source_file = ""

            if "SELECT" in initial_res.upper():
                try:
                    # Ruta de DATOS
                    source_file = "Base_maestra_kinesis.csv"
                    sql_match = re.search(r'SELECT.*', initial_res.replace('\n', ' '), re.IGNORECASE)
                    if sql_match:
                        query = sql_match.group(0).split('```')[0].strip().replace(" tu_tabla", " kinesis").replace(" tabla", " kinesis")
                        if not query.endswith(';'): query += ';'
                        
                        df_res = sql_db.execute(query).df()
                        
                        # Narrativa
                        narrativa = get_ai_response(user_input, df_data=df_res)
                        final_response = narrative = narrativa
                        
                        # Visualización
                        if len(df_res) == 1 and len(df_res.columns) == 1:
                            viz_obj = df_res.iloc[0,0]
                            viz_type = "kpi"
                        elif len(df_res) > 0:
                            # Gráfico con COLORES CORPORATIVOS
                            fig = px.bar(
                                df_res, 
                                x=df_res.columns[0], 
                                y=df_res.columns[1], 
                                template="plotly_dark",
                                color_discrete_sequence=[COLOR_FUCSIA, COLOR_AZUL_ENCENDIDO, COLOR_AZUL_OPACO]
                            )
                            fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_family="Manrope")
                            viz_obj = fig
                            viz_type = "chart"
                    else:
                        final_response = initial_res
                        source_file = "Modelo de Lenguaje General"
                except Exception:
                    final_response = "Disculpa, no pude procesar el cálculo. Intenta reformular la pregunta."
                    source_file = "Error del Sistema"
            else:
                # Ruta de TEXTO (PDF)
                final_response = initial_res
                # Intentamos sacar el nombre del archivo si existe metadata, sino genérico
                if docs:
                    fuentes = set([d.metadata.get('source', 'Documentación Kinesis') for d in docs])
                    # Limpiamos ruta completa para dejar solo el nombre
                    nombres_archivos = [os.path.basename(f) for f in fuentes]
                    source_file = ", ".join(nombres_archivos)
                else:
                    source_file = "Conocimiento General Kinesis"

            # Renderizar respuesta
            st.markdown(final_response)
            
            if viz_obj:
                if viz_type == "kpi":
                    st.markdown(f'<div class="kpi-box"><div class="kpi-value">{viz_obj}</div></div>', unsafe_allow_html=True)
                else:
                    st.plotly_chart(viz_obj, use_container_width=True)
            
            # Pie de página con la fuente
            st.markdown(f"<p style='font-size: 0.8rem; color: #8b949e !important; margin-top: 10px;'>📚 Esta respuesta te la doy revisando esta fuente: {source_file}</p>", unsafe_allow_html=True)

            # Guardar en historial
            msg_data = {"role": "assistant", "content": final_response}
            if viz_obj: 
                msg_data["viz"] = viz_obj
                msg_data["viz_type"] = viz_type
            st.session_state.messages.append(msg_data)