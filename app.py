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

# Colores del Proyecto
COLOR_FONDO = "#002f6cff"      # Azul Opaco (Fondo General)
COLOR_TEXTO = "#c9e0ffff"      # Celeste (Texto General)
COLOR_BUBBLE_USER = "#ba0c2fff" # Fucsia (Usuario)
COLOR_BUBBLE_AI = "#002060ff"   # Azul Encendido (IA)

# --- 2. GENERADOR DEL MAPA DEL SISTEMA ---
def crear_mapa_sistema():
    contenido_mapa = """
    RADIOGRAFÍA DEL SISTEMA KINESIS NEXUS
    =====================================
    1. app.py: Lógica principal.
    2. requirements.txt: Dependencias.
    3. data/raw/Base_maestra_kinesis.csv: Datos.
    4. data/vectors/: Memoria PDF.
    """
    try:
        with open("Mapa del sistema.txt", "w", encoding="utf-8") as f:
            f.write(contenido_mapa)
    except: pass
crear_mapa_sistema()

# --- 3. CSS PERSONALIZADO (Corrigiendo Input y Colores) ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;700&family=Sora:wght@400;700&display=swap');

    /* 1. FONDO Y TEXTO GENERAL */
    .stApp {{ background-color: {COLOR_FONDO} !important; }}
    
    html, body, p, li, span, label, h1, h2, h3, div, .stMarkdown {{
        font-family: 'Manrope', sans-serif;
        color: {COLOR_TEXTO} !important;
    }}
    
    h1, h2, h3 {{ font-family: 'Sora', sans-serif !important; }}

    /* 2. BARRA DE INPUT (CORREGIDA: ANCHO LIMITADO) */
    [data-testid="stChatInput"] {{
        max-width: 800px !important;
        margin: 0 auto !important;
        background-color: {COLOR_FONDO} !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background-color: {COLOR_BUBBLE_AI} !important;
        color: {COLOR_TEXTO} !important;
        border: 1px solid {COLOR_TEXTO} !important;
    }}
    
    /* Ocultar elementos extra */
    footer {{display: none;}}
    [data-testid="stBottom"] {{ background-color: {COLOR_FONDO} !important; border-top: 1px solid {COLOR_BUBBLE_AI}; }}

    /* 3. ALINEACIÓN DE CHATS */
    .stChatMessage {{ background-color: transparent !important; border: none !important; }}
    
    /* Usuario (Derecha - Fucsia) */
    [data-testid="stChatMessage"]:nth-child(odd) {{ flex-direction: row-reverse; text-align: right; }}
    [data-testid="stChatMessage"]:nth-child(odd) .stMarkdown {{
        background-color: {COLOR_BUBBLE_USER} !important;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        display: inline-block;
        text-align: right;
    }}

    /* IA (Izquierda - Azul) */
    [data-testid="stChatMessage"]:nth-child(even) {{ flex-direction: row; text-align: left; }}
    [data-testid="stChatMessage"]:nth-child(even) .stMarkdown {{
        background-color: {COLOR_BUBBLE_AI} !important;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        display: inline-block;
        text-align: left;
    }}

    /* 4. BIENVENIDA (Color corregido) */
    .welcome-text {{
        text-align: center;
        color: {COLOR_TEXTO} !important;
        margin-top: 40px;
        font-size: 1.5rem;
        font-family: 'Sora', sans-serif;
    }}

    /* 5. KPI BOX */
    .kpi-box {{
        background: {COLOR_BUBBLE_AI};
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid {COLOR_TEXTO};
        margin: 10px 0;
    }}
    .kpi-value {{ font-family: 'Sora'; font-size: 3rem; font-weight: bold; color: {COLOR_TEXTO}; }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. SEGURIDAD ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "Kinesis2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Logo pequeño en login
        if os.path.exists("Logo Kinesis_Negativo.png"):
            st.image("Logo Kinesis_Negativo.png", width=100)
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

# --- 5. CARGA DE RECURSOS (Restaurado EXACTAMENTE de tu código funcional) ---
load_dotenv()
API_KEY = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)
DB_PATH = "data/vectors"
CSV_FILE = "data/raw/Base_maestra_kinesis.csv"

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

# --- 6. LOGO HEADER (Pequeño y centrado/arriba) ---
col1, col2 = st.columns([1, 8])
with col1:
    if os.path.exists("Logo Kinesis_Negativo.png"):
        st.image("Logo Kinesis_Negativo.png", width=80) # Reducido aún más como pediste
with col2:
    st.markdown('<div style="margin-top: 15px; font-size: 20px; font-family: Sora;">Sistema de Inteligencia Generativa</div>', unsafe_allow_html=True)

# --- 7. BIENVENIDA (Texto plano y color corregido) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.markdown(f"""
    <div class="welcome-text">
        <b>¡Hola, ChangeLabiano!</b><br>
        Soy el experto en el proyecto Kinesis. ¿En qué te puedo ayudar hoy?
    </div>
    """, unsafe_allow_html=True)

# --- 8. FUNCIONES DE IA (RESTAURADO DEL CÓDIGO ORIGINAL QUE SÍ FUNCIONABA) ---
def get_ai_response(prompt, context="", df_data=None):
    if df_data is not None:
        sys_msg = "Eres un analista experto. Resume los datos en una frase natural y breve. Sé directo."
        content = f"Datos obtenidos: {df_data.to_string()}\nPregunta: {prompt}"
    else:
        # ESTE ES EL PROMPT CLAVE QUE FALTABA
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

# --- 9. CHAT INTERACTIVO (Lógica original + Colores nuevos) ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "viz" in m:
            if m["viz_type"] == "chart": st.plotly_chart(m["viz"], use_container_width=True)
            elif m["viz_type"] == "kpi": st.markdown(f'<div class="kpi-box"><div class="kpi-value">{m["viz"]}</div></div>', unsafe_allow_html=True)

if user_input := st.chat_input("Escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."): # Efecto escribiendo
            # 1. Buscar en PDF
            docs = v_db.similarity_search(user_input, k=2)
            context_text = "\n".join([d.page_content for d in docs])
            
            # Fuente por defecto
            fuente = "Documentación Kinesis (PDF)" if docs else "Conocimiento General"

            # 2. Obtener respuesta inicial (SQL o Texto)
            initial_res = get_ai_response(user_input, context_text)
            
            if "SELECT" in initial_res.upper():
                try:
                    # 3. Extraer y limpiar SQL (Regex del código original)
                    sql_match = re.search(r'SELECT.*', initial_res.replace('\n', ' '), re.IGNORECASE)
                    if sql_match:
                        query = sql_match.group(0).split('```')[0].strip()
                        query = query.replace(" tu_tabla", " kinesis").replace(" tabla", " kinesis")
                        if not query.endswith(';'): query += ';'
                        
                        # Ejecutar
                        df_res = sql_db.execute(query).df()
                        fuente = "Base_maestra_kinesis.csv"
                        
                        # Narrativa
                        narrativa = get_ai_response(user_input, df_data=df_res)
                        
                        # Mostrar texto
                        st.markdown(narrativa)
                        st.markdown(f"**📚 Fuente:** {fuente}")
                        
                        msg_data = {"role": "assistant", "content": f"{narrativa}\n\n**📚 Fuente:** {fuente}"}
                        
                        # Visualización (Con tus colores nuevos)
                        if len(df_res) == 1 and len(df_res.columns) == 1:
                            val = df_res.iloc[0,0]
                            st.markdown(f'<div class="kpi-box"><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)
                            msg_data.update({"viz": val, "viz_type": "kpi"})
                        elif len(df_res) > 0:
                            # Gráfico de barras con tus colores
                            fig = px.bar(df_res, x=df_res.columns[0], y=df_res.columns[1], 
                                         template="plotly_dark", 
                                         color_discrete_sequence=[COLOR_BUBBLE_USER, COLOR_BUBBLE_AI, COLOR_TEXTO])
                            
                            # Ajuste de fondo del gráfico para que coincida con la app
                            fig.update_layout(paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO, font_family="Manrope")
                            
                            st.plotly_chart(fig, use_container_width=True)
                            msg_data.update({"viz": fig, "viz_type": "chart"})
                        
                        st.session_state.messages.append(msg_data)
                    else:
                        st.markdown(initial_res)
                        st.session_state.messages.append({"role": "assistant", "content": initial_res})
                        
                except Exception as e:
                    # Si falla el cálculo, mostramos el error técnico en logs pero mensaje suave al usuario
                    print(e) 
                    st.error("No pude generar el gráfico. Intenta reformular la pregunta.")
            else:
                st.markdown(initial_res)
                st.markdown(f"**📚 Fuente:** {fuente}")
                st.session_state.messages.append({"role": "assistant", "content": f"{initial_res}\n\n**📚 Fuente:** {fuente}"})