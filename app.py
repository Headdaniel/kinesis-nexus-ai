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
COLOR_BUBBLE = "#002060ff"     # Azul Encendido (Burbujas)
COLOR_FUCSIA = "#ba0c2fff"     # Fucsia (Detalles)

# --- 2. GENERADOR DEL MAPA DEL SISTEMA (Feature solicitado previamente) ---
def crear_mapa_sistema():
    contenido_mapa = """
    RADIOGRAFÍA DEL SISTEMA KINESIS NEXUS
    =====================================
    1. app.py: Lógica principal, interfaz y cerebro de IA.
    2. requirements.txt: Dependencias para la nube.
    3. data/raw/Base_maestra_kinesis.csv: Fuente de datos tabular.
    4. data/vectors/: Memoria vectorial del PDF.
    """
    try:
        with open("Mapa del sistema.txt", "w", encoding="utf-8") as f:
            f.write(contenido_mapa)
    except: pass
crear_mapa_sistema()

# --- 3. CSS PERSONALIZADO (Ajustado a tus requerimientos exactos) ---
st.markdown(f"""
    <style>
    /* Importar fuentes */
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;700&family=Sora:wght@400;700&display=swap');

    /* 1. FONDO Y TEXTO GENERAL */
    .stApp {{ background-color: {COLOR_FONDO} !important; }}
    html, body, p, li, span, label, h1, h2, h3, div {{
        font-family: 'Manrope', sans-serif;
        color: {COLOR_TEXTO} !important;
    }}
    
    /* Títulos en Sora */
    h1, h2, h3 {{ font-family: 'Sora', sans-serif !important; }}

    /* 2. BARRA DE INPUT (LIMITADA AL ANCHO DEL CHAT) */
    [data-testid="stChatInput"] {{
        max-width: 800px !important;
        margin: 0 auto !important;
        background-color: {COLOR_FONDO} !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background-color: {COLOR_BUBBLE} !important;
        color: {COLOR_TEXTO} !important;
        border: 1px solid {COLOR_TEXTO} !important;
    }}
    
    /* Ocultar elementos extra */
    footer {{display: none;}}
    [data-testid="stBottom"] {{ background-color: {COLOR_FONDO} !important; border-top: 1px solid {COLOR_BUBBLE}; }}

    /* 3. ALINEACIÓN DE CHATS (DERECHA / IZQUIERDA) */
    .stChatMessage {{
        background-color: transparent !important;
        border: none !important;
    }}
    
    /* Mensajes del USUARIO (Derecha) */
    [data-testid="stChatMessage"]:nth-child(odd) {{
        flex-direction: row-reverse;
        text-align: right;
    }}
    [data-testid="stChatMessage"]:nth-child(odd) .stMarkdown {{
        background-color: {COLOR_FUCSIA};
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        display: inline-block;
        text-align: right;
    }}

    /* Mensajes de la IA (Izquierda) */
    [data-testid="stChatMessage"]:nth-child(even) {{
        flex-direction: row;
        text-align: left;
    }}
    [data-testid="stChatMessage"]:nth-child(even) .stMarkdown {{
        background-color: {COLOR_BUBBLE};
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        display: inline-block;
        text-align: left;
    }}

    /* 4. MENSAJE DE BIENVENIDA */
    .welcome-text {{
        text-align: center;
        color: {COLOR_TEXTO};
        margin-top: 40px;
        font-size: 1.5rem;
        font-family: 'Sora', sans-serif;
    }}

    /* 5. KPI BOX */
    .kpi-box {{
        background: {COLOR_BUBBLE};
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
            st.image("Logo Kinesis_Negativo.png", width=120)
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

# --- 5. CARGA DE RECURSOS ---
load_dotenv()
API_KEY = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)
DB_PATH = "data/vectors"
CSV_FILE = "data/raw/Base_maestra_kinesis.csv"

@st.cache_resource
def load_all():
    # Spinner personalizado para no romper estética
    with st.spinner("Conectando con el núcleo Kinesis..."):
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
    st.error(f"Error crítico: {e}")
    st.stop()

# --- 6. LOGO Y HEADER (Logo reducido y centrado en móvil) ---
# Usamos columnas para posicionar en web, en móvil se apilan y centramos la imagen
col_logo, col_title = st.columns([1, 6])
with col_logo:
    if os.path.exists("Logo Kinesis_Negativo.png"):
        # Width 120 es pequeño, ideal para móvil y web elegante
        st.image("Logo Kinesis_Negativo.png", width=120)
with col_title:
    st.markdown('<div style="margin-top: 20px; font-size: 20px; font-family: Sora;">Sistema de Inteligencia Generativa</div>', unsafe_allow_html=True)

# --- 7. BIENVENIDA ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    # Mensaje plano con color #c9e0ffff como pediste
    st.markdown("""
    <div class="welcome-text">
        <b>¡Hola, ChangeLabiano!</b><br>
        Soy el experto en el proyecto Kinesis. ¿En qué te puedo ayudar hoy?
    </div>
    """, unsafe_allow_html=True)

# --- 8. LÓGICA DE IA (RESTAURADA A LA VERSIÓN QUE FUNCIONA) ---
def get_ai_response(prompt, context="", df_data=None):
    # Usamos la lógica del código anterior que sí funcionaba
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

# --- 9. CHAT INTERACTIVO ---
# Mostrar historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "viz" in m:
            if m["viz_type"] == "chart": st.plotly_chart(m["viz"], use_container_width=True)
            elif m["viz_type"] == "kpi": st.markdown(f'<div class="kpi-box"><div class="kpi-value">{m["viz"]}</div></div>', unsafe_allow_html=True)

# Input del usuario
if user_input := st.chat_input("Escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."): # Efecto de escritura
            
            # 1. Búsqueda Vectorial (PDF)
            docs = v_db.similarity_search(user_input, k=2)
            context_text = "\n".join([d.page_content for d in docs])
            
            # Fuente por defecto
            fuente_respuesta = "Conocimiento General Kinesis"
            if docs:
                fuente_respuesta = "Documentación Metodológica (PDF)"

            # 2. Primera consulta a la IA
            initial_res = get_ai_response(user_input, context_text)
            
            # 3. Lógica de Decisión (Restaurada del código funcional)
            if "SELECT" in initial_res.upper():
                try:
                    # Extracción SQL robusta (Regex del código viejo)
                    sql_match = re.search(r'SELECT.*', initial_res.replace('\n', ' '), re.IGNORECASE)
                    if sql_match:
                        query = sql_match.group(0).split('```')[0].strip()
                        query = query.replace(" tu_tabla", " kinesis").replace(" tabla", " kinesis")
                        if not query.endswith(';'): query += ';'
                        
                        # Ejecutar SQL
                        df_res = sql_db.execute(query).df()
                        fuente_respuesta = "Base_maestra_kinesis.csv"
                        
                        # Narrativa
                        narrativa = get_ai_response(user_input, df_data=df_res)
                        
                        # Renderizar Respuesta
                        st.markdown(narrativa)
                        st.markdown(f"**📚 Fuente:** {fuente_respuesta}")
                        
                        msg_data = {"role": "assistant", "content": f"{narrativa}\n\n**📚 Fuente:** {fuente_respuesta}"}
                        
                        # Visualización
                        if len(df_res) == 1 and len(df_res.columns) == 1:
                            val = df_res.iloc[0,0]
                            st.markdown(f'<div class="kpi-box"><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)
                            msg_data.update({"viz": val, "viz_type": "kpi"})
                        elif len(df_res) > 0:
                            # Gráfico con COLORES OFICIALES
                            fig = px.bar(
                                df_res, 
                                x=df_res.columns[0], 
                                y=df_res.columns[1], 
                                template="plotly_dark", 
                                color_discrete_sequence=[COLOR_FUCSIA, COLOR_BUBBLE, COLOR_TEXTO]
                            )
                            # Fondo del gráfico igual al de la app
                            fig.update_layout(paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO, font_family="Manrope")
                            st.plotly_chart(fig, use_container_width=True)
                            msg_data.update({"viz": fig, "viz_type": "chart"})
                        
                        st.session_state.messages.append(msg_data)
                    else:
                        # Si falló el regex pero parecía SQL
                        st.markdown(initial_res)
                        st.session_state.messages.append({"role": "assistant", "content": initial_res})
                
                except Exception as e:
                    # Si falla el SQL, mostramos error sutil
                    error_msg = "Tuve un problema calculando esos datos, por favor reformula la pregunta."
                    st.markdown(error_msg)
            else:
                # Respuesta de Texto (PDF)
                st.markdown(initial_res)
                st.markdown(f"**📚 Fuente:** {fuente_respuesta}")
                st.session_state.messages.append({"role": "assistant", "content": f"{initial_res}\n\n**📚 Fuente:** {fuente_respuesta}"})