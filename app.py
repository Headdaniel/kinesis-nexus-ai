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

# --- 2. MAPA DEL SISTEMA ---
def crear_mapa_sistema():
    contenido = "RADIOGRAFÍA KINESIS NEXUS\n1. app.py: Cerebro.\n2. data/: Datos y Vectores."
    try:
        with open("Mapa del sistema.txt", "w", encoding="utf-8") as f: f.write(contenido)
    except: pass
crear_mapa_sistema()

# --- 3. CSS (CORREGIDO: ALINEACIÓN Y TABLAS) ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;700&family=Sora:wght@400;700&display=swap');

    /* ESTILOS GENERALES */
    .stApp {{ background-color: {COLOR_FONDO} !important; }}
    html, body, p, li, span, label, h1, h2, h3, div, .stMarkdown {{
        font-family: 'Manrope', sans-serif;
        color: {COLOR_TEXTO} !important;
    }}
    h1, h2, h3 {{ font-family: 'Sora', sans-serif !important; }}

    /* INPUT DEL CHAT (Ancho limitado) */
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

    /* ARREGLO DE LA TABLA (FONDO BLANCO A OSCURO) */
    [data-testid="stDataFrame"] {{
        background-color: {COLOR_FONDO} !important;
    }}
    [data-testid="stDataFrame"] div {{
        background-color: {COLOR_FONDO} !important;
        color: {COLOR_TEXTO} !important;
    }}
    [data-testid="stTable"] {{
        background-color: {COLOR_FONDO} !important;
        color: {COLOR_TEXTO} !important;
    }}

    /* ALINEACIÓN DE CHATS (CORREGIDO) */
    .stChatMessage {{ background-color: transparent !important; border: none !important; }}

    /* USUARIO -> DERECHA (row-reverse) */
    [data-testid="stChatMessage"]:nth-child(odd) {{
        flex-direction: row-reverse !important; 
        text-align: right !important;
    }}
    [data-testid="stChatMessage"]:nth-child(odd) .stMarkdown {{
        background-color: {COLOR_BUBBLE_USER} !important;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        text-align: right !important;
        margin-left: auto; /* Empuja a la derecha */
    }}

    /* IA -> IZQUIERDA (row) */
    [data-testid="stChatMessage"]:nth-child(even) {{
        flex-direction: row !important;
        text-align: left !important;
    }}
    [data-testid="stChatMessage"]:nth-child(even) .stMarkdown {{
        background-color: {COLOR_BUBBLE_AI} !important;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        text-align: left !important;
        margin-right: auto; /* Empuja a la izquierda */
    }}

    /* OCULTAR COSAS EXTRA */
    footer {{display: none;}}
    [data-testid="stBottom"] {{ background-color: {COLOR_FONDO} !important; border-top: 1px solid {COLOR_BUBBLE_AI}; }}
    
    /* BIENVENIDA */
    .welcome-text {{ text-align: center; color: {COLOR_TEXTO} !important; margin-top: 40px; font-size: 1.5rem; font-family: 'Sora'; }}
    
    /* KPI BOX */
    .kpi-box {{ background: {COLOR_BUBBLE_AI}; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid {COLOR_TEXTO}; margin: 10px 0; }}
    .kpi-value {{ font-family: 'Sora'; font-size: 3rem; font-weight: bold; color: {COLOR_TEXTO}; }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. SEGURIDAD ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    def password_entered():
        if st.session_state["password"] == "Kinesis2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        if os.path.exists("Logo Kinesis_Negativo.png"):
            st.image("Logo Kinesis_Negativo.png", width=100)
        st.markdown("<h1 style='text-align: center;'>Kinesis Nexus</h1>", unsafe_allow_html=True)
        st.text_input("Introduce la clave de acceso", type="password", on_change=password_entered, key="password")
        return False
    return True

if not check_password(): st.stop()

# --- 5. CARGA DE RECURSOS ---
load_dotenv()
API_KEY = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)
DB_PATH = "data/vectors"
CSV_FILE = "data/raw/Base_maestra_kinesis.csv"

@st.cache_resource
def load_all():
    with st.spinner("Cargando cerebro Kinesis..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        v_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        con = duckdb.connect(database=':memory:')
        df = pd.read_csv(CSV_FILE)
        # Limpieza de columnas estándar
        df.columns = [re.sub(r'[^\w]', '_', c.lower().strip().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')) for c in df.columns]
        con.execute("CREATE TABLE kinesis AS SELECT * FROM df")
        esquema = con.execute("DESCRIBE kinesis").df()[['column_name', 'column_type']].to_string()
        return v_db, con, esquema

try:
    v_db, sql_db, esquema_cols = load_all()
except Exception as e:
    st.error(f"Error de carga: {e}")
    st.stop()

# --- 6. HEADER ---
col1, col2 = st.columns([1, 8])
with col1:
    if os.path.exists("Logo Kinesis_Negativo.png"): st.image("Logo Kinesis_Negativo.png", width=80)
with col2:
    st.markdown('<div style="margin-top: 15px; font-size: 20px; font-family: Sora;">Sistema de Inteligencia Generativa</div>', unsafe_allow_html=True)

# --- 7. BIENVENIDA ---
if "messages" not in st.session_state: st.session_state.messages = []
if len(st.session_state.messages) == 0:
    st.markdown(f"""<div class="welcome-text"><b>¡Hola, ChangeLabiano!</b><br>Soy el experto en el proyecto Kinesis. ¿En qué te puedo ayudar hoy?</div>""", unsafe_allow_html=True)

# --- 8. CEREBRO IA (LÓGICA CORREGIDA PARA FORZAR SQL) ---
def get_ai_response(prompt, context="", df_data=None):
    if df_data is not None:
        # FASE 2: NARRATIVA DE DATOS
        sys_msg = "Eres un analista de datos senior. Tu trabajo es interpretar los datos numéricos que recibes y dar una respuesta directa, profesional y breve en español. NO expliques el SQL."
        content = f"Datos resultantes: {df_data.to_string()}\nPregunta original: {prompt}"
    else:
        # FASE 1: GENERACIÓN DE SQL (PROMPT AGRESIVO)
        sys_msg = f"""
        ERES UN GENERADOR DE SQL DUCKDB PARA UNA TABLA LLAMADA 'kinesis'.
        
        TU OBJETIVO:
        1. Si el usuario pide contar, promediar, sumar, listar, graficar, comparar o analizar DATOS -> DEBES RESPONDER SOLO CON CÓDIGO SQL.
        2. NO ESCRIBAS TEXTO. NO DIGAS "AQUÍ ESTÁ EL GRÁFICO". NO DIBUJES GRÁFICOS ASCII CON ASTERISCOS.
        3. SOLO CÓDIGO SQL DENTRO DE ```sql ... ```.
        
        ESQUEMA DE LA TABLA 'kinesis':
        {esquema_cols}
        
        SI LA PREGUNTA ES TEÓRICA (NO DATOS):
        Usa este contexto PDF: {context}
        """
        content = prompt

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": content}],
        temperature=0.1
    )
    return res.choices[0].message.content

# --- 9. BUCLE DEL CHAT ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "viz" in m:
            if m["viz_type"] == "chart": st.plotly_chart(m["viz"], use_container_width=True)
            elif m["viz_type"] == "kpi": st.markdown(f'<div class="kpi-box"><div class="kpi-value">{m["viz"]}</div></div>', unsafe_allow_html=True)

if user_input := st.chat_input("Escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Procesando..."):
            # 1. Buscar Contexto
            docs = v_db.similarity_search(user_input, k=2)
            context_text = "\n".join([d.page_content for d in docs])
            
            # 2. Consultar IA
            initial_res = get_ai_response(user_input, context_text)
            
            # 3. Detectar SQL
            if "SELECT" in initial_res.upper() or "```sql" in initial_res.lower():
                try:
                    # Limpieza agresiva del SQL
                    sql_match = re.search(r'SELECT.*', initial_res.replace('\n', ' '), re.IGNORECASE)
                    if sql_match:
                        query = sql_match.group(0).split('```')[0].strip()
                        query = query.replace(" tu_tabla", " kinesis").replace(" tabla", " kinesis")
                        if not query.endswith(';'): query += ';'
                        
                        # Ejecutar Query
                        df_res = sql_db.execute(query).df()
                        
                        # Generar Narrativa
                        narrativa = get_ai_response(user_input, df_data=df_res)
                        
                        # Mostrar Resultado
                        st.markdown(narrativa)
                        st.markdown("**📚 Fuente:** Base_maestra_kinesis.csv")
                        msg_data = {"role": "assistant", "content": f"{narrativa}\n\n**📚 Fuente:** Base_maestra_kinesis.csv"}
                        
                        # Visualización
                        if len(df_res) == 1 and len(df_res.columns) == 1:
                            val = df_res.iloc[0,0]
                            st.markdown(f'<div class="kpi-box"><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)
                            msg_data.update({"viz": val, "viz_type": "kpi"})
                        elif len(df_res) > 0:
                            # Gráfico
                            fig = px.bar(df_res, x=df_res.columns[0], y=df_res.columns[1], 
                                         template="plotly_dark", 
                                         color_discrete_sequence=[COLOR_BUBBLE_USER, COLOR_BUBBLE_AI, COLOR_TEXTO])
                            fig.update_layout(paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO, font_family="Manrope")
                            st.plotly_chart(fig, use_container_width=True)
                            msg_data.update({"viz": fig, "viz_type": "chart"})
                        else:
                            # Si la tabla está vacía o es compleja, mostrar tabla corregida
                            st.dataframe(df_res)

                        st.session_state.messages.append(msg_data)
                    else:
                        # Fallo de regex
                        st.markdown(initial_res)
                        st.session_state.messages.append({"role": "assistant", "content": initial_res})
                except Exception as e:
                    # Si falla el SQL, a veces la IA genera texto explicativo, lo mostramos como fallback
                    st.markdown(initial_res) 
                    st.session_state.messages.append({"role": "assistant", "content": initial_res})
            else:
                # Respuesta Teórica
                st.markdown(initial_res)
                st.markdown("**📚 Fuente:** Documentación Kinesis (PDF)")
                st.session_state.messages.append({"role": "assistant", "content": f"{initial_res}\n\n**📚 Fuente:** Documentación Kinesis (PDF)"})