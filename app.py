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

# --- 2. CSS DE DISEÑO (ESTO ES SOLO FORMA, NO TOCA LÓGICA) ---
st.markdown("""
    <style>
    /* Importar Fuentes */
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;700&family=Sora:wght@400;700&display=swap');

    /* FONDO Y TEXTOS GLOBALES */
    .stApp { background-color: #002f6cff !important; }
    
    html, body, p, li, span, label, h1, h2, h3, div, .stMarkdown {
        font-family: 'Manrope', sans-serif !important;
        color: #c9e0ffff !important;
    }
    h1, h2, h3 { font-family: 'Sora', sans-serif !important; }

    /* INPUT DE TEXTO */
    [data-testid="stChatInput"] {
        max-width: 800px !important;
        margin: 0 auto !important;
        background-color: #002f6cff !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #002f6cff !important;
        color: #c9e0ffff !important;
        border: 1px solid #c9e0ffff !important;
    }

    /* MENSAJES DE CHAT (DERECHA/IZQUIERDA) */
    [data-testid="stChatMessage"]:nth-child(odd) {
        flex-direction: row-reverse !important;
        text-align: right !important;
    }
    [data-testid="stChatMessage"]:nth-child(even) {
        flex-direction: row !important;
        text-align: left !important;
    }

    /* KPI BOX */
    .kpi-box { 
        background: #002060ff; 
        padding: 25px; 
        border-radius: 12px; 
        text-align: center; 
        border: 2px solid #ba0c2fff; 
        margin: 15px 0; 
    }
    .kpi-value { 
        font-family: 'Sora', sans-serif; 
        font-size: 3.8rem; 
        font-weight: 800; 
        color: #c9e0ffff; 
    }
    
    /* Ocultar elementos extra */
    footer {display: none;}
    [data-testid="stBottomBlockContainer"] { background-color: #002f6cff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SEGURIDAD ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "Kinesis2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<h1 style='text-align: center;'>Kinesis Nexus</h1>", unsafe_allow_html=True)
        st.text_input("Introduce la clave de acceso", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Clave incorrecta", type="password", on_change=password_entered, key="password")
        st.error("😕 Ups, esa no es la clave.")
        return False
    return True

if not check_password():
    st.stop()

# --- 4. RECURSOS ---
load_dotenv()
API_KEY = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)
DB_PATH = "data/vectors"
CSV_FILE = "data/raw/Base_maestra_kinesis.csv"

# --- LOGO PEQUEÑO ---
if os.path.exists("Logo Kinesis_Negativo.png"):
    col1, col2 = st.columns([1, 10]) 
    with col1:
        st.image("Logo Kinesis_Negativo.png", width=80)

# --- CARGA DE DATOS ---
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
    st.error(f"Error crítico: {e}")
    st.stop()

# --- 5. FUNCIONES DE IA (TU LÓGICA ORIGINAL INTACTA) ---
def get_ai_response(prompt, context="", df_data=None):
    if df_data is not None:
        sys_msg = "Eres un analista experto. Resume los datos en una frase natural y breve. Sé directo."
        content = f"Datos obtenidos: {df_data.to_string()}\nPregunta: {prompt}"
    else:
        # ESTE ES EL PROMPT QUE FUNCIONA, NO LO TOQUÉ
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

# --- 6. CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# BIENVENIDA ESTILO GEMINI
if len(st.session_state.messages) == 0:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="font-family: 'Sora'; font-size: 2.5rem; color: #c9e0ffff;">¡Hola, ChangeLabiano!</h1>
        <p style="font-size: 1.2rem; color: #c9e0ffff;">Soy el experto en el proyecto Kinesis. ¿En qué te puedo ayudar hoy?</p>
    </div>
    """, unsafe_allow_html=True)

# HISTORIAL
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "viz" in m:
            if m["viz_type"] == "chart": st.plotly_chart(m["viz"], use_container_width=True)
            elif m["viz_type"] == "kpi": st.markdown(f'<div class="kpi-box"><div class="kpi-value">{m["viz"]}</div></div>', unsafe_allow_html=True)

# INPUT USUARIO
if user_input := st.chat_input("¿Qué quieres consultar hoy?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Efecto de escribiendo...
        with st.spinner("Analizando..."):
            docs = v_db.similarity_search(user_input, k=2)
            context_text = "\n".join([d.page_content for d in docs])
            
            # 1. Obtener respuesta cruda (SQL o Texto)
            initial_res = get_ai_response(user_input, context_text)
            
            # Texto de fuente por defecto
            fuente_str = "\n\n**Esta respuesta te la doy revisando esta fuente:** Documentación Metodológica (PDF)"
            
            if "SELECT" in initial_res.upper():
                try:
                    sql_match = re.search(r'SELECT.*', initial_res.replace('\n', ' '), re.IGNORECASE)
                    query = sql_match.group(0).split('```')[0].strip()
                    query = query.replace(" tu_tabla", " kinesis").replace(" tabla", " kinesis")
                    if not query.endswith(';'): query += ';'
                    
                    df_res = sql_db.execute(query).df()
                    
                    narrativa = get_ai_response(user_input, df_data=df_res)
                    
                    # Actualizar fuente a CSV
                    fuente_str = "\n\n**Esta respuesta te la doy revisando esta fuente:** Base_maestra_kinesis.csv"
                    
                    # Mostrar Narrativa + Fuente
                    st.markdown(narrativa + fuente_str)
                    
                    msg_data = {"role": "assistant", "content": narrativa + fuente_str}
                    
                    if len(df_res) == 1 and len(df_res.columns) == 1:
                        val = df_res.iloc[0,0]
                        st.markdown(f'<div class="kpi-box"><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)
                        msg_data.update({"viz": val, "viz_type": "kpi"})
                    elif len(df_res) > 0:
                        # Gráfico con COLORES CORPORATIVOS
                        fig = px.bar(df_res, x=df_res.columns[0], y=df_res.columns[1], 
                                     template="plotly_dark", 
                                     color_discrete_sequence=['#ba0c2fff', '#002060ff', '#002f6cff']) # Fucsia, Azul, Azul Opaco
                        
                        # Fondo transparente para integrar con la app
                        fig.update_layout(paper_bgcolor="#002f6cff", plot_bgcolor="#002f6cff", font_family="Manrope", font_color="#c9e0ffff")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        msg_data.update({"viz": fig, "viz_type": "chart"})
                    
                    st.session_state.messages.append(msg_data)
                    
                except Exception as e:
                    st.error("Tuve un problema técnico procesando esos datos.")
            else:
                # Mostrar respuesta texto + Fuente
                st.markdown(initial_res + fuente_str)
                st.session_state.messages.append({"role": "assistant", "content": initial_res + fuente_str})