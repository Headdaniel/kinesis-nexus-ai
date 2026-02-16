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

# --- 1. CONFIGURACIÓN DE SEGURIDAD (PASSWORD) ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "Kinesis2026": # <--- CLAVE PARA TUS SOCIOS
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 Acceso Restringido")
        st.text_input("Introduce la clave de Kinesis Nexus", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 Acceso Restringido")
        st.text_input("Clave incorrecta. Intenta de nuevo", type="password", on_change=password_entered, key="password")
        st.error("😕 Ups, esa no es la clave.")
        return False
    return True

# Si la contraseña no es correcta, detenemos la ejecución aquí
if not check_password():
    st.stop()

# --- 2. EL RESTO DE TU CÓDIGO (IA, DATOS Y DISEÑO) ---
load_dotenv()
API_KEY = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)
DB_PATH = "data/vectors"
CSV_FILE = "data/raw/Base_maestra_kinesis.csv"

# (Aquí sigue todo el código de diseño CSS, carga de datos y lógica del chat que ya tienes)
# ... [EL RESTO DEL CÓDIGO QUE YA PROBAMOS] ...