import os
import json
import duckdb
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -------------------------
# SCHEMA ROUTER (IMPORTS PROTEGIDOS)
# -------------------------
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    _SCHEMA_ROUTER_AVAILABLE = True
except Exception:
    _SCHEMA_ROUTER_AVAILABLE = False

from pathlib import Path

# -------------------------
# CONFIGURACIÓN
# -------------------------
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)
CSV_FILE = "data/raw/Base_maestra_kinesis.csv"
DICT_CSV_FILE = "data/raw/diccionario_kinesis.csv"

# Vector DB para schema routing (column dictionary)
SCHEMA_VDB_PATH = "data/vectors_schema_router"
SCHEMA_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SCHEMA_TOP_K = 15  # entre 10 y 20 como pediste

ALLOWED_AGGREGATIONS = ["sum", "count", "avg"]
ALLOWED_TIPOS_CONSULTA = ["comparativo_metricas", "agregacion_simple", "distribucion"]
ALLOWED_TIPOS_GRAFICO = ["bar", "pie"]

# -------------------------
# UTIL: NORMALIZAR COLUMNAS SIN REGEX
# -------------------------
def normalize_colname(name: str) -> str:
    s = (name or "").lower().strip()
    s = (
        s.replace("á", "a").replace("é", "e")
        .replace("í", "i").replace("ó", "o").replace("ú", "u")
    )
    out = []
    prev_us = False
    for ch in s:
        ok = ch.isalnum() or ch == "_"
        if ok:
            out.append(ch)
            prev_us = (ch == "_")
        else:
            if not prev_us:
                out.append("_")
            prev_us = True
    res = "".join(out).strip("_")
    return res

# -------------------------
# SCHEMA ROUTER
# -------------------------
def _normalize_df_headers(df_in: pd.DataFrame) -> pd.DataFrame:
    df2 = df_in.copy()
    df2.columns = [normalize_colname(c) for c in df2.columns]
    return df2

def cargar_diccionario_columnas(dict_csv_path: str) -> pd.DataFrame:
    """
    Lee diccionario_kinesis.csv y devuelve un DF normalizado.
    Debe contener al menos un campo de nombre de columna; descripción y sinónimos son opcionales.
    """
    df_dict = pd.read_csv(dict_csv_path)
    df_dict = _normalize_df_headers(df_dict)

    # Detectar columnas (flexible)
    posibles_nombre = ["col_name", "column_name", "nombre_columna", "variable", "columna"]
    posibles_desc = ["descripcion", "description", "desc"]
    posibles_syn = ["sinonimos", "synonyms", "alias", "aliases"]

    name_col = next((c for c in posibles_nombre if c in df_dict.columns), None)
    if not name_col:
        raise ValueError("diccionario_kinesis.csv no tiene una columna reconocible para el nombre de columna.")

    desc_col = next((c for c in posibles_desc if c in df_dict.columns), None)
    syn_col = next((c for c in posibles_syn if c in df_dict.columns), None)

    df_dict = df_dict.rename(columns={name_col: "col_name"})
    if desc_col and desc_col != "descripcion":
        df_dict = df_dict.rename(columns={desc_col: "descripcion"})
    if syn_col and syn_col != "sinonimos":
        df_dict = df_dict.rename(columns={syn_col: "sinonimos"})

    if "descripcion" not in df_dict.columns:
        df_dict["descripcion"] = ""
    if "sinonimos" not in df_dict.columns:
        df_dict["sinonimos"] = ""

    df_dict["col_name"] = df_dict["col_name"].astype(str).apply(normalize_colname)
    df_dict["descripcion"] = df_dict["descripcion"].fillna("").astype(str)
    df_dict["sinonimos"] = df_dict["sinonimos"].fillna("").astype(str)

    return df_dict[["col_name", "descripcion", "sinonimos"]].drop_duplicates()

def construir_docs_columnas(schema_df: pd.DataFrame, dict_df: pd.DataFrame) -> list:
    """
    Construye Document por columna: col_name + dtype + descripción + sinónimos.
    """
    dict_map = {r["col_name"]: r for _, r in dict_df.iterrows()}
    docs = []
    for _, row in schema_df.iterrows():
        col = row["column_name"]
        dtype = row["column_type"]
        info = dict_map.get(col, {"descripcion": "", "sinonimos": ""})
        descripcion = info.get("descripcion", "") or ""
        sinonimos = info.get("sinonimos", "") or ""
        content = (
            f"col_name: {col}\n"
            f"dtype: {dtype}\n"
            f"descripcion: {descripcion}\n"
            f"sinonimos: {sinonimos}\n"
        )
        docs.append(Document(page_content=content, metadata={"col_name": col, "dtype": str(dtype)}))
    return docs

def init_schema_router(schema_df: pd.DataFrame) -> object | None:
    """
    Inicializa (o carga) el vector store del diccionario de columnas.
    Retorna el objeto v_db o None si no hay dependencias.
    """
    if not _SCHEMA_ROUTER_AVAILABLE:
        print("⚠️ Schema router: dependencias LangChain/Chroma no disponibles. Se usará fallback simple.")
        return None

    dict_df = cargar_diccionario_columnas(DICT_CSV_FILE)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=SCHEMA_EMBED_MODEL)

    # Persistencia
    Path(SCHEMA_VDB_PATH).mkdir(parents=True, exist_ok=True)

    # Crear/cargar
    v_db = Chroma(
        collection_name="kinesis_schema_router",
        persist_directory=SCHEMA_VDB_PATH,
        embedding_function=embeddings,
    )

    # Si está vacío, ingesta 1 vez
    try:
        existing = v_db._collection.count()  # type: ignore
    except Exception:
        existing = 0

    if existing == 0:
        docs = construir_docs_columnas(schema_df, dict_df)
        v_db.add_documents(docs)
        print(f"✅ Schema router: diccionario vectorizado creado ({len(docs)} columnas).")
    else:
        print(f"✅ Schema router: cargado ({existing} embeddings).")

    return v_db

def retrieve_candidate_columns(user_input: str, v_db: object | None, k: int) -> list[str]:
    """
    Devuelve lista de columnas candidatas (10-20 típicamente).
    Si no hay v_db, fallback a coincidencias simples.
    """
    if v_db is None:
        q = normalize_colname(user_input)
        toks = [t for t in q.split("_") if t]
        scored = []
        for c in SCHEMA_COLUMNS:
            score = sum(1 for t in toks if t in c)
            if score > 0:
                scored.append((score, c))
        scored.sort(reverse=True)
        cols = [c for _, c in scored][:k]
        return cols if cols else SCHEMA_COLUMNS[:min(k, len(SCHEMA_COLUMNS))]

    # 🔍 Retrieval con score
    docs_with_scores = v_db.similarity_search_with_score(user_input, k=k)

    cols = []
    print("\n📡 RESULTADO RAW DEL VECTOR RETRIEVAL:")
    for d, score in docs_with_scores:
        col = (d.metadata or {}).get("col_name")
        print(f"   → {col} | score: {score}")
        if col and col in SCHEMA_COLUMNS and col not in cols:
            cols.append(col)

    return cols[:k]

# -------------------------
# CARGA DE DATOS (DuckDB)
# -------------------------
print("Cargando base de datos SQL...")
sql_db = duckdb.connect(database=":memory:")
df = pd.read_csv(CSV_FILE)
df.columns = [normalize_colname(c) for c in df.columns]
sql_db.execute("CREATE TABLE kinesis AS SELECT * FROM df")
schema_df = sql_db.execute("DESCRIBE kinesis").df()
SCHEMA_COLUMNS = schema_df["column_name"].tolist()

print("\nCOLUMNAS QUE CONTIENEN 'pais':")
for c in SCHEMA_COLUMNS:
    if "pais" in c:
        print(">>", c)

NUMERIC_COLUMNS = [
    row["column_name"]
    for _, row in schema_df.iterrows()
    if "int" in row["column_type"].lower()
    or "double" in row["column_type"].lower()
    or "float" in row["column_type"].lower()
    or "decimal" in row["column_type"].lower()
]

# -------------------------
# COLUMNAS ESTRUCTURALES SIEMPRE DISPONIBLES
# -------------------------
COLUMNAS_ESTRUCTURALES = [
    "pais_empresa_principal",
    "sector_industria_empresa_principal"
]

schema_router_db = init_schema_router(schema_df)
print("✅ Base de datos lista.")

# -------------------------
# SYSTEM PROMPT CONTROLADO
# -------------------------
def build_system_prompt(candidate_cols: list[str], candidate_numeric_cols: list[str]) -> str:
    return f"""
Eres el Analista Principal del proyecto Kinesis. Tu tarea es traducir preguntas sobre datos en una estructura JSON estricta.

REGLAS OBLIGATORIAS:
1. Responde EXCLUSIVAMENTE en JSON válido (objeto JSON). Sin texto adicional.
2. NUNCA generes SQL ni fragmentos de código.
3. Solo puedes usar columnas EXACTAS de esta lista (sub-schema dinámico): {candidate_cols}
4. Las métricas para agregar deben ser columnas numéricas (solo estas del sub-schema): {candidate_numeric_cols}
5. Agregaciones permitidas: {ALLOWED_AGGREGATIONS}
6. tipo_consulta permitidos: {ALLOWED_TIPOS_CONSULTA}
7. tipo_grafico permitidos: {ALLOWED_TIPOS_GRAFICO}
8. OPCIONAL: Si la pregunta requiere filtrar por algo, agrega la llave "filtro" con un diccionario de condiciones.
9. Si no puedes contestar la pregunta con estos datos, responde exactamente: {{"error":"No hay datos suficientes para responder a esta pregunta."}}

FORMATOS PERMITIDOS:

--- PLANTILLA 1: COMPARATIVO DE MÉTRICAS ---
(Úsala cuando se comparen exactamente 2 métricas numéricas)
{{
  "tipo_consulta":"comparativo_metricas",
  "metricas":["columna_num_1","columna_num_2"],
  "agregacion":"sum",
  "tipo_grafico":"bar",
  "filtro": {{"nombre_columna_cualquiera": "Valor a filtrar"}}
}}

--- PLANTILLA 2: AGREGACIÓN SIMPLE ---
(Úsala para un solo KPI, monto total, promedio, o para CONTAR filas que cumplen una condición)
{{
  "tipo_consulta":"agregacion_simple",
  "metrica":"columna_a_operar",
  "agregacion":"sum",
  "filtro": {{"nombre_columna_condicion": "Valor Buscado"}}
}}

--- PLANTILLA 3: DISTRIBUCIÓN / COMPARACIÓN DE CLASES ---
Úsala SOLO cuando el usuario pida comparar categorías entre sí 
(Ej: "comparar empresas por país", "distribución por sector").

SI la pregunta contiene palabras como:
- promedio
- suma
- total
- cuánto
- valor
- monto

DEBES usar "agregacion_simple" y NO "distribucion".

{{
  "tipo_consulta":"distribucion",
  "columna":"nombre_columna_categoria",
  "tipo_grafico":"bar",
  "filtro": {{"nombre_columna_condicion": "Valor Buscado"}}
}}
""".strip()

# -------------------------
# VALIDACIÓN DURA
# -------------------------
def validar_json(data: dict, allowed_cols: list[str], allowed_numeric_cols: list[str]):
    if not isinstance(data, dict):
        return False, "El modelo no devolvió un objeto JSON válido."
    if "error" in data:
        return False, data.get("error")
    if "tipo_consulta" not in data:
        return False, "Falta tipo_consulta en el JSON."

    tipo = data["tipo_consulta"]
    if tipo not in ALLOWED_TIPOS_CONSULTA:
        return False, f"Tipo de consulta '{tipo}' no permitido."

    # Validar filtro si existe
    if "filtro" in data:
        if not isinstance(data["filtro"], dict):
            return False, "El campo 'filtro' debe ser un diccionario."
        for col_filtro in data["filtro"].keys():
            if col_filtro not in allowed_cols:
                return False, f"Columna de filtro no válida o no existe: {col_filtro}"

    if tipo in ["comparativo_metricas", "distribucion"]:
        if "tipo_grafico" not in data:
            return False, "Falta tipo_grafico."
        if data["tipo_grafico"] not in ALLOWED_TIPOS_GRAFICO:
            return False, "tipo_grafico no permitido."

    if tipo == "comparativo_metricas":
        if "metricas" not in data or "agregacion" not in data:
            return False, "Formato incompleto para comparativo_metricas."
        metricas = data["metricas"]
        if not isinstance(metricas, list) or len(metricas) != 2:
            return False, "comparativo_metricas requiere exactamente 2 métricas."
        if data["agregacion"] not in ALLOWED_AGGREGATIONS:
            return False, "Agregación no permitida."
        for col in metricas:
            if col not in allowed_cols:
                return False, f"Columna no válida: {col}"
            if col not in allowed_numeric_cols:
                return False, f"Columna no numérica: {col}"
        return True, None

    if tipo == "agregacion_simple":
        if "metrica" not in data or "agregacion" not in data:
            return False, "Formato incompleto para agregacion_simple."
        col = data["metrica"]
        if col not in allowed_cols:
            return False, f"Columna no válida: {col}"
        # MAGIA AQUÍ: Si es count, dejamos pasar columnas de texto. Si es sum/avg, exigimos numéricas.
        if data["agregacion"] != "count" and col not in allowed_numeric_cols:
            return False, f"Columna no numérica para esta operación: {col}"
        if data["agregacion"] not in ALLOWED_AGGREGATIONS:
            return False, "Agregación no permitida."
        return True, None

    if tipo == "distribucion":
        # No puede incluir metrica ni agregacion
        if "metrica" in data or "agregacion" in data:
            return False, "Distribucion no puede incluir metrica ni agregacion."

        if "columna" not in data:
            return False, "Falta columna."

        col = data["columna"]
        if col not in allowed_cols:
            return False, f"Columna no válida: {col}"

        return True, None

    return False, "Consulta no válida."

# -------------------------
# SQL CONTROLADO (backend)
# -------------------------
def construir_filtros_sql(filtros: dict) -> str:
    """Convierte un diccionario de filtros en un string WHERE para SQL."""
    if not filtros:
        return ""
    condiciones = []
    for col, val in filtros.items():
        # Limpiamos comillas simples en el valor para evitar errores SQL
        val_limpio = str(val).replace("'", "''")
        condiciones.append(f'"{col}" = \'{val_limpio}\'')
    return " WHERE " + " AND ".join(condiciones)

def construir_sql(data: dict) -> str | None:
    tipo = data["tipo_consulta"]
    # Extraemos el string del WHERE (ej: " WHERE pais = 'México'") o queda vacío ""
    filtro_sql = construir_filtros_sql(data.get("filtro", {}))

    if tipo == "comparativo_metricas":
        col1, col2 = data["metricas"][0], data["metricas"][1]
        agg = data["agregacion"].upper()
        return f"""
            SELECT '{col1}' as categoria, {agg}("{col1}") as valor FROM kinesis {filtro_sql}
            UNION ALL
            SELECT '{col2}' as categoria, {agg}("{col2}") as valor FROM kinesis {filtro_sql};
        """

    if tipo == "agregacion_simple":
        col = data["metrica"]
        agg = data["agregacion"].upper()
        return f"""
            SELECT {agg}("{col}") as valor FROM kinesis {filtro_sql};
        """

    if tipo == "distribucion":
        col = data["columna"]
        return f"""
            SELECT "{col}" as categoria, COUNT(*) as valor FROM kinesis {filtro_sql} GROUP BY "{col}";
        """

    return None

# -------------------------
# NARRATIVA LLM
# -------------------------
def resumir_resultado(user_input: str, df_res: pd.DataFrame):
    try:
        resp = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0,
            messages=[
                {"role": "system", "content": "Resume el resultado en una frase natural, breve y clara. Sé directo."},
                {"role": "user", "content": f"Datos extraídos:\n{df_res.to_string(index=False)}\n\nPregunta original:\n{user_input}"}
            ],
        )

        usage = resp.usage
        tokens_resumen = usage.total_tokens if usage else 0

        print("\n🧮 TOKENS (RESUMEN):")
        print(f"  prompt_tokens: {usage.prompt_tokens}")
        print(f"  completion_tokens: {usage.completion_tokens}")
        print(f"  total_tokens: {tokens_resumen}")

        return resp.choices[0].message.content, tokens_resumen

    except Exception:
        print("⚠️ No se pudo obtener uso de tokens (RESUMEN).")
        return "Aquí tienes los resultados de tu consulta.", 0
    return resp.choices[0].message.content

# -------------------------
# RESPUESTAS JSON API
# -------------------------
def error_response(msg: str):
    return {"tipo": "texto", "respuesta": msg}

def kpi_response(valor, interpretacion: str):
    return {"tipo": "kpi", "valor": str(valor), "interpretacion": interpretacion}

def grafico_response(user_input: str, df_res: pd.DataFrame, tipo_grafico: str, interpretacion: str):
    labels = df_res["categoria"].astype(str).tolist()
    values = pd.to_numeric(df_res["valor"], errors="coerce").fillna(0).tolist()

    if tipo_grafico == "pie":
        plotly_data = {
            "data": [{"type": "pie", "labels": labels, "values": values}],
            "layout": {
                "template": "plotly_white",
                # 1. <b> para negrita, <br><br> para espacio extra
                "title": {"text": f"<b>{user_input}</b><br><br>", "font": {"family": "Plus Jakarta Sans, sans-serif", "size": 14, "color": "#0d0d55"}, "x": 0.02},
                # 2. Aumentamos 't' (top margin) de 45 a 70
                "margin": {"t": 70, "b": 40, "l": 40, "r": 20}
            }
        }
    else:
        plotly_data = {
            "data": [{"type": "bar", "x": labels, "y": values, "marker": {"color": "#d50262"}}],
            "layout": {
                "template": "plotly_white",
                # 1. <b> para negrita, <br><br> para espacio extra
                "title": {"text": f"<b>{user_input}</b><br><br>", "font": {"family": "Plus Jakarta Sans, sans-serif", "size": 14, "color": "#0d0d55"}, "x": 0.02},
                # 2. Aumentamos 't' (top margin) de 45 a 70
                "margin": {"t": 70, "b": 40, "l": 40, "r": 20},
                "xaxis": {"title": "Categoría", "type": "category"},
                "yaxis": {"title": "Valor"}
            }
        }

    return {"tipo": "grafico", "datos_del_grafico": plotly_data, "interpretacion": interpretacion}


import re
import unicodedata
from collections import Counter

def ejecutar_analisis_semantico(user_input: str):

    texto_lower = user_input.lower()
    texto_lower = unicodedata.normalize("NFKD", texto_lower).encode("ascii", "ignore").decode("utf-8")

    if "menos" in texto_lower:
        col = "cierre_menos_gusto"
    elif "mas" in texto_lower:
        col = "cierre_mas_gusto"
    elif "significado" in texto_lower:
        col = "cierre_significado_kinesis_empresa"
    else:
        return error_response("No se pudo identificar la variable a analizar.")

    df_text = sql_db.execute(
        f'SELECT "{col}" FROM kinesis WHERE "{col}" IS NOT NULL'
    ).df()

    textos = " ".join(df_text[col].astype(str).tolist()).lower()

    textos = re.sub(r"[^a-záéíóúñ\s]", " ", textos)
    palabras = [p for p in textos.split() if len(p) > 3]

    stopwords = {
        "para","porque","pero","esto","esta","este","como",
        "más","menos","muy","que","con","del","las","los",
        "una","unos","unas","sobre","entre","desde"
    }

    palabras = [p for p in palabras if p not in stopwords]

    conteo = Counter(palabras)
    top = conteo.most_common(25)

    df_wc = pd.DataFrame(top, columns=["categoria","valor"])

    muestra = textos[:3500]

    resumen = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        messages=[
            {
                "role":"system",
                "content":"Analiza comentarios abiertos y entrega máximo dos párrafos claros sobre patrones principales encontrados. No uses listas."
            },
            {"role":"user","content":muestra}
        ]
    ).choices[0].message.content

    import random

    x_vals = [random.uniform(0, 1) for _ in range(len(df_wc))]
    y_vals = [random.uniform(0, 1) for _ in range(len(df_wc))]

    plotly_data = {
        "data": [
            {
                "type": "scatter",
                "mode": "text",
                "x": x_vals,
                "y": y_vals,
                "text": df_wc["categoria"].tolist(),
                "textfont": {
                    "size": [12 + v * 4 for v in df_wc["valor"].tolist()],
                    "color": [
                        ["#d50262", "#8e7dc6", "#0082bd", "#b2df20", "#2a4c8d"][i % 5]
                        for i in range(len(df_wc))
                    ]
                }
            }
        ],
        "layout": {
            "template": "plotly_white",
            "paper_bgcolor": "#F3F4F6",
            "plot_bgcolor": "#F3F4F6",
            "title": {
                "text": "<b>Nube de palabras</b>",
                "x": 0.02,
                "font": {
                    "family": "Plus Jakarta Sans, sans-serif",
                    "size": 16,
                    "color": "#0d0d55"
                }
            },
            "xaxis": {"visible": False, "range": [0, 1]},
            "yaxis": {"visible": False, "range": [0, 1]},
            "margin": {"t": 80, "b": 40, "l": 40, "r": 40}
        }
    }

    return {
        "tipo": "grafico",
        "datos_del_grafico": plotly_data,
        "interpretacion": resumen
    }

# -------------------------
# API
# -------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    texto: str

@app.post("/chat")
def chat(req: ChatRequest):
    user_input = req.texto or ""

    # 🔹 ANALISIS SEMANTICO DIRECTO (ANTES DE TODO EL FLUJO SQL NORMAL)
    if user_input.lower().startswith("haz un análisis semántico"):
        return ejecutar_analisis_semantico(user_input)


    # 0) Schema routing (ANTES del LLM)
    candidate_cols = retrieve_candidate_columns(
        user_input,
        schema_router_db,
        SCHEMA_TOP_K
    )

    # Agregamos columnas estructurales siempre
    candidate_cols = list(set(candidate_cols + COLUMNAS_ESTRUCTURALES))

    # -------------------------
    # DEBUG SCHEMA ROUTING
    # -------------------------
    print("\n" + "🧠" * 20)
    print("DEBUG - SCHEMA ROUTER")
    print(f"Pregunta usuario: {user_input}")
    print(f"\nColumnas candidatas (TOP {len(candidate_cols)}):")
    for i, col in enumerate(candidate_cols, 1):
        print(f"{i}. {col}")

    if "desempeno_pais_empresa" in candidate_cols:
        print("\n✅ 'desempeno_pais_empresa' ESTÁ en las candidatas.")
    else:
        print("\n❌ 'desempeno_pais_empresa' NO está en las candidatas.")

    # Mostrar también candidatas numéricas
    candidate_numeric_cols = [
        c for c in candidate_cols if c in NUMERIC_COLUMNS
    ]
    print("\nColumnas numéricas candidatas:")
    for col in candidate_numeric_cols:
        print(f"- {col}")
    print("🧠" * 20 + "\n")

    candidate_numeric_cols = [
        c for c in candidate_cols if c in NUMERIC_COLUMNS
    ]

    # 1) IA traduce lenguaje natural a JSON (Text-to-JSON)
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # Usamos un modelo rápido para la traducción a JSON
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": build_system_prompt(candidate_cols, candidate_numeric_cols)},
                {"role": "user", "content": user_input}
            ],
        )
        data = json.loads(response.choices[0].message.content)

        # --- TOKENS USADOS (JSON INTENT) ---
        try:
            usage = response.usage
            print("\n🧮 TOKENS (JSON INTENT):")
            print(f"  prompt_tokens: {usage.prompt_tokens}")
            print(f"  completion_tokens: {usage.completion_tokens}")
            print(f"  total_tokens: {usage.total_tokens}")
        except Exception:
            print("⚠️ No se pudo obtener uso de tokens (JSON INTENT).")

        # ---> AGREGADO: Imprimir el JSON generado por la IA en la terminal <---
        print("\n" + "=" * 40)
        print("🤖 JSON GENERADO POR LA IA:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("=" * 40 + "\n")

    except Exception as e:
        return error_response(f"Error procesando la solicitud con la IA: {str(e)}")


    # --- CORRECCIÓN AUTOMÁTICA DE INTENCIÓN ---
    preg_lower = user_input.lower()

    if data.get("tipo_consulta") == "distribucion":
        if any(p in preg_lower for p in ["promedio", "suma", "total", "cuanto", "valor", "monto"]):
            data["tipo_consulta"] = "agregacion_simple"
            data["metrica"] = data.get("metrica") or candidate_numeric_cols[0]
            data["agregacion"] = data.get("agregacion", "avg")
            data.pop("columna", None)
            data.pop("tipo_grafico", None)

    # 2) Validación Dura en Backend
    valido, err = validar_json(data, candidate_cols, candidate_numeric_cols)
    if not valido:
        # ---> AGREGADO: Imprimir por qué falló la validación <---
        print(f"❌ Error de validación: {err}")
        return error_response(err)

    # 3) Backend construye SQL
    try:
        sql = construir_sql(data)
        # ---> AGREGADO: Imprimir el SQL resultante en la terminal <---
        if sql:
            print("📊 SQL CONSTRUIDO:")
            print(sql)
            print("=" * 40 + "\n")
    except Exception as e:
        return error_response(f"Error construyendo la consulta SQL: {str(e)}")

    if not sql:
        return error_response("No se pudo construir la consulta.")

    # 4) DuckDB ejecuta SQL
    try:
        df_res = sql_db.execute(sql).df()
    except Exception as e:
        return error_response(f"Error en la base de datos: {str(e)}")

    if df_res.empty:
        return error_response("La consulta se ejecutó correctamente pero no arrojó resultados.")

    # 5) Retorno de formato (KPI vs Gráfico) + Narrativa
    debug_info = {
        "json_ia": data,
        "sql": sql if sql else "No se generó SQL"
    }

    if df_res.shape[0] == 1 and df_res.shape[1] == 1:
        val = df_res.iat[0, 0]
        # ---> INICIO NUEVO FORMATO NUMÉRICO <---
        try:
            val_num = float(val)
            texto_json = str(data).lower()

            es_moneda = any(p in texto_json for p in ["dolar", "usd", "venta", "costo", "ingreso", "monto", "presupuesto"])
            es_porcentaje = (
                any(p in texto_json for p in ["porcentaje", "percent", "%", "score"])
                or "porcentaje" in user_input.lower()
                or "score" in user_input.lower()
            )

            if es_moneda:
                val_str = f"${val_num:,.0f}"
            elif es_porcentaje:
                # Si viene como 0.86 → 86%
                if 0 <= val_num <= 1:
                    val_str = f"{val_num*100:.1f}%"
                else:
                    val_str = f"{val_num:.1f}%"
            else:
                val_str = f"{int(val_num):,}" if val_num.is_integer() else f"{val_num:,.2f}"
        except (ValueError, TypeError):
            val_str = str(val)

        interpretacion, tokens_resumen = resumir_resultado(user_input, df_res)
        # Enviamos el val_str (formateado) en lugar del val crudo
        resp = kpi_response(val_str, interpretacion)
        resp["debug"] = debug_info
        return resp

    interpretacion, tokens_resumen = resumir_resultado(user_input, df_res)
    tipo_grafico = data.get("tipo_grafico", "bar")
    resp = grafico_response(user_input, df_res, tipo_grafico, interpretacion)
    resp["debug"] = debug_info  # <- INYECTAMOS LA INFO AQUÍ

    # --- SUMA TOTAL TOKENS ---
    try:
        total_json = response.usage.total_tokens
    except Exception:
        total_json = 0

    total_tokens_request = total_json + tokens_resumen

    print("\n🔢 TOTAL TOKENS REQUEST:", total_tokens_request)
    return resp
