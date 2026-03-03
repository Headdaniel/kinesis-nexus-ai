# lookup_app.py
import re

# Solo estas 4 columnas existen para lookup
COL_ID = "id"
COL_NOMBRE = "nombre_empresa_principal"
COL_EMAIL = "email_empresa_principal"
COL_REP = "digital_nombre_empresario"

def es_lookup_simple(user_input: str) -> bool:
    t = (user_input or "").lower()

    # Solo activar lookup si parece pregunta directa de atributo específico
    triggers = ["email", "correo", "id", "represent", "particip"]

    return any(k in t for k in triggers)

def _extraer_id(texto: str):
    m = re.search(r"\b(\d{3,})\b", texto)  # IDs suelen ser >=3 dígitos
    return m.group(1) if m else None

def _extraer_nombre_empresa(texto: str):
    """
    Intenta extraer el nombre de empresa de frases como:
    - "email de la empresa X"
    - "email de X"
    - "nombre de la empresa X"
    - "empresa X"
    """
    t = (texto or "").strip()

    # 1) Si aparece "empresa", toma lo que venga después
    m = re.search(r"\bempresa\b\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" ?¿\"'")

    # 2) Si aparece "de", toma lo que venga después
    m = re.search(r"\bde\b\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" ?¿\"'")

    # 3) fallback: nada claro
    return None

def _objetivo_lookup(texto_lower: str):
    """
    ¿Qué quiere el usuario que le devolvamos?
    """
    if "correo" in texto_lower or "email" in texto_lower:
        return COL_EMAIL
    if "represent" in texto_lower or "particip" in texto_lower:
        return COL_REP
    if "id" in texto_lower:
        return COL_ID
    if "nombre" in texto_lower:
        return COL_NOMBRE
    return None

def _normalizar_para_like(s: str) -> str:
    """
    Normalización simple del lado Python para buscar:
    - lower
    - quitar signos
    - colapsar espacios
    """
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9áéíóúñ]+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ejecutar_lookup_simple(user_input: str, db_connection):
    texto = (user_input or "").strip()
    texto_lower = texto.lower()

    objetivo = _objetivo_lookup(texto_lower)
    if not objetivo:
        return {"tipo": "texto", "respuesta": "No pude interpretar si quieres ID, nombre, email o representante."}

    id_val = _extraer_id(texto)
    nombre_empresa = None if id_val else _extraer_nombre_empresa(texto)

    print("\n🔎 LOOKUP DEBUG")
    print("   pregunta:", texto)
    print("   objetivo:", objetivo)
    print("   id_detectado:", id_val)
    print("   nombre_detectado:", nombre_empresa)

    # --- Caso A: buscar por ID ---
    if id_val:
        sql = f'''
            SELECT "{objetivo}" AS valor
            FROM kinesis
            WHERE "{COL_ID}" = ?
            LIMIT 1;
        '''
        print("   SQL:", sql.strip(), "| param:", id_val)
        res = db_connection.execute(sql, [id_val]).fetchone()
        print("   RESULT:", res)

        if not res or res[0] is None or str(res[0]).strip() == "":
            return {"tipo": "texto", "respuesta": "No se encontró información con ese criterio."}

        return {"tipo": "texto", "respuesta": str(res[0]).strip()}

    # --- Caso B: buscar por nombre (empresa o representante) ---
    if not nombre_empresa:
        return {"tipo": "texto", "respuesta": "No pude extraer el nombre para buscar."}

    nombre_norm = _normalizar_para_like(nombre_empresa)

    # 🔹 Decidir en qué columna buscar
    columna_busqueda = COL_NOMBRE

    # Si la pregunta NO menciona "empresa", asumimos que puede ser persona
    if "empresa" not in texto_lower:
        columna_busqueda = COL_REP

    # Si explícitamente menciona representante/participó
    if "represent" in texto_lower or "particip" in texto_lower:
        columna_busqueda = COL_REP

    sql = f'''
        SELECT "{objetivo}" AS valor
        FROM kinesis
        WHERE regexp_replace(lower("{columna_busqueda}"), '[^a-z0-9áéíóúñ]+', ' ', 'g') LIKE ?
        LIMIT 1;
    '''

    param = f"%{nombre_norm}%"
    print("   SQL:", sql.strip(), "| param:", param)

    res = db_connection.execute(sql, [param]).fetchone()
    print("   RESULT:", res)

    if not res or res[0] is None or str(res[0]).strip() == "":
        return {"tipo": "texto", "respuesta": "No se encontró información con ese criterio."}

    return {"tipo": "texto", "respuesta": str(res[0]).strip()}