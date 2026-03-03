# main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag_app as rag
import sql_app as sql
from mining_app import ejecutar_mineria_determinista


from intent_router import IntentRouter

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# (Opcional) montar los sub-endpoints para debug manual:
# - POST /rag/chat
# - POST /sql/chat
app.mount("/rag", rag.app)
app.mount("/sql", sql.app)

router = IntentRouter(
    persist_dir="data/vectors_intent_router",
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
)

class ChatRequest(BaseModel):
    texto: str

@app.post("/chat")
def unified_chat(req: ChatRequest):
    user_input = (req.texto or "").strip()
    if not user_input:
        return {"tipo": "texto", "respuesta": "Escribe una pregunta."}

    # 🔴 0) MINERÍA DETERMINISTA (ANTES DE TODO)
    if user_input.lower() == "ejecuta un análisis minería de datos para encontrar el patrón oculto más relevante":
        return ejecutar_mineria_determinista(sql.df)

    # 🔹 1) SALUDOS
    if user_input.lower() in ["hola", "buenas", "hey", "buenos dias", "buenos días"]:
        return rag.chat(rag.ChatRequest(texto=user_input))

    # 🔹 2) ROUTER NORMAL
    decision = router.decide(user_input)

    print("\n🧭 ROUTER:")
    print(f"  route: {decision.route}")
    print(f"  rag_score: {decision.rag_score}")
    print(f"  sql_score: {decision.sql_score}")

    if decision.route == "sql":
        return sql.chat(sql.ChatRequest(texto=user_input))

    return rag.chat(rag.ChatRequest(texto=user_input))