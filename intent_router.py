# intent_router.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, List, Tuple

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
except Exception as e:
    raise RuntimeError(
        "Faltan dependencias para router (langchain_huggingface, langchain_chroma, langchain_core)."
    ) from e


Route = Literal["rag", "sql"]


@dataclass
class RouteDecision:
    route: Route
    rag_score: float
    sql_score: float
    winner: str


class IntentRouter:
    """
    Router por intención usando dos diccionarios vectorizados (RAG vs SQL).

    NOTA SOBRE SCORES:
    - similarity_search_with_score en Chroma suele devolver "distancia" (menor = mejor).
    - Por eso gana el score más pequeño.
    """

    def __init__(
        self,
        persist_dir: str = "data/vectors_intent_router",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        self.db = Chroma(
            collection_name="intent_router",
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
        )

    def ensure_seed(self) -> None:
        # si ya hay docs, no reinsertar
        try:
            existing = self.db._collection.count()  # type: ignore
        except Exception:
            existing = 0
        if existing > 0:
            return

        rag_prompts = [
            "¿Qué es Kinesis?",
            "¿En qué consiste la metodología?",
            "¿Cuáles son los criterios de selección para ser aceptados por el programa?",
            "¿Cuál es el valor diferencial del programa?",
            "¿Qué países participan en el proyecto?",
            "¿A qué nos referimos con 'momentum de impacto'?",
        ]

        sql_prompts = [
            "¿Cuál es el promedio del ticket promedio de las empresas de México?",
            "¿A cuánto ascienden el total de las ventas anuales promedio del sector Agroindustria?",
            "Haz un análisis de la cantidad de empresas según su sector o industria y muéstrame la comparación en un gráfico",
            "Representa en un gráfico la distribución del genero de las y los participantes",
            "Genera un gráfico comparativo entre el total de empleos generados versus el total de empleos mantenidos",
            "¿Cuál es el total de ventas?",
            "¿Cuál es la suma de ingresos?",
            "¿Cuál es el conteo por categoría?",
        ]

        docs: List[Document] = []
        for q in rag_prompts:
            docs.append(Document(page_content=q, metadata={"route": "rag"}))
        for q in sql_prompts:
            docs.append(Document(page_content=q, metadata={"route": "sql"}))

        self.db.add_documents(docs)
        # persistencia (Chroma suele persistir automáticamente, pero esto ayuda)
        try:
            self.db.persist()
        except Exception:
            pass

    def _best_score(self, query: str, route: Route, k: int = 4) -> float:
        docs_scores: List[Tuple[Document, float]] = self.db.similarity_search_with_score(query, k=k)
        # filtrar por route
        filtered = [s for (d, s) in docs_scores if (d.metadata or {}).get("route") == route]
        # si no hay, penaliza
        if not filtered:
            return 1e9
        return min(filtered)  # menor = mejor (distancia)

    def decide(self, query: str) -> RouteDecision:
        self.ensure_seed()
        rag_score = self._best_score(query, "rag", k=8)
        sql_score = self._best_score(query, "sql", k=8)

        if sql_score < rag_score:
            return RouteDecision(route="sql", rag_score=rag_score, sql_score=sql_score, winner="sql")
        return RouteDecision(route="rag", rag_score=rag_score, sql_score=sql_score, winner="rag")