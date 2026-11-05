"""שכבת השאילתה: רטרייבר וקטורי או היברידי, והרחבת שאלה אופציונלית."""

import time
from functools import lru_cache
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    LLM_MODEL,
    SYSTEM_PROMPT,
    TOP_K,
)


@lru_cache(maxsize=1)
def get_index() -> VectorStoreIndex:
    """טוען את האינדקס הקיים מ-ChromaDB (פעם אחת לכל תהליך)."""
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    Settings.llm = OpenAI(model=LLM_MODEL, temperature=0.0)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )


def expand_query(question: str) -> str:
    """הרחבת שאלה קצרה דרך LLM - קריאה אחת בלבד."""
    prompt = (
        "להלן שאלה קצרה של עובד תחנת טיפת חלב. נסח אותה מחדש כשאילתה מפורטת "
        "הכוללת את המונחים המנהליים הרלוונטיים (זימון תורים, מסמכים נדרשים, "
        "העברת תיק, מלאי, שעות פעילות). החזר את השאילתה בלבד, בלי הסברים.\n\n"
        "השאלה: " + question
    )
    try:
        expanded = Settings.llm.complete(prompt).text.strip()
    except Exception as exc:
        print("[אזהרה] הרחבת שאלה נכשלה, ממשיכים בשאלה המקורית: " + str(exc))
        return question
    return expanded or question

@lru_cache(maxsize=1)
def get_bm25_nodes():
    """טוען את הקטעים השמורים ב-Chroma עבור חיפוש BM25."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    records = collection.get(include=["documents", "metadatas"])

    nodes = []
    for text, metadata in zip(
        records["documents"], records["metadatas"]
    ):
        if not text or not text.strip():
            continue

        node = metadata_dict_to_node(metadata)
        node.set_content(text)
        nodes.append(node)

    if not nodes:
        raise ValueError("לא נמצאו קטעי טקסט ב-Chroma עבור BM25.")

    print(f"נטענו {len(nodes)} קטעים ל-BM25.")
    return nodes
def build_retriever(use_hybrid: bool = True, top_k: int = TOP_K):
    """חיפוש וקטורי או היברידי באותם קטעים השמורים ב-Chroma."""
    if top_k < 1:
        raise ValueError("top_k חייב להיות גדול מאפס.")

    index = get_index()
    vector_retriever = index.as_retriever(similarity_top_k=top_k)

    if not use_hybrid:
        return vector_retriever

    nodes = get_bm25_nodes()
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=min(top_k, len(nodes)),
    )

    return QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=top_k,
        num_queries=1,
        mode=FUSION_MODES.RELATIVE_SCORE,
        use_async=False,
        verbose=False,
    )

def retrieve(question, top_k=TOP_K, use_hybrid=True, use_expansion=False):
    """מחזיר (צ'אנקים, השאילתה שנשלחה בפועל)."""
    retriever = build_retriever(use_hybrid=use_hybrid, top_k=top_k)
    effective_query = expand_query(question) if use_expansion else question
    return retriever.retrieve(effective_query), effective_query


def format_context(nodes) -> str:
    parts = []
    for i, node in enumerate(nodes, 1):
        source = node.metadata.get("file_name") or node.metadata.get("file_path", "לא ידוע")
        parts.append("[קטע " + str(i) + " | מקור: " + str(source) + "]\n" + node.get_text())
    return "\n\n".join(parts)


def answer(question, top_k=TOP_K, use_hybrid=True, use_expansion=False) -> dict:
    """מחזיר תשובה מלאה: טקסט, מקורות, השאילתה האפקטיבית וזמן ריצה."""
    start = time.perf_counter()
    nodes, effective_query = retrieve(
        question, top_k=top_k, use_hybrid=use_hybrid, use_expansion=use_expansion
    )

    if not nodes:
        return {
            "answer": "לא נמצא מידע רלוונטי במסמכי הארגון.",
            "sources": [],
            "nodes": [],
            "effective_query": effective_query,
            "elapsed": time.perf_counter() - start,
        }

    prompt = (
        SYSTEM_PROMPT
        + "\n\nשאלת העובד:\n" + question
        + "\n\nקטעי ההקשר מתוך מסמכי הארגון:\n" + format_context(nodes)
        + "\n\nענה על סמך הקטעים בלבד."
    )
    response = Settings.llm.complete(prompt).text.strip()

    sources = []
    for node in nodes:
        name = node.metadata.get("file_name") or str(
            node.metadata.get("file_path", "")
        )
        if name and name not in sources:
            sources.append(name)

    if sources:
        display_sources = []
        for source in sources:
            name = source.replace("\\", "/").rsplit("/", 1)[-1]
            name = name.removesuffix(".txt").replace("_", " ")
            display_sources.append(name)

        response += "\n\n**מקורות הקטעים שנשלחו למודל:**\n"
        response += "\n".join(
            f"- \u200f{name}" for name in display_sources
        )

    return {
        "answer": response,
        "sources": sources,
        "nodes": nodes,
        "effective_query": effective_query,
        "elapsed": time.perf_counter() - start,
    }


if __name__ == "__main__":
    result = answer("מהן שעות קבלת הקהל בתחנת קטמון?")
    print(result["answer"])
    print("מקורות: " + str(result["sources"]))
