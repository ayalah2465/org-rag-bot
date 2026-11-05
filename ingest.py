"""בניית אינדקס הידע: טעינה, פיצול, אמבדינג ושמירה ב-ChromaDB."""

from pathlib import Path

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_MODEL,
)


def build_index(rebuild: bool = False) -> int:
    """בונה את האינדקס ב-ChromaDB. מחזיר את מספר הצ'אנקים שנוצרו."""
    text_files = sorted(DATA_DIR.glob("*.txt"))
    if not text_files:
        raise FileNotFoundError("לא נמצאו קבצי ידע בתיקייה: " + str(DATA_DIR))

    # 1. טעינת המסמכים
    documents = SimpleDirectoryReader(input_files=[str(p) for p in text_files]).load_data()
    print("נטענו " + str(len(documents)) + " מסמכים")
    for doc in documents:
        print("   - " + Path(doc.metadata.get("file_path", doc.doc_id)).name)

    # 2. פיצול לפי גבולות משפט ופסקה (ללא עלות API)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    print(
        "נוצרו " + str(len(nodes)) + " צ'אנקים (chunk_size=" + str(CHUNK_SIZE)
        + ", overlap=" + str(CHUNK_OVERLAP) + ")"
    )

    # 3. חיבור ל-ChromaDB
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("הקולקציה הישנה נמחקה (rebuild).")
        except Exception:
            pass
    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # 4. יצירת אמבדינגים ושמירה לקולקציה
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=OpenAIEmbedding(model=EMBED_MODEL),
        show_progress=True,
    )
    print("האינדקס נשמר ב-" + str(CHROMA_DIR) + " (collection=" + COLLECTION_NAME + ")")
    return len(nodes)


if __name__ == "__main__":
    import sys

    build_index(rebuild="--rebuild" in sys.argv)
