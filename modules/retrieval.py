# modules/retrieval.py

import chromadb
from chromadb.config import Settings

# --- Configuration ---
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
MAX_DISTANCE = 0.25  # Lower = stricter. Adjust based on your embedding model.

# --- Initialize Chroma client ---
def get_chroma_client():
    """Return a Chroma v2 client."""
    return chromadb.Client(Settings(
        chroma_server_host=CHROMA_HOST,
        chroma_server_http_port=CHROMA_PORT,
        chroma_api_impl="chromadb.api.fastapi.FastAPI",
    ))

# --- Retrieval logic ---
def retrieve_context(query: str, n_results: int = 5):
    """
    Retrieve relevant past memory entries for a given query.
    Returns a formatted context block or an empty string.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(name="CAM_Collection")

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )

    if not results or not results.get("documents"):
        print("⚠️ No matching memory found.")
        return ""

    docs = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    relevant = []
    for doc, dist, meta in zip(docs, distances, metadatas):
        if dist <= MAX_DISTANCE:
            relevant.append((doc, dist, meta))

    if not relevant:
        print(f"⚠️ No relevant items under distance threshold ({MAX_DISTANCE}).")
        return ""

    # Build the formatted context block
    context_lines = []
    for doc, dist, meta in relevant:
        tag = meta.get("tag", "unknown")
        context_lines.append(
            f"[Memory — tag: {tag} | distance: {dist:.3f}]\n{doc}\n"
        )

    context_block = "\n---\n".join(context_lines)
    print(f"✅ Retrieved {len(relevant)} relevant memories (≤ {MAX_DISTANCE})")

    return context_block
