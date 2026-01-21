from modules.embedding import embed_text
from modules.retrieval import retrieve_similar
from modules.context_service import decide_context


def get_context(prompt: str, top_k: int = 5):
    """
    Core context retrieval logic.
    Safe to call from API, CLI, or other applications.
    """
    query_embedding = embed_text(prompt)

    retrieved = retrieve_similar(
        query_embedding=query_embedding,
        top_k=top_k
    )

    return decide_context(
        prompt=prompt,
        candidates=retrieved
    )
