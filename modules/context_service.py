from modules.retrieval import retrieve_context


def get_context(prompt: str, top_k: int = 5) -> str:
    """
    Core context retrieval logic.
    Returns a formatted context string from memory.
    """
    return retrieve_context(
        query=prompt,
        n_results=top_k
    )
