"""
intent_classifier.py
Determines whether a user prompt is a 'fact', 'query', or 'meta' statement.
"""

from openai import OpenAI

client = OpenAI()

def classify_intent(prompt: str) -> str:
    """
    Classify a user prompt into:
    - 'fact'  → contains new information to store
    - 'query' → asks for information or clarification
    - 'meta'  → refers to the system’s memory, timestamps, or past interactions
    """
    if not prompt or len(prompt.split()) < 2:
        return "meta"

    lowered = prompt.lower().strip()

    # --- Quick heuristics (fast path) ---
    if any(lowered.startswith(x) for x in [
        "what", "who", "when", "where", "why", "how", "is", "are", "can", "do", "does", "did"
    ]):
        return "query"

    if any(x in lowered for x in [
        "remember", "tell me what i said", "what did i", "when did i", "show me", "recall", "you said"
    ]):
        return "meta"

    if lowered.endswith("?"):
        return "query"

    # --- Fallback: small LLM classification for edge cases ---
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=(
                "Classify this message strictly as one of: 'fact', 'query', or 'meta'. "
                "A 'fact' adds new personal or contextual info. "
                "A 'query' asks for knowledge. "
                "A 'meta' refers to previous messages or timestamps.\n\n"
                f"Message: {prompt}"
            ),
        )
        result = response.output_text.strip().lower()
        if "query" in result:
            return "query"
        elif "meta" in result:
            return "meta"
        else:
            return "fact"
    except Exception:
        # if OpenAI fails, default to fact
        return "fact"
