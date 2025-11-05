# modules/intent_classifier.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def classify_intent(prompt: str) -> str:
    """
    Classify a user prompt into:
    - 'fact' → new personal/contextual info
    - 'query' → asks for info
    - 'meta' → refers to system memory/history
    """
    if not prompt or len(prompt.split()) < 2:
        return "meta"

    lowered = prompt.lower().strip()

    if any(lowered.startswith(x) for x in ["what", "who", "when", "where", "why", "how", "is", "are", "can", "do", "does", "did"]):
        return "query"
    if any(x in lowered for x in ["remember", "recall", "you said", "what did i", "when did i", "show me"]):
        return "meta"
    if lowered.endswith("?"):
        return "query"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": (
                    "Classify this message strictly as one of: 'fact', 'query', or 'meta'. "
                    "A 'fact' adds new personal or contextual info. "
                    "A 'query' asks for knowledge. "
                    "A 'meta' refers to previous messages or timestamps.\n\n"
                    f"Message: {prompt}"
                )}
            ],
            temperature=0,
        )
        result = response.choices[0].message.content.strip().lower()
        if "query" in result:
            return "query"
        elif "meta" in result:
            return "meta"
        else:
            return "fact"
    except Exception:
        return "fact"
