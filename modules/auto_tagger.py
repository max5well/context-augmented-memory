# modules/auto_tagger.py

import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ALLOWED_TAGS = ["refund", "complaint", "review", "fact", "question", "instruction", "NONE"]

def auto_tag(text: str, tags=ALLOWED_TAGS) -> str:
    """
    Auto-tag text using allowed labels (fallbacks to 'NONE').
    """
    prompt = (
        f"You are a precise tagger. Allowed tags: {tags}. "
        "Choose one that clearly applies, or NONE. "
        f"Output only the tag in plain text (no explanation).\n\nText:\n{text}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw_tag = response.choices[0].message.content.strip().upper()
        return raw_tag if raw_tag in tags else "NONE"
    except Exception:
        return "NONE"
