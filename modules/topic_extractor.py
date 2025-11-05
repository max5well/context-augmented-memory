# modules/topic_extractor.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_topic(prompt: str) -> str:
    """
    Extract a one-word topic or theme from the given prompt.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": (
                    "From the following message, extract the main topic or subject in one word.\n\n"
                    f"Message:\n{prompt}"
                )}
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception:
        return "unknown"
