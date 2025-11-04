from mistralai import Mistral
from dotenv import load_dotenv
import os

load_dotenv()


def ask(prompt: str, api_key: str = None, model: str = "mistral-large-latest") -> str:
    """
    Sends a chat completion request to Mistral API.
    Compatible with mistralai>=1.8.0.
    """
    key = api_key or os.getenv("MISTRAL_API_KEY")
    if not key:
        return "⚠️ Missing Mistral API key."

    try:
        client = Mistral(api_key=key)

        response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract the content properly
        content = response.choices[0].message["content"]
        return content.strip()

    except Exception as e:
        return f"⚠️ Mistral Error: {e}"
