# proxy_api/clients/mistral_client.py
"""
Handles chat completions using the Mistral API.
Compatible with the modern `mistralai` SDK (>=0.2.0).
"""

import os
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def ask(prompt: str, api_key: str = None, model: str = "mistral-large-latest") -> str:
    """
    Send a chat request to the Mistral API and return the response.
    """
    key = api_key or os.getenv("MISTRAL_API_KEY")
    if not key:
        return "⚠️ Missing Mistral API key."

    try:
        # Initialize the new unified client
        client = Mistral(api_key=key)

        # Use the new chat completion API
        response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message["content"].strip()

    except Exception as e:
        return f"⚠️ Mistral Error: {e}"