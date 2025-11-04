import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

load_dotenv()
def ask(prompt: str, api_key: str = None, model: str = "mistral-large-latest") -> str:
    key = api_key or os.getenv("MISTRAL_API_KEY")
    if not key:
        return "⚠️ Missing Mistral API key."
    client = MistralClient(api_key=key)
    try:
        response = client.chat(
            model=model,
            messages=[ChatMessage(role="user", content=prompt)],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Mistral Error: {e}"
