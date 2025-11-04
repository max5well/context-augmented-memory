import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
def ask(prompt: str, api_key: str = None, model: str = "claude-3-5-sonnet") -> str:
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return "⚠️ Missing Anthropic API key."
    client = Anthropic(api_key=key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"⚠️ Anthropic Error: {e}"
