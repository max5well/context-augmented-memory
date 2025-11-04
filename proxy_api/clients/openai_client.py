import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
def ask(prompt: str, api_key: str = None, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return "⚠️ Missing OpenAI API key."
    client = OpenAI(api_key=key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI Error: {e}"
