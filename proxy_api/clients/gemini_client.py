# proxy_api/clients/gemini_client.py
"""
Client for Google Gemini models.
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def ask(prompt: str, api_key: str = None, model: str = "gemini-1.5-flash") -> str:
    """
    Sends a text prompt to Gemini and returns its response.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return "⚠️ Missing Gemini API key."

    try:
        genai.configure(api_key=key)
        model_client = genai.GenerativeModel(model)
        response = model_client.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"
