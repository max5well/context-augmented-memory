from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(user_prompt: str, model="gpt-4o-mini"):
    """Generate an LLM response and return text + metadata."""
    resp = client.responses.create(model=model, input=user_prompt, temperature=0)
    content = resp.output[0].content[0].text

    metadata = {
        "timestamp": resp.created_at,
        "model": resp.model,
        "temperature": resp.temperature,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "total_tokens": resp.usage.total_tokens,
    }

    return content, metadata
