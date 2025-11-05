# modules/llm_client.py
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    """
    Send a prompt to the LLM and return the response text.
    """
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
        )

        # Extract response text safely
        output = response.output[0].content[0].text
        return output

    except Exception as e:
        print(f"❌ LLM request failed: {e}")
        return "(Error: LLM request failed)"
