"""
modules/usefulness_filter.py
Evaluates whether a prompt is semantically meaningful enough to store in memory.
"""

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def is_useful(prompt: str) -> bool:
    """
    Uses a lightweight LLM check to decide whether the user prompt
    carries standalone meaning or depends too heavily on missing context.
    """
    if len(prompt.strip()) < 3:
        return False  # trivial one-word or empty prompts

    system_prompt = (
        "You are a strict memory gatekeeper. Decide if the following text is meaningful "
        "enough to be stored as a standalone memory. Respond only with 'true' or 'false'. "
        "Return 'false' if the text depends on prior context (e.g., 'What about him?', 'Yes.', 'Why?')."
    )

    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"{system_prompt}\n\nText: {prompt}",
        temperature=0
    )

    decision = response.output[0].content[0].text.strip().lower()
    return "true" in decision
