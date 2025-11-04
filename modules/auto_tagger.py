from openai import OpenAI
import json, os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ALLOWED_TAGS = ["refund", "complaint", "review"]

def auto_tag(text: str, tags=ALLOWED_TAGS) -> str:
    """Auto-tag text using allowed labels."""
    prompt = (
        f"You are a precise tagger. Allowed tags: {tags}. "
        "Choose one that clearly applies, or NONE. "
        f"Output JSON only: {{'tag': '<one of {tags} or NONE>'}}.\nText: {text}"
    )

    resp = client.responses.create(model="gpt-4o-mini", input=prompt, temperature=0)
    raw = resp.output[0].content[0].text
    try:
        return json.loads(raw)["tag"]
    except Exception:
        return "NONE"
