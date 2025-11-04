"""
usefulness_filter.py
Determines whether a user prompt should be stored in long-term memory.
Fully configurable — thresholds and blacklist loaded dynamically.
"""

import re
from openai import OpenAI
from modules import config_manager

client = OpenAI()

# Load configuration dynamically
config = config_manager.load_config()
MIN_WORD_COUNT = config["usefulness_filter"]["min_word_count"]
MIN_CHAR_COUNT = config["usefulness_filter"]["min_char_count"]
BLACKLIST_PHRASES = config["usefulness_filter"]["blacklist_phrases"]


def is_useful(prompt: str) -> bool:
    """
    Returns True if the user input contains meaningful factual information.
    Uses both heuristic and optional LLM-based checks.
    """

    if not prompt or len(prompt.strip()) < MIN_CHAR_COUNT:
        return False

    prompt_lower = prompt.lower().strip()

    # 🚫 Skip trivial or blacklisted phrases
    if any(phrase in prompt_lower for phrase in BLACKLIST_PHRASES):
        return False

    # 🚫 Skip short or filler responses
    if len(prompt.split()) < MIN_WORD_COUNT:
        return False

    if re.fullmatch(r"(yes|no|ok|okay|sure|maybe|hmm|thanks|thank you|cool)", prompt_lower):
        return False

    # ✅ Allow descriptive / factual statements
    if re.search(
        r"\b(is|am|are|was|were|have|has|called|named|lives|works|likes|owns|contains)\b",
        prompt_lower,
    ):
        return True

    # 🧠 Optional fallback: use LLM for semantic judgment
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=(
                "Determine if the following text contains storable factual information. "
                "Reply 'true' if it conveys personal facts, relationships, or properties. "
                "Reply 'false' if it's trivial, meta, or uninformative.\n\n"
                f"Text: {prompt}"
            ),
        )
        decision = response.output_text.strip().lower()
        return decision.startswith("true")
    except Exception:
        return False
