# proxy_api/utils/fallback_llm.py

from proxy_api.clients import openai_client

def recover_response_format(raw) -> dict:
    """
    Uses a lightweight LLM (e.g. GPT-4o mini) to extract useful info
    from an unknown provider's response.
    """

    # Prepare prompt for the fallback LLM
    recovery_prompt = (
        "The following is a malformed or unknown LLM API response:\n\n"
        f"{raw}\n\n"
        "Please extract the user-facing message and return only the answer content."
    )

    response = openai_client.ask(
        prompt=recovery_prompt,
        model="gpt-4o-mini"  # use cheap model
    )

    return {
        "response": response.strip() if response else "(Fallback failed)",
    }
