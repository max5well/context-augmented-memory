"""
main.py
Standalone CLI runner for Context-Augmented Memory (CAM)
Includes:
- short-term sliding context window for factual continuity
- robust dual-mode retrieval
- safe metadata serialization for Chroma
"""

import os
import datetime
from uuid import uuid4
from openai import OpenAI
from modules import (
    embedding,
    memory,
    retrieval,
    auto_tagger,
    intent_classifier,
    context_service,
    config_manager,
)

# --- Initialize ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
config = config_manager.load_config()

print("🧠 Context-Augmented Memory System (CAM)")
print("Type 'exit' to quit or 'clear memory' to reset stored context.\n")

# Maintain a small rolling memory of recent factual prompts
recent_facts = []
MAX_CONTEXT_WINDOW = 3


def sanitize_metadata(meta: dict) -> dict:
    """
    Ensures all metadata values are Chroma-safe (JSON-serializable primitives).
    Converts bools and unsupported types to strings.
    """
    safe_meta = {}
    for k, v in meta.items():
        if isinstance(v, (bool,)):
            safe_meta[k] = str(v).lower()
        elif isinstance(v, (str, int, float)) or v is None:
            safe_meta[k] = v
        else:
            safe_meta[k] = str(v)
    return safe_meta


def main():
    while True:
        user_prompt = input("Enter your prompt: ").strip()
        if not user_prompt:
            continue

        if user_prompt.lower() in ["exit", "quit"]:
            break
        if user_prompt.lower() in ["clear memory", "reset"]:
            os.system("rm -rf CAM_project/chroma_db")
            recent_facts.clear()
            print("🧹 Memory cleared.")
            continue

        # Step 1 — Determine intent
        intent = intent_classifier.classify_intent(user_prompt)
        print(f"🎯 Detected intent: {intent}")

        # Step 2 — Retrieval decision
        should_use_context = False
        context = ""

        try:
            if intent == "query":
                print("🔍 Query detected — searching memory globally...")
                context = retrieval.retrieve_context(user_prompt, n_results=5, mode="global")
                should_use_context = bool(context)
            elif intent == "fact":
                should_use_context = context_decider.should_retrieve(user_prompt)
        except Exception as e:
            print(f"⚠️ Retrieval decision failed: {e}")

        if should_use_context and context:
            print("📚 Retrieved context found — augmenting your prompt..\n")
            full_prompt = f"{context}\n\nUser: {user_prompt}"
        elif should_use_context:
            print("🔎 Semantic continuity detected — retrieving context...\n")
            context = retrieval.retrieve_context(user_prompt)
            full_prompt = f"{context}\n\nUser: {user_prompt}"
        else:
            if intent == "query":
                print("⚙️ No context found — running standalone query.")
            else:
                print("🚫 No relevant context found — proceeding without memory.")
            full_prompt = user_prompt

        # Step 3 — LLM generation
        print("💬 Sending prompt to LLM...\n")
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=full_prompt,
            )
            llm_output = response.output_text.strip()
        except Exception as e:
            print(f"❌ LLM request failed: {e}")
            llm_output = "(Error: LLM request failed)"

        print(f"\n🤖 LLM Output:\n {llm_output}\n")

        # Step 4 — Skip trivial/meta messages
        if intent in ["meta", "query"] and not should_use_context:
            print("🚫 Skipped storing trivial, query, or meta prompt.")
            print("------------------------------------------------------------\n")
            continue

        tag = auto_tagger.auto_tag(user_prompt)

        # Step 5 — Build metadata
        episode_id = str(uuid4())[:12]
        meta = {
            "episode_id": episode_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "session_id": f"sess_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "schema_version": "2025-11",
            "user_prompt": user_prompt,
            "tag": tag,
            "intent": intent,
            "topic_continued": should_use_context,
        }

        # ✅ Make metadata Chroma-safe
        meta = sanitize_metadata(meta)

        # Step 6 — Short-term context window for embeddings
        if intent == "fact":
            recent_facts.append(user_prompt)
            if len(recent_facts) > MAX_CONTEXT_WINDOW:
                recent_facts.pop(0)

        embedding_input = " ".join(recent_facts[-MAX_CONTEXT_WINDOW:]) + " " + llm_output

        try:
            embedding_vector = embedding.get_embedding(embedding_input)
            memory.store(llm_output, meta, embedding_vector)
            print(f"🧠 Stored memory: {episode_id} ({len(embedding_vector)} dims) ✅")
        except Exception as e:
            print(f"❌ Failed to store memory: {e}")

        print(f"🧠 Episode {episode_id} stored (tag: {tag}, intent: {intent})")
        print("------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
