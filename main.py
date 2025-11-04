"""
main.py
Main entry point for the Context-Augmented Memory (CAM) system.
"""

import os
from datetime import datetime
from nanoid import generate
from modules import (
    llm_client,
    auto_tagger,
    memory,
    retrieval,
    context_decider,
    usefulness_filter,
)

# Silence tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    print("🧠 Context-Augmented Memory System (CAM)")
    print("Type 'exit' to quit or 'clear memory' to reset stored context.\n")

    while True:
        user_prompt = input("Enter your prompt: ").strip()

        # --- Quit command ---
        if user_prompt.lower() == "exit":
            break

        # --- 🧹 Memory clear command ---
        if user_prompt.lower() in {"clear memory", "reset memory"}:
            try:
                ids = memory.collection.get()["ids"]
                if ids:
                    memory.collection.delete(ids=ids)
                    print("🧹 Memory cleared.\n")
                else:
                    print("ℹ️ No memories to clear.\n")
            except Exception as e:
                print(f"⚠️ Could not clear memory: {e}")
            continue

        # --- Context retrieval decision ---
        should_use_context = context_decider.should_retrieve(user_prompt)
        context = ""

        if should_use_context:
            print("🔎 Semantic continuity detected — retrieving context...\n")
            context = retrieval.retrieve_context(user_prompt)

        # --- Build augmented prompt ---
        if context:
            print("\n📚 Retrieved context found — augmenting your prompt...\n")
            full_prompt = f"Context:\n{context}\n\nUser: {user_prompt}"
        else:
            full_prompt = user_prompt

        # --- Send to LLM ---
        print("💬 Sending prompt to LLM...\n")
        llm_output = llm_client.ask(full_prompt)

        print(f"\n🤖 LLM Output:\n {llm_output}\n")

        # --- Prepare metadata ---
        episode_id = generate(size=12)
        timestamp = datetime.now().isoformat()
        selected_tag = auto_tagger.auto_tag(user_prompt)

        metadata = {
            "timestamp": timestamp,
            "user_prompt": user_prompt,
            "tag": selected_tag,
            "topic_continued": str(should_use_context),
        }

        # --- Filter trivial or context-dependent prompts ---
        if usefulness_filter.is_useful(user_prompt):
            memory.collection.add(
                ids=[episode_id],
                documents=[llm_output],
                metadatas=[metadata],
            )
            print(
                f"🧠 Episode {episode_id} stored (tag: {selected_tag}, continued: {should_use_context})"
            )
        else:
            print("🚫 Skipped storing trivial or context-dependent prompt.")

        print("------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
