"""
main.py
Main entry point for the Context-Augmented Memory (CAM) system.
"""

from modules import llm_client, auto_tagger, memory, retrieval
from datetime import datetime
from nanoid import generate
import os

# Silence parallel tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    print("🧠 Context-Augmented Memory System (CAM)")
    print("Type 'exit' to quit or 'clear memory' to reset stored context.\n")

    while True:
        user_prompt = input("Enter your prompt: ").strip()

        # --- Quit command ---
        if user_prompt.lower() == "exit":
            print("👋 Goodbye!")
            break

        # --- 🧹 Memory clear command ---
        if user_prompt.lower() in {"clear memory", "reset memory"}:
            all_ids = memory.collection.get().get("ids", [])
            if all_ids:
                memory.collection.delete(ids=all_ids)
                print(f"🧹 Memory cleared ({len(all_ids)} entries removed).\n")
            else:
                print("⚠️ No memory entries to clear.\n")
            continue

        # --- Retrieve relevant context ---
        context = retrieval.retrieve_context(user_prompt)

        if context:
            print("\n📚 Retrieved context found — augmenting your prompt...\n")
            print("🔎 Retrieved memory content:\n")
            print("=" * 80)
            print(context)
            print("=" * 80 + "\n")
            full_prompt = f"Context:\n{context}\n\nUser: {user_prompt}"
        else:
            full_prompt = user_prompt

        # --- Send to LLM ---
        print("💬 Sending prompt to LLM...\n")
        llm_output = llm_client.ask(full_prompt)

        # --- Tag and store memory ---
        selected_tag = auto_tagger.auto_tag(user_prompt)
        episode_id = generate(size=12)
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_prompt": user_prompt,
            "tag": selected_tag,
        }

        memory.collection.add(
            ids=[episode_id],
            documents=[user_prompt],  # ✅ store user input, not model output
            metadatas=[metadata],
        )

        print(f"✅ Added episode {episode_id}\n")
        print("🤖 LLM Output:\n", llm_output)
        print(f"\n🧠 Episode {episode_id} stored (tag: {selected_tag})")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
