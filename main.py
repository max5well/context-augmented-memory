from modules import llm_client, auto_tagger, memory
from datetime import datetime

def main():
    user_prompt = input("Enter your prompt: ")

    # 1. Get LLM response
    llm_output, meta = llm_client.generate_response(user_prompt)

    # 2. Auto-tag
    tag = auto_tagger.auto_tag(user_prompt)

    # 3. Build metadata
    metadata = {
        **meta,
        "user": "local_dev",
        "tag": tag,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # 4. Store episode in memory
    episode_id = memory.add_episode(user_prompt, llm_output, metadata)

    print(f"\n🧠 Episode {episode_id} stored.")
    print("Metadata:", metadata)
    print("LLM Output:", llm_output)

if __name__ == "__main__":
    main()