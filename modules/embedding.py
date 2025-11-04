import main
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from nanoid import generate


#chroma run --path /db_path
client = chromadb.HttpClient(host="localhost", port=8000)


embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


collection = client.get_or_create_collection(
    name="Test_Collection",
    embedding_function=embedder,
)

chroma_embeddings_count = collection.count()
print(chroma_embeddings_count)

episode = {"prompt": main.user_prompt, "episode_id": generate(size=12)}
print(episode)

episode_id = generate(size=12)
prompt_text = main.llm_output
metadata = main.metadata



collection.add(
    ids=[episode_id],
    documents=[prompt_text],
    metadatas=[metadata],
)

print("Added:", episode_id)



results = collection.query(
    query_texts=prompt_text,
    n_results=2
)

print(results)





