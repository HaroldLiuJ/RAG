import chromadb
import json
import re
from configs import memory_path, database_path, embedding_model, database_collection_name
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from utils import parse_memories

if __name__ == "__main__":
    # read memory file
    with open(memory_path, 'r') as f:
        raw_memories = json.load(f)["memories"]
    parsed_memories = parse_memories(raw_memories)

    # Initialize Chroma client (persistent database)
    client = chromadb.PersistentClient(path=database_path)

    # Create or get a collection
    huggingface_ef = SentenceTransformerEmbeddingFunction(
        model_name=embedding_model, normalize_embeddings=True
    )
    collection = client.get_or_create_collection(name=database_collection_name, embedding_function=huggingface_ef)

    # Add parsed memories to the collection
    ids = [f"mem-{i}" for i in range(len(parsed_memories))]
    metadatas = [{"user": user} for user, _ in parsed_memories]
    documents = [content for _, content in parsed_memories]
    collection.add(
        ids=ids,
        metadatas=metadatas,
        documents=documents
    )
    print(f"{len(raw_memories)} memories added to the collection {database_collection_name}.")
