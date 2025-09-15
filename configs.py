memory_path = "./memories/sample_memories.json"
database_path = "./database"
database_collection_name = "mems"

# embedding model
embedding_model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# retrieval settings
retrieval_threshold = 0.5  # similarity threshold for retrieval
top_k = 3  # number of top relevant memories to retrieve

# LLM model
model = "gemma3:4b"
