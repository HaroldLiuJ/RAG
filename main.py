import subprocess

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from utils import retrieve_context, generate_personalized_prompt
import chromadb
from configs import database_path, database_collection_name, retrieval_threshold, model, embedding_model
from ollama import chat, ChatResponse, pull


def run_ollama(prompt, model="gemma3"):
    """
    Run an Ollama model. If the model is not available locally,
    it will be pulled automatically.
    """
    # Try to pull the model (safe even if it already exists)
    pull(model)

    # Run the chat
    response: ChatResponse = chat(model=model, messages=[
        {"role": "user", "content": prompt}
    ])
    return response.message.content


def answer_query(query, user_id, collection, model="llama2", threshold=0.6):
    context = retrieve_context(query, collection, user_id, threshold=threshold)
    prompt = generate_personalized_prompt(query, context)
    response = run_ollama(prompt, model=model)
    return response.strip()


if __name__ == "__main__":


    # Initialize Chroma client (persistent database)
    client = chromadb.PersistentClient(path=database_path)
    # Get the collection
    huggingface_ef = SentenceTransformerEmbeddingFunction(
        model_name=embedding_model, normalize_embeddings=True
    )
    collection = client.get_collection(name=database_collection_name, embedding_function=huggingface_ef)

    # Example usage
    user_id = "user-001"
    query = "What's my favorite movie?"

    print("-"*60)
    print("User ID:", user_id)
    print("Query:", query)
    print("-" * 60)

    context = retrieve_context(query, collection, user_id, threshold=retrieval_threshold)
    print("Retrieved Context:", context)
    print("-" * 60)

    prompt = generate_personalized_prompt(query, context)
    response = run_ollama(prompt, model=model).strip()
    print(f"Response: {response}")


