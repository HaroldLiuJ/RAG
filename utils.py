import re
def retrieve_context(query, collection, user_id, top_k=3, threshold=0.6):
    """
    Retrieve the most relevant memories for a given query, restricted to a specific user.

    Args:
        query (str): The user query string.
        collection (chromadb.api.models.Collection): The ChromaDB collection to search.
        user_id (str): The user identifier (e.g., "user-001").
        top_k (int): Maximum number of memories to return.
        threshold (float): Minimum similarity threshold (0–1) for relevance.

    Returns:
        list of dict: Top relevant memories with document + metadata (or empty list if none match).
    """
    # Query the collection, filtering by user metadata
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"user": user_id}   # filter by same user
    )

    # Extract results
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    # Convert cosine distance to similarity (1 - distance)
    similarities = [1 - d for d in distances]

    # Collect only relevant results
    relevant_memories = [
        {"memory": doc, "similarity": sim, "metadata": meta}
        for doc, sim, meta in zip(documents, similarities, metadatas)
        if sim >= threshold
    ]

    return relevant_memories

def parse_memories(raw_memories):
    """
    Parse raw memory strings into structured format.
    Example input: ['user-1: "I love programming."', 'user-2: "Python is my favorite language."']
    Example output: [('user-1', 'I love programming.'), ('user-2, 'Python is my favorite language.')]

    Args:
        raw_memories (list of str): List of raw memory strings.
    Returns:
        list of tuples: Parsed memories as (user, content) tuples.
    """

    parsed = []
    for mem in raw_memories:
        match = re.match(r'^(user-\d+):\s*"?(.+?)"?$', mem)
        if match:
            user = match.group(1)
            content = match.group(2)
            parsed.append((user, content))
    return parsed


def generate_personalized_prompt(query, context):
    """
    Generate a personalized prompt for the LLM using retrieved memories.

    Args:
        query (str): The original user query.
        context (list of dict or list of str): Retrieved memories. Can be empty.

    Returns:
        str: Final crafted prompt for the LLM.
    """

    if not context:
        # No relevant context found → return query as-is
        default_prompt = f"""
            You are an assistant. Answer the following user query thoughtfully:
            User Query:
            {query}
            Instructions:
            - Do not make up information about the user.
            - If you don't know the answer, just say you don't know.
            - Be clear, concise, and helpful.
            """
        return default_prompt

    # If context exists, format it
    if isinstance(context[0], dict):
        # If using dicts (memory, similarity, metadata)
        context_texts = [c["memory"] for c in context]
    else:
        # If using plain strings
        context_texts = context

    context_str = "\n".join([f"- {c}" for c in context_texts])

    # Build the final prompt
    prompt = f"""You are an personalized assistant. Use the following personal context 
    to tailor your answer to the user. The memories are facts about the user and 
    should be respected when crafting the response.
    
    Relevant Memories:
    {context_str}
    
    User Query:
    {query}
    
    Instructions:
    - Ground your response in the provided memories.
    - If the context seems relevant, weave it naturally into your answer.
    - If the memories are not relevant, ignore them and answer normally.
    - Do not make up information about the user.
    - If you don't know the answer, just say you don't know.
    - Be clear, concise, and helpful.
    """

    return prompt
