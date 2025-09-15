# Personalized Memory QA Writeup

## Overview
This project demonstrates a retrieval-augmented generation (RAG) pipeline that personalizes responses using stored user memories. It covers the full loop:  

1. **Embedding**: Encode user memories and queries with a SentenceTransformer model.  
2. **Store**: Persist embeddings in a ChromaDB vector store.  
3. **Retrieve**: Query the database for the most relevant memories.  
4. **Generate**: Construct a personalized prompt and send it to an Ollama LLM for response generation.  

---

### Directory Structure
```
rag_adobe/
├── database/                  # ChromaDB persistent storage
├── memories/                  # Memory-related data and scripts
│   └── sample_memories.json   # Example memory JSON file (if included here)
├── memory_indexing.py         # Script for parsing and indexing user memories into ChromaDB
├── main.py                    # End-to-end answering pipeline (retrieval + Ollama response)
├── configs.py                 # Centralized configuration (paths, model names, etc.)
├── utils.py                   # Helper functions (e.g., retrieve_context, generate_personalized_prompt)
├── README.md                  # Project documentation

```
---

## Pipeline Steps
1. Memories are provided in JSON, e.g.:  
   ```json
   {
     "memories": [
       "user-001: \"Python is my favorite programming language.\"",
       "user-001: \"I prefer Windows for development.\""
     ]
   }
   ```
2. Memories are parsed into `(user, content)` tuples and stored in ChromaDB with metadata.  
3. When a new query arrives, the system:  
   - Embeds the query with the same model as the stored memories.  
   - Performs a similarity search, restricted to the same user ID.  
   - Applies a similarity threshold to filter results.  
   - Returns the top 2–3 relevant matches if they pass the threshold.  
4. A structured, personalized prompt is generated, containing the relevant memories, the user query, and explicit instructions for the model.  
5. The prompt is sent to Ollama (default model: `gemma3:4B`). If the model is not present locally, it is pulled automatically.  

---



## Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd rag_adobe
```

### 2. Create a Virtual Environment
It’s recommended to use Python 3.11 with anaconda. The following commands shows the environment setup using conda for MacOS. If you are running on other OS, please first install pytorch 2.8 for your OS.

```bash
conda create -n py311 python=3.11
conda activate py311
```

### 3. Install Dependencies
```bash
pin install torch torchvision
pip install -r requirements.txt
```

### 4. Run Memory Indexing
Before asking questions, you need to embed and store memories into ChromaDB.  

```bash
python memory_indexing.py
```

This will parse `sample_memories.json`, embed them, and persist them into the `database/` folder.  

### 5. Run the QA Pipeline
```bash
python main.py
```
**Exemplar Case 1: Context Match**
```
------------------------------------------------------------
User ID: user-001
Query: Can you please give me some music recommendations?
------------------------------------------------------------
Retrieved Context: [{'memory': "Rock is the user's preferred genre.", 'similarity': 0.6428422927856445, 'metadata': {'user': 'user-001'}}]
------------------------------------------------------------
Response: Okay, based on your preference for rock music, here are a few recommendations:

*   **The Black Keys:** They’ve got a really great bluesy rock sound.
*   **Royal Blood:** If you’re into driving, heavy riffs, they’re fantastic.
*   **Arctic Monkeys:** A bit more modern rock with some fantastic songwriting.

To help me narrow it down further and give you even better suggestions, is there any specific subgenre of rock you’re in the mood for today?
```


**Exemplar Case 2: Context Not Match**
```
------------------------------------------------------------
User ID: user-001
Query: Can you please give me some movie recommendations?
------------------------------------------------------------
Retrieved Context: []
------------------------------------------------------------
Response: Okay, I can definitely offer some movie recommendations! To give you the best suggestions, could you tell me a little bit about what you enjoy? 

For example, do you have any preferences regarding:

*   **Genre?** (e.g., comedy, action, drama, sci-fi, horror, romance, documentary)
*   **Specific themes or topics?** (e.g., space exploration, historical events, coming-of-age stories)
*   **Are there any actors or directors you particularly like?**
*   **Are there any movies you've really enjoyed recently?**

If you don’t have any preferences, I can offer some popular and well-regarded films across different genres.

Otherwise, here are a few starting points:

*   **Paddington 2:** (Family, Comedy) – Universally loved and charming.
*   **Parasite:** (Thriller, Drama, Comedy) – A critically acclaimed and suspenseful film.
*   **Everything Everywhere All at Once:** (Action, Sci-Fi, Comedy) - A mind-bending and emotional adventure.

I look forward to hearing more about what you’re in the mood for!
```
**Exemplar Case 3: Factual Test**
```
------------------------------------------------------------
User ID: user-001
Query: What's my favorite movie?
------------------------------------------------------------
Retrieved Context: []
------------------------------------------------------------
Response: I don't know what your favorite movie is. I have no way of knowing that information.

```
---

## Design Choices
### Embedding Model
- **Model**: `sentence-transformers/multi-qa-mpnet-base-dot-v1`  
- **Rationale**: This model is tuned for QA and semantic similarity tasks, providing embeddings that work well for both short factual memories and natural language queries.  

### Vector Database
- **Database**: ChromaDB  
- **Rationale**: ChromaDB is lightweight, persistent, and integrates cleanly with Python. It also supports metadata filtering, which allows memory retrieval to be restricted to a specific user.  

### Threshold Selection
- **Experimentation**: We tested thresholds at 0.7, 0.6, 0.5, and 0.4, evaluating results through manual inspection.  
  - At **>0.5** (e.g., 0.6 or 0.7), some important but less strongly-matching memories were missed, reducing personalization.  
  - At **<0.5** (e.g., 0.4), irrelevant or weakly related memories began to appear in the context, which risked confusing the model.  
- **Final Choice**: A threshold of **0.5** was selected as the balance point. It ensures that relevant context is retained without introducing noise.  

### Prompt Structure
During testing, we found that LLMs often struggled to effectively use retrieved context unless the prompt was very clearly structured. Therefore, we designed the prompt as follows to let it leverage memory effectively and reduce hallucinations:

- The prompt is structured into three clear sections:  
  - **Relevant Memories**: Retrieved context presented as factual bullet points.  
  - **User Query**: The original question or request.  
  - **Instructions**: Explicit guidance to the model to use the memories if they are relevant, and otherwise ignore them. Additionally, the model is instructed to admit if it does not know the answer rather than fabricating user and answer information.
- This format provides clarity, reduces ambiguity for the LLM, and improves the consistency of personalized responses.

The prompt template is as follows:
```
    You are an personalized assistant. Use the following personal context 
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
```

---

## Limitations

1. **Missing incremental memory updates**  
   Currently, the pipeline indexes memories in bulk from a JSON file. There is no mechanism to add or update memories dynamically at runtime.  

2. **Lack of persistent high-level memory**  
   The system retrieves memories from the vector database for each query, but it does not maintain higher-level, long-term abstractions or summaries that could improve efficiency and personalization. When no memories are relevant, the model defaults to generic behavior without any user-specific context.

3. **Static similarity threshold**  
   The similarity threshold is fixed (0.5). This may not generalize across different users, domains, or embedding models. Adaptive or learned thresholding could yield better results.  


---

