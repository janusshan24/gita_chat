
from fastapi import FastAPI, HTTPException
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # <-- NEW IMPORT
import os

app = FastAPI()

query_engine = None
INDEX_DIR = "index_storage"

@app.on_event("startup")
async def startup_event():
    global query_engine

    if not os.path.exists(INDEX_DIR):
        # ... (error handling) ...
        pass # Assuming this check passes

    print("✅ Initializing LlamaIndex components...")
    
    try:
        # 1. CONFIGURE EMBEDDING MODEL (Crucial Fix)
        # This must match the model used when the index was built.
        # Otherwise, LlamaIndex defaults to OpenAI and fails.
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = embed_model
        print("✅ Configured HuggingFace Embedding Model.")

        # 2. Configure the LLM (Ollama)
        Settings.llm = Ollama(
            model="llama3:8b-instruct-q4_0",
            request_timeout=120
        )

        print(f"✅ Configured Ollama LLM ({Settings.llm.model}).")
        
        # 3. Load the Index from Storage (Now it uses the configured embed_model)
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context)
        
        # 4. Create the Query Engine
        query_engine = index.as_query_engine()
        print("✅ Index and Query Engine loaded successfully.")

    except Exception as e:
        print(f"FATAL ERROR during startup: {e}")
        raise RuntimeError("Failed to initialize RAG components.")


@app.get("/ask")
# ... (rest of the code is unchanged)
def ask(q: str):
    if query_engine is None:
        raise HTTPException(status_code=503, detail="RAG system is still starting up or failed to load.")

    try:
        ans = query_engine.query(q)
        return {"answer": str(ans)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")