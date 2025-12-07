import streamlit as st
import os
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
# CRITICAL IMPORT for Memory Synchronization
from llama_index.core.llms import ChatMessage, MessageRole 

# --- CONFIGURATION ---
INDEX_DIR = "index_storage"

# --- 1. SETUP AND INITIALIZATION ---

st.set_page_config(page_title="Bhagavad Gita Chatbot", layout="wide")
st.title("🕉️ Bhagavad Gita Scholar")

# Check if the index is available
if not os.path.exists(INDEX_DIR):
    st.error(
        f"Error: Index storage directory '{INDEX_DIR}' not found. "
        "Please ensure your index is uploaded to the root of your GitHub repository."
    )
    st.stop()

# Function to initialize the RAG pipeline (runs only once)
@st.cache_resource(show_spinner=True)
def initialize_chat_pipeline():
    # 1. Configure Secrets & LLM
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("GEMINI_API_KEY not found in Streamlit secrets.")
        st.stop()
        
    st.write("Connecting to Gemini Cloud API...")
    Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash", 
        api_key=api_key,
        request_timeout=120
    )

    # 2. Configure Embedding Model
    st.write("Loading embedding model...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        device="cpu"
    )
    
    # 3. Load the Index
    st.write("Loading vector index...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)

    # 4. Initialize Memory Buffer
    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    
    # 5. Create the Chat Engine (CRITICAL CHANGE HERE)
    st.write("Creating chat engine...")
    
    # Get the retriever from the index
    retriever = index.as_retriever(similarity_top_k=3) # Use top_k=3 for good context
    
    chat_engine = ContextChatEngine.from_defaults(
        # The ContextChatEngine takes a retriever, not a query_engine
        retriever=retriever,
        memory=memory,
        system_prompt=(
            "You are a wise and objective scholar specializing in the Bhagavad Gita. "
            "Maintain a continuous conversation, referencing previous turns. "
            "Answer the user's question based ONLY on the provided context (Sanskrit verse, translation, and purport). "
            "Render all Sanskrit diacritics correctly (e.g., Kṛṣṇa, dharma-kṣetra)."
        ),
        llm=Settings.llm # Explicitly pass the LLM for clarity
    )
    return chat_engine

# Initialize the chat engine
chat_engine = initialize_chat_pipeline()

# Initialize chat history UI
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! I am a RAG scholar of the Bhagavad Gita. How can I help you today?"}
    ]

# --- 2. DISPLAY CHAT HISTORY ---

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. HANDLE USER INPUT (WITH MEMORY SYNCHRONIZATION) ---

if prompt := st.chat_input("Ask a question about a verse, chapter, or concept..."):
    # 1. Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate response from the RAG engine
    with st.chat_message("assistant"):
        with st.spinner("Meditating on the answer..."):
            try:
                # --- CRITICAL MEMORY FIX ---
                # 2a. Reset the engine's internal memory buffer (Now accessed correctly)
                # The ContextChatEngine's memory object is accessed via the 'memory' attribute
                # of the engine, which has the 'reset' method.
                chat_engine.memory.reset() 

                # 2b. Re-load the Streamlit history into the LlamaIndex memory buffer
                # Access the internal history list of the memory object to fill it
                for message in st.session_state.messages:
                    role = MessageRole.USER if message["role"] == "user" else MessageRole.ASSISTANT
                    chat_engine.memory.put(ChatMessage(role=role, content=message["content"]))
                
                # 2c. Run the LlamaIndex chat
                response = chat_engine.chat(prompt)
                
                # 2d. Display the response and sources
                st.markdown(response.response)
                
                if response.source_nodes:
                    st.info(f"Source Verse: **{response.source_nodes[0].metadata.get('chapter_verse', 'N/A')}**")
                    
                # 3. Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": response.response})
                
            except Exception as e:
                error_msg = f"An error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
