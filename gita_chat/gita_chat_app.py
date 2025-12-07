import streamlit as st
import os
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure the LLM using Streamlit Secrets
try:
    # 1. Access the API key from Streamlit secrets
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # Fallback for local testing or if the secret isn't set
    st.error("GEMINI_API_KEY not found in secrets. Please set it up.")
    st.stop()

# 2. Initialize the LLM with the API key and a suitable model
llm = Gemini(model="gemini-2.5-flash", api_key=api_key)

# --- CONFIGURATION ---
INDEX_DIR = "index_storage"
OLLAMA_MODEL = "llama3:8b-instruct-q4_0"

# --- 1. SETUP AND INITIALIZATION ---

# Set up the page title and initial messages
st.set_page_config(page_title="Bhagavad Gita Chatbot", layout="wide")
st.title("🕉️ Bhagavad Gita Scholar")

# Check if the index is available
if not os.path.exists(INDEX_DIR):
    st.error(
        f"Error: Index storage directory '{INDEX_DIR}' not found. "
        "Please run `build_index.py` first to create the RAG index."
    )
    st.stop()
    
# Function to initialize the RAG pipeline (runs only once)
@st.cache_resource(show_spinner=False)
def initialize_rag_pipeline():
    # 1. Access the API key securely
    gemini_api_key = st.secrets["GEMINI_API_KEY"]

    # 2. Configure the LLM using the correct class and API key
    Settings.llm = Gemini(
        model="gemini-2.5-flash", 
        api_key=gemini_api_key, # Pass the key here
        request_timeout=120.0
    )
    
    # 3. Load the Index from Storage
    st.write("Loading vector index...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)

    # 4. Create the Query Engine (with a system prompt for good style)
    st.write("Creating query engine...")
    query_engine = index.as_query_engine(
        system_prompt=(
            "You are a wise and objective scholar specializing in the Bhagavad Gita. "
            "Answer the user's question based ONLY on the provided context (Sanskrit verse, translation, and purport). "
            "Render all Sanskrit diacritics correctly (e.g., Kṛṣṇa, dharma-kṣetra) and do not use corrupted characters."
        ),
    )
    return query_engine

# Initialize the query engine and store it in session state
query_engine = initialize_rag_pipeline()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! I am a RAG scholar of the Bhagavad Gita. How can I help you today?"}
    ]

# --- 2. DISPLAY CHAT HISTORY ---

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. HANDLE USER INPUT AND RESPONSE ---

if prompt := st.chat_input("Ask a question about a verse, chapter, or concept..."):
    # 1. Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate response from the RAG engine
    with st.chat_message("assistant"):
        with st.spinner("Meditating on the answer..."):
            try:
                # Run the LlamaIndex query
                response = query_engine.query(prompt)
                
                # Display the response
                st.markdown(response.response)
                
                # Optionally display sources (very helpful for RAG)
                if response.source_nodes:
                    st.info(f"Source Verse: **{response.source_nodes[0].metadata.get('chapter_verse', 'N/A')}**")
                    
                # 3. Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": response.response})
                
            except Exception as e:
                error_msg = f"An error occurred during query: {e}. Check if your Ollama server is running."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
