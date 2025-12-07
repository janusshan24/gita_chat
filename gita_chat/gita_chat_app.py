import streamlit as st
import os
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- CONFIGURATION ---
INDEX_DIR = "index_storage"

# --- 1. SETUP AND INITIALIZATION ---

# Set up the page title and initial messages
st.set_page_config(page_title="Bhagavad Gita Chatbot", layout="wide")
st.title("🕉️ Bhagavad Gita Scholar")

# Check if the index is available
if not os.path.exists(INDEX_DIR):
    st.error(
        f"Error: Index storage directory '{INDEX_DIR}' not found. "
        "Please ensure your index is located in the root of your project on GitHub."
    )
    st.stop()
    
# Function to initialize the RAG pipeline (runs only once)
@st.cache_resource(show_spinner=True)
def initialize_rag_pipeline():
    # Configure the LLM using Streamlit Secrets
    try:
        # 1. Access the API key from Streamlit secrets
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        # Halt deployment if the secret is missing
        st.error("GEMINI_API_KEY not found in Streamlit secrets. Please configure it in the 'Manage app' settings.")
        st.stop()
    
    # 2. Configure the Embedding Model (Must match index builder)
    st.write("Loading embedding model...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    # 3. Configure the LLM (Gemini Cloud API)
    st.write("Connecting to Gemini Cloud API...")
    Settings.llm = Gemini(
        model="gemini-2.5-flash",
        api_key=api_key, # Use the retrieved secret
        request_timeout=120
    )
    
    # 4. Load the Index from Storage
    st.write("Loading vector index...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)

    # 5. Create the Query Engine
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
                # Removed reference to Ollama in error message
                error_msg = f"An error occurred during query: {e}."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
