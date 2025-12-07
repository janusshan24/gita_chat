import streamlit as st
import os
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer

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
    Settings.llm = Gemini(
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
    
    # 5. Create the Chat Engine
    st.write("Creating chat engine...")
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=index.as_query_engine(), 
        memory=memory,
        system_prompt=(
            "You are a wise and objective scholar specializing in the Bhagavad Gita. "
            "Maintain a continuous conversation, referencing previous turns. "
            "Answer the user's question based ONLY on the provided context (Sanskrit verse, translation, and purport). "
            "Render all Sanskrit diacritics correctly (e.g., Kṛṣṇa, dharma-kṣetra)."
        ),
        verbose=True
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

# --- 3. HANDLE USER INPUT ---

if prompt := st.chat_input("Ask a question about a verse, chapter, or concept..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Meditating on the answer..."):
            try:
                # Use the chat engine with history
                response = chat_engine.chat(prompt)
                
                st.markdown(response.response)
                
                if response.source_nodes:
                    st.info(f"Source Verse: **{response.source_nodes[0].metadata.get('chapter_verse', 'N/A')}**")
                    
                st.session_state.messages.append({"role": "assistant", "content": response.response})
                
            except Exception as e:
                error_msg = f"An error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
