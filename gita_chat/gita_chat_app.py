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
        system_prompt = (
            "You are a wise and objective scholar specializing in the Bhagavad Gita. "
            "You have two main duties: "
            "1. For general conversation (like 'Hello' or 'How are you?'), answer politely and naturally using your base knowledge. "
            "2. For questions about the Bhagavad Gita, verses, or concepts, answer ONLY based on the provided context (Sanskrit verse, translation, and purport). "
            "Maintain a continuous conversation, referencing previous turns. "
            "Render all Sanskrit diacritics correctly (e.g., Kṛṣṇa, dharma-kṣetra)."
        ),
    )
    return query_engine

# Initialize the query engine and store it in session state
query_engine = initialize_rag_pipeline()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! I'm here to help you explore the Bhagavad-gītā. How can I assist you today?"}
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

                # --- CRITICAL FIX: Add Relevance Filtering Here ---
                if response.source_nodes:
                    # Get the score of the most relevant node (the first one)
                    top_score = response.source_nodes[0].score 
                    
                    # Set a relevance threshold. Adjust this value (e.g., 0.70 to 0.85) based on testing.
                    RELEVANCE_THRESHOLD = 0.75 
                    
                    if top_score and top_score >= RELEVANCE_THRESHOLD:
                        # If the top score is high, assume it's a RAG question and display sources
                        
                        # Display the source using st.expander for a clean UI (as suggested previously)
                        metadata = response.source_nodes[0].metadata
                        
                        # Display a custom alert for the source
                        st.info(f"Source Verse: **{metadata.get('chapter_verse', 'N/A')}** (Relevance: {top_score:.2f})")
                        
                        with st.expander("Show Detailed Source Context"):
                            st.markdown(f"**Sanskrit:** {metadata.get('sanskrit', 'N/A')}")
                            st.markdown(f"**Translation:** {metadata.get('translation', 'N/A')}")
                            st.markdown("---")
                            st.markdown(f"**Purport Snippet:** {response.source_nodes[0].get_text()[:300]}...")
                            
                    else:
                        # If the score is too low, do NOT display any sources.
                        st.caption("*(Answer provided using base knowledge.)*")
                    
                # 3. Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": response.response})
                
            except Exception as e:
                # Removed reference to Ollama in error message
                error_msg = f"An error occurred during query: {e}."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
