import json
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from unidecode import unidecode # <-- NEW IMPORT

# --- Helper Function to Clean Text ---
import json
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from unidecode import unidecode # <-- Use this for definitive conversion
import unicodedata # Retained for standard normalization check

def clean_text(text: str) -> str:
    """Converts text with diacritics (like ā, ṛ, ṣ) to plain ASCII equivalents (a, r, s)."""
    # Step 1: Normalize any existing Unicode issues (optional but good practice)
    text = unicodedata.normalize('NFKC', text)

    # Step 2: Convert to pure ASCII, replacing non-standard characters
    # This will convert 'Kṛṣṇa' to 'Krishna' or 'Krsna'.
    return unidecode(text).strip()
# ------------------------------------

# Load your JSON file
with open("/Users/janusshan/Documents/gita_chat/bhagavad_gita.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to Documents, applying the cleaning function
documents = []
for d in data:
    # APPLY CLEANING TO ALL FIELDS BEFORE COMBINING
    combined_text = (
        f"Verse: {clean_text(d.get('chapter_verse', ''))}\n"
        f"Sanskrit: {clean_text(d.get('sanskrit', ''))}\n"
        f"Translation: {clean_text(d.get('translation', ''))}\n"
        f"Purport: {clean_text(d.get('purport', ''))}"
    )
    
    doc_id = clean_text(d.get("chapter_verse"))
    
    if doc_id:
        documents.append(
            Document(
                text=combined_text, 
                doc_id=doc_id,
                metadata={
                    "chapter_verse": clean_text(d.get('chapter_verse', '')),
                    "translation": clean_text(d.get('translation', '')),
                }
            )
        )

# --- NEW CONFIGURATION ---
# Setup local embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Set the embedding model using the global Settings object
Settings.embed_model = embed_model
# --- END NEW CONFIGURATION ---

# Build vector store index. 
# It will automatically use the embed_model defined in Settings.
index = VectorStoreIndex.from_documents(documents)

# Save index locally
index.storage_context.persist("index_storage")

print(f"✅ Index created successfully with {len(documents)} documents.")