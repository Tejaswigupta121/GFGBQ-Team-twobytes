import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load corpus
with open("data/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

texts = [doc["text"] for doc in corpus]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Convert to numpy
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, "data/corpus.index")

# Save metadata
with open("data/doc_ids.json", "w") as f:
    json.dump([doc["id"] for doc in corpus], f)

print("FAISS index built successfully")
