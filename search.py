import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load index
index = faiss.read_index("data/corpus.index")

# Load doc IDs
with open("data/doc_ids.json", "r") as f:
    doc_ids = json.load(f)

# Load corpus
with open("data/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        results.append(corpus[idx])

    return results

# Test
results = search("AI hallucination in large language models")

for r in results:
    print("-", r["text"])
