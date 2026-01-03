from sentence_transformers import SentenceTransformer
from transformers import pipeline

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading NLI model...")
nli = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli"
)

print(" Models loaded successfully....!")
