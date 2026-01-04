import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =========================
# Load Models
# =========================

print("🔄 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("🔄 Loading NLI model...")
nli = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    top_k=None  # IMPORTANT: returns list of label-score dicts
)

# =========================
# Load FAISS Index & Corpus
# =========================
CORPUS_PATH = "data/corpus.json"

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)
print("📦 Loading FAISS index...")
import os

INDEX_PATH = "data/corpus.index"

if os.path.exists(INDEX_PATH):
    print("📦 Loading FAISS index...")
    faiss_index = None
    def load_faiss():
         global faiss_index
         if faiss_index is None:
             faiss_index = faiss.read_index("data/corpus.index")

else:
    print("⚠️ FAISS index not found. Building index...")

    texts = [doc["text"] for doc in corpus]

    embeddings = embedder.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, INDEX_PATH)
    print("✅ FAISS index built and saved")


print("📚 Loading corpus...")
with open("data/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

# =========================
# Evidence Retrieval (SAFE)
# =========================

def retrieve_evidence(claim, top_k=3):
    load_faiss()
    """
    Retrieves top-k relevant documents from FAISS index safely
    Returns full document metadata
    """
    embedding = embedder.encode([claim])
    embedding = np.array(embedding).astype("float32")

    _, indices = faiss_index.search(embedding, top_k)

    evidence_docs = []

    for idx in indices[0]:
        # ❗ SAFETY CHECKS (VERY IMPORTANT)
        if idx == -1:
            continue
        if idx >= len(corpus):
            continue

        doc = corpus[idx]

        # Ensure consistent structure
        if isinstance(doc, dict):
            evidence_docs.append({
                "text": doc.get("text", ""),
                "source": doc.get("source", "Internal Dataset"),
                "id": doc.get("id", "unknown"),
                "url": doc.get("url", "")
            })
        else:
            evidence_docs.append({
                "text": str(doc),
                "source": "Internal Dataset",
                "id": "unknown",
                "url": ""
            })

    return evidence_docs

# =========================
# Label Mapping
# =========================

LABEL_MAP = {
    "ENTAILMENT": "Supported",
    "CONTRADICTION": "Contradicted",
    "NEUTRAL": "Not enough information"
}

# =========================
# Explanation Generator
# =========================

def generate_explanation(label, evidence):
    if label == "Supported":
        return "Claim is supported by retrieved evidence from the corpus."

    if label == "Contradicted":
        return "Retrieved evidence contradicts the claim."

    return "No sufficient supporting evidence found in the indexed corpus."

# =========================
# Claim Verification (MNLI)
# =========================

def verify_claim(claim, evidence_docs):
    """
    Verifies a claim against retrieved evidence using MNLI.

    Returns:
    {
        label: Supported | Contradicted | Not enough information
        confidence: float
        evidence: dict
        explanation: str
    }
    """

    best_label = "Not enough information"
    best_confidence = 0.0
    best_evidence = None

    for doc in evidence_docs:
        premise = doc.get("text", "")
        if not premise:
            continue

        # MNLI inference
        outputs = nli(
            premise,
            text_pair=claim,
            top_k=None
        )

        # outputs is a LIST of dicts
        for r in outputs:
            score = r.get("score", 0.0)
            raw_label = r.get("label", "")

            mapped_label = LABEL_MAP.get(raw_label, "Not enough information")

            if score > best_confidence:
                best_confidence = score
                best_label = mapped_label
                best_evidence = doc

    # 🔒 NUMERIC CLAIM SAFEGUARD (VERY IMPORTANT)
    if best_label == "Supported" and best_evidence:
        claim_has_number = any(char.isdigit() for char in claim)
        evidence_text = best_evidence.get("text", "")
        evidence_has_number = any(char.isdigit() for char in evidence_text)

        # Numeric claim but generic evidence → downgrade
        if claim_has_number and not evidence_has_number:
            best_label = "Not enough information"
            best_confidence = min(best_confidence, 0.6)

    explanation = generate_explanation(best_label, best_evidence)
    if best_label == "Not enough information":
        display_confidence = 1 - best_confidence   # uncertainty-aware
    else:
        display_confidence = best_confidence


    return {
        "label": best_label,
        "confidence": round(display_confidence, 3),
        "evidence": best_evidence,
        "explanation": explanation
    }

# =========================
# Full Verification Pipeline
# =========================

def verify_claim_pipeline(claim):
    """
    End-to-end pipeline:
    claim → retrieve evidence → verify → return structured result
    """
    evidence_docs = retrieve_evidence(claim)
    result = verify_claim(claim, evidence_docs)

    return {
        "claim": claim,
        "label": result["label"],
        "confidence": result["confidence"],
        "evidence": result["evidence"],
        "explanation": result["explanation"]
    }



# =========================
# Trust Score
# =========================

def compute_trust_score(results):
    """
    Trust Score = (Supported Claims / Total Claims) × 100
    """
    if not results:
        return 0

    supported = sum(1 for r in results if r["label"] == "Supported")
    total = len(results)

    return round((supported / total) * 100, 2)

# =========================
# Local Test
# =========================

if __name__ == "__main__":
    test_claim = "Large language models often hallucinate factual information."

    result = verify_claim_pipeline(test_claim)

    print("\n🔍 Claim:", result["claim"])
    print("📌 Verdict:", result["label"])
    print("📊 Confidence:", result["confidence"])
    print("🧠 Explanation:", result["explanation"])
    print("📚 Evidence:", result["evidence"]["text"][:200], "...")
