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
    top_k=None   # IMPORTANT: returns list of label-score dicts
)

# =========================
# Load FAISS Index & Corpus
# =========================

print("📦 Loading FAISS index...")
faiss_index = faiss.read_index("data/corpus.index")

print("📚 Loading corpus...")
with open("data/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

# =========================
# Evidence Retrieval
# =========================

def retrieve_evidence(claim, top_k=3):
    """
    Retrieves top-k relevant documents from FAISS index
    """
    embedding = embedder.encode([claim])
    embedding = np.array(embedding).astype("float32")

    _, indices = faiss_index.search(embedding, top_k)

    return [corpus[i]["text"] for i in indices[0]]

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
        evidence: str
        explanation: str
    }
    """

    best_label = "Not enough information"
    best_confidence = 0.0
    best_evidence = None

    for doc in evidence_docs:
        # BART-MNLI expects: premise=doc, hypothesis=claim
        results = nli(
            doc,
            text_pair=claim,
            top_k=None
        )

        for r in results:
            score = r.get("score", 0.0)
            label = r.get("label", "")

            if score > best_confidence:
                best_confidence = score
                best_label = LABEL_MAP.get(label, "Not enough information")
                best_evidence = doc

    explanation = generate_explanation(best_label, best_evidence)

    return {
        "label": best_label,
        "confidence": round(best_confidence, 3),
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
    print("📚 Evidence:", result["evidence"][:200], "...")
