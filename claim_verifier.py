import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =========================
# Load Models
# =========================

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading NLI model...")
nli = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    top_k=None   # replaces deprecated return_all_scores
)

# =========================
# Load FAISS Index & Corpus
# =========================

print("Loading FAISS index...")
faiss_index = faiss.read_index("data/corpus.index")

print("Loading corpus...")
with open("data/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

# =========================
# Evidence Retrieval
# =========================

def retrieve_evidence(claim, top_k=3):
    embedding = embedder.encode([claim])
    embedding = np.array(embedding).astype("float32")

    distances, indices = faiss_index.search(embedding, top_k)
    return [corpus[i]["text"] for i in indices[0]]

# =========================
# Claim Verification (MNLI)
# =========================

LABEL_MAP = {
    "ENTAILMENT": "Supported",
    "CONTRADICTION": "Contradicted",
    "NEUTRAL": "Not enough information"
}


def generate_explanation(label, evidence):
    if label == "Supported":
        return "Claim is supported by retrieved evidence from the corpus."

    if label == "Contradicted":
        return "Retrieved evidence contradicts the claim."

    return "No sufficient supporting evidence found in the indexed corpus."


def verify_claim(claim, evidence_docs):
    best_label = "Not enough information"
    best_confidence = 0.0
    best_evidence = None

    for doc in evidence_docs:
        results = nli(doc, text_pair=claim)[0]

        for r in results:
            score = r["score"]
            label = r["label"]

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
# Full Pipeline
# =========================

def verify_claim_pipeline(claim):
    evidence = retrieve_evidence(claim)
    label, confidence = verify_claim(claim, evidence)

    return {
        "claim": claim,
        "label": label,
        "confidence": confidence,
        "evidence": evidence
    }

# =========================
# Test Run
# =========================

if __name__ == "__main__":
    test_claim = "Studies show that large language models hallucinate frequently."

    result = verify_claim_pipeline(test_claim)

    print("\n🔍 Claim:")
    print(result["claim"])

    print("\n📌 Verdict:")
    print(result["label"])

    print("\n📊 Confidence:")
    print(result["confidence"])

    print("\n📚 Evidence:")
    for e in result["evidence"]:
        print("-", e[:200], "...")

def compute_trust_score(results):
    """
    results: list of dicts
    each dict contains {label, confidence, explanation}
    """
    if not results:
        return 0

    verified = sum(1 for r in results if r["label"] == "Supported")
    total = len(results)

    return round((verified / total) * 100, 2)

