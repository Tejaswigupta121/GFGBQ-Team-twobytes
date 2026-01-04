import nltk
import re
from nltk.tokenize import sent_tokenize

# =========================
# NLTK SETUP (Streamlit-safe)
# =========================
nltk.download("punkt", quiet=True)

# =========================
# CLAIM HEURISTICS
# =========================

CLAIM_KEYWORDS = [
    "according to",
    "studies show",
    "study shows",
    "research indicates",
    "research shows",
    "reported that",
    "found that",
    "suggests",
    "evidence shows"
]

# =========================
# HELPERS
# =========================

def split_sentences(text: str):
    return sent_tokenize(text)


def contains_number(sentence: str) -> bool:
    return bool(re.search(r"\d", sentence))


def is_claim(sentence: str) -> bool:
    s = sentence.lower()

    # Keyword-based claims
    if any(keyword in s for keyword in CLAIM_KEYWORDS):
        return True

    # Numeric claims (percentages, years, stats)
    if contains_number(sentence):
        return True

    return False


# =========================
# MAIN API (USED BY app.py)
# =========================

def extract_claims(text: str):
    """
    Extracts factual claims from input text.
    """
    sentences = split_sentences(text)
    claims = [s for s in sentences if is_claim(s)]
    return claims
