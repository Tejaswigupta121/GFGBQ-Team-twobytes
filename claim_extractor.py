import nltk
from nltk.tokenize import sent_tokenize
import re

def split_sentences(text):
    return sent_tokenize(text)
text = "LLMs hallucinate. Studies show hallucination rates above 60%."
print(split_sentences(text))
CLAIM_KEYWORDS = [
    "according to",
    "studies show",
    "research indicates",
    "research shows",
    "reported that",
    "found that",
    "suggests",
    "evidence shows"
]

def contains_number(sentence):
    return bool(re.search(r"\d", sentence))
def is_claim(sentence):
    s = sentence.lower()

    # Keyword-based
    if any(keyword in s for keyword in CLAIM_KEYWORDS):
        return True

    # Numeric claims
    if contains_number(sentence):
        return True

    return False
def extract_claims(text):
    sentences = split_sentences(text)
    claims = [s for s in sentences if is_claim(s)]
    return claims
if __name__ == "__main__":
    sample_text = """
    Large language models are widely used.
    Studies show LLMs hallucinate in over 60% of cases.
    According to recent research, citation errors are common.
    Smith et al. (2021) proposed hallucination mitigation.
    """

    claims = extract_claims(sample_text)

    print("🔍 Extracted Claims:")
    for c in claims:
        print("-", c)

