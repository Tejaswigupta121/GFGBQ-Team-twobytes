import re
import json
import requests

AUTHOR_YEAR_PATTERN = r"\([A-Z][a-zA-Z]+ et al\., \d{4}\)"
NUMERIC_PATTERN = r"\[\d+\]"
URL_PATTERN = r"https?://\S+|doi:\S+"

def extract_citations(text):
    citations = []

    citations += re.findall(AUTHOR_YEAR_PATTERN, text)
    citations += re.findall(NUMERIC_PATTERN, text)
    citations += re.findall(URL_PATTERN, text)

    return citations
if __name__ == "__main__":
    sample = """
    Studies show hallucinations are common (Smith et al., 2021).
    See details in [2] and https://example.com/fake-paper
    """

    print(extract_citations(sample))


def load_corpus_texts():
    with open("data/corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
    return [doc["text"].lower() for doc in corpus]

def validate_author_year(citation, corpus_texts):
    match = re.search(r"\(([A-Za-z]+) et al\., (\d{4})\)", citation)
    if not match:
        return "⚠️ Incomplete"

    author, year = match.groups()

    for text in corpus_texts:
        if author.lower() in text and year in text:
            return "✅ Valid"

    return "❌ Non-existent"


def validate_url(url):
    try:
        r = requests.head(url, timeout=5, allow_redirects=True)
        if r.status_code < 400:
            return "✅ Valid"
        else:
            return "❌ Non-existent"
    except:
        return "❌ Non-existent"
    
def verify_citations(text):
    citations = extract_citations(text)
    corpus_texts = load_corpus_texts()

    results = []

    for c in citations:
        if c.startswith("("):  # Author–Year
            status = validate_author_year(c, corpus_texts)

        elif c.startswith("http") or c.startswith("doi"):
            status = validate_url(c)

        else:  # [1], [2]
            status = "⚠️ Incomplete"

        results.append({
            "citation": c,
            "status": status
        })

    return results

if __name__ == "__main__":
    text = """
    Studies show hallucinations occur frequently (Smith et al., 2021).
    Refer to [3] and https://example.com/nonexistent
    """

    results = verify_citations(text)

    print("🔎 Citation Verification Results:")
    for r in results:
        print(f"{r['citation']} → {r['status']}")

