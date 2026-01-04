import wikipediaapi
import json
import os

WIKI_TOPICS = [
    "Artificial intelligence",
    "Large language models",
    "Natural language processing",
    "Machine learning",
    "Hallucination (artificial intelligence)"
]

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="AI-Hallucination-Checker/1.0"
)

documents = []

for topic in WIKI_TOPICS:
    page = wiki.page(topic)
    if page.exists():
        documents.append({
            "id": f"wiki_{topic.replace(' ', '_')}",
            "text": page.summary,
            "source": "Wikipedia"
        })

os.makedirs("data", exist_ok=True)

with open("data/wiki_corpus.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=2)

print(f"Saved {len(documents)} Wikipedia documents")
