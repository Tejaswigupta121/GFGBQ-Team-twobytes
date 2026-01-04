import json

with open("data/corpus.json", "r", encoding="utf-8") as f:
    internal = json.load(f)

with open("data/wiki_corpus.json", "r", encoding="utf-8") as f:
    wiki = json.load(f)

merged = internal + wiki

with open("data/corpus.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

print(f"✅ Total documents: {len(merged)}")
