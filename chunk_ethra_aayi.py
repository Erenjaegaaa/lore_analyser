import json
with open("./data/chunks/chunks.json") as f:
    chunks = json.load(f)
if isinstance(chunks, dict):
    chunks = list(chunks.values())
for i, c in enumerate(chunks):
    if c["chunk_id"] == "c3_89owyn_003":
        print(i)
        break