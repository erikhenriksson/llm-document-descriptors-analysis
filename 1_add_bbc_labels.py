import json
from datasets import load_dataset

# Load HuggingFace dataset
hf_data = load_dataset("SetFit/bbc-news", split="train")

# Create lookup by text
hf_lookup = {row["text"]: {"label": row["label"], "label_text": row["label_text"]} for row in hf_data}

# Process JSONL
with open("data/bbc_harmonized.jsonl") as f, open("processed/bbc_harmonized_with_labels.jsonl", "w") as out:
    for line in f:
        row = json.loads(line)
        labels = hf_lookup[row["document"]]  # Will fail if text doesn't match
        row.update(labels)
        out.write(json.dumps(row) + "\n")