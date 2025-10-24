import json

# Load filtered data
documents = []
with open("processed/bbc_harmonized_with_labels_threshold_1.jsonl", "r") as f:
    for line in f:
        documents.append(json.loads(line))

# Find documents with 0 descriptors
zero_descriptor_docs = []
for doc in documents:
    # Extract unique descriptors
    descriptors = set()
    for desc in doc["harmonized_descriptors"]:
        descriptors.add(desc.split(";")[0].strip())

    if len(descriptors) == 0:
        zero_descriptor_docs.append(doc)

print("=" * 60)
print("ZERO DESCRIPTOR DOCUMENTS")
print("=" * 60)
print(f"Total documents with 0 descriptors: {len(zero_descriptor_docs)}")
print(f"Percentage of dataset: {len(zero_descriptor_docs) / len(documents) * 100:.1f}%")
print()

# Category breakdown
from collections import Counter

categories = Counter([doc["label_text"] for doc in zero_descriptor_docs])
print("Category distribution:")
for category, count in categories.most_common():
    print(f"  {category}: {count}")
print()

# Show 3 examples
print("=" * 60)
print("SAMPLE DOCUMENTS WITH 0 DESCRIPTORS")
print("=" * 60)
for i, doc in enumerate(zero_descriptor_docs[:3]):
    print(f"\nExample {i + 1}:")
    print(f"  Label: {doc['label_text']}")
    if "text" in doc:
        # Show first 200 chars if text exists
        text_preview = doc["text"][:200].replace("\n", " ")
        print(f"  Text preview: {text_preview}...")
