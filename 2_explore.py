import json
from collections import Counter

# Load data
documents = []
with open("processed/bbc_harmonized_with_labels_threshold_1.jsonl", "r") as f:
    for line in f:
        documents.append(json.loads(line))

# Extract descriptors (first part before semicolon) and labels
all_descriptors = []
descriptors_per_doc = []
labels = []

for doc in documents:
    # Extract first part of each descriptor, treat as set to remove duplicates
    descriptors = set()
    for desc in doc["harmonized_descriptors"]:
        first_part = desc.split(";")[0].strip()
        descriptors.add(first_part)

    all_descriptors.extend(descriptors)
    descriptors_per_doc.append(len(descriptors))
    labels.append(doc["label_text"])

# Calculate stats
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total documents: {len(documents)}")
print(f"Total unique categories: {len(set(labels))}")
print()

print("=" * 60)
print("CATEGORY DISTRIBUTION")
print("=" * 60)
label_counts = Counter(labels)
for label, count in label_counts.most_common():
    print(f"{label}: {count}")
print()

print("=" * 60)
print("DESCRIPTOR STATISTICS")
print("=" * 60)
print(f"Total unique descriptors: {len(set(all_descriptors))}")
print(
    f"Avg descriptors per document: {sum(descriptors_per_doc) / len(descriptors_per_doc):.2f}"
)
print(f"Min descriptors per document: {min(descriptors_per_doc)}")
print(f"Max descriptors per document: {max(descriptors_per_doc)}")
print()

print("=" * 60)
print("TOP 20 MOST COMMON DESCRIPTORS")
print("=" * 60)
descriptor_counts = Counter(all_descriptors)
for desc, count in descriptor_counts.most_common(20):
    print(f"{count:4d} | {desc}")
print()

print("=" * 60)
print("SAMPLE DOCUMENT")
print("=" * 60)
sample_doc = documents[0]
print(f"Label: {sample_doc['label_text']}")
print(f"Descriptors:")
descriptors = set()
for desc in sample_doc["harmonized_descriptors"]:
    descriptors.add(desc.split(";")[0].strip())
for desc in sorted(descriptors):
    print(f"  - {desc}")
