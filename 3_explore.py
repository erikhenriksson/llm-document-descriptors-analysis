import json
from collections import Counter

# Load data
documents = []
with open("processed/bbc_harmonized_with_labels_threshold_1.jsonl", "r") as f:
    for line in f:
        documents.append(json.loads(line))

# Extract descriptors (treat as set per document)
all_descriptors = []
for doc in documents:
    descriptors = set()
    for desc in doc["harmonized_descriptors"]:
        first_part = desc.split(";")[0].strip()
        descriptors.add(first_part)
    all_descriptors.extend(descriptors)

# Count how many documents each descriptor appears in
descriptor_counts = Counter(all_descriptors)

print("=" * 60)
print("DESCRIPTOR FREQUENCY DISTRIBUTION")
print("=" * 60)
print()

# Count descriptors by frequency threshold
frequency_buckets = [1, 2, 3, 4, 5, 10, 20, 50, 100]
for threshold in frequency_buckets:
    count = sum(1 for freq in descriptor_counts.values() if freq == threshold)
    print(f"Descriptors appearing in exactly {threshold:3d} documents: {count:5d}")

print()
print("=" * 60)
print("CUMULATIVE FILTERING IMPACT")
print("=" * 60)
print()

# Show what happens if we filter at different thresholds
total_descriptors = len(descriptor_counts)
for min_freq in [1, 2, 3, 4, 5, 10, 20]:
    remaining = sum(1 for freq in descriptor_counts.values() if freq >= min_freq)
    removed = total_descriptors - remaining
    pct_removed = (removed / total_descriptors) * 100
    print(
        f"Keep descriptors appearing in >= {min_freq:2d} docs: {remaining:5d} kept, {removed:5d} removed ({pct_removed:5.1f}% removed)"
    )

print()
print("=" * 60)
print("DISTRIBUTION PERCENTILES")
print("=" * 60)
print()

frequencies = sorted(descriptor_counts.values(), reverse=True)
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    idx = int(len(frequencies) * p / 100)
    print(f"{p:2d}th percentile: descriptor appears in {frequencies[idx]} documents")

print()
print(f"Median frequency: {frequencies[len(frequencies) // 2]}")
print(f"Mean frequency: {sum(frequencies) / len(frequencies):.2f}")
