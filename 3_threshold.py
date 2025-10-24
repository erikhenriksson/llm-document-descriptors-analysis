import json

# Step 1: Load IDs with count >= 10
allowed_ids = set()
with open("data/final_id_counts.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        if item["count"] > 1:
            allowed_ids.add(item["id"])

print(f"Found {len(allowed_ids)} descriptor IDs with count > 1")

# Step 2: Map IDs to descriptor prefixes
id_to_descriptor = {}
with open("data/schema.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        id_to_descriptor[item["id"]] = item["descriptor"]

# Get allowed descriptor prefixes
allowed_descriptors = set()
for desc_id in allowed_ids:
    if desc_id in id_to_descriptor:
        allowed_descriptors.add(id_to_descriptor[desc_id])

print(f"Mapped to {len(allowed_descriptors)} unique descriptor prefixes")

# Step 3: Filter documents
filtered_count = 0
with (
    open("processed/bbc_harmonized_with_labels.jsonl", "r") as f_in,
    open("processed/bbc_harmonized_with_labels_threshold_1.jsonl", "w") as f_out,
):
    for line in f_in:
        doc = json.loads(line)

        # Filter harmonized_descriptors
        filtered_descriptors = []
        for desc in doc["harmonized_descriptors"]:
            prefix = desc.split(";")[0].strip()
            if prefix in allowed_descriptors:
                filtered_descriptors.append(desc)

        # Update document
        doc["harmonized_descriptors"] = filtered_descriptors

        # Write to output
        f_out.write(json.dumps(doc) + "\n")
        filtered_count += 1

print(f"Filtered {filtered_count} documents")
print(f"Saved to: processed/bbc_harmonized_with_labels_threshold_1.jsonl")
