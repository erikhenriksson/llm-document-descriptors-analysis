"""
Analyze predictions with probability buckets (for skewed distributions)
"""

import json
from collections import Counter

# =============================================================================
# CONFIG
# =============================================================================

INPUT_PATH = "data/descriptors_fineweb-edu_harmonized_labelled_with_review_pred.jsonl"
MAX_TEXT_LENGTH = 500
SAMPLES_PER_BUCKET = 5

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading data...")
rows = []
with open(INPUT_PATH, "r") as f:
    for line in f:
        rows.append(json.loads(line))

print(f"Loaded {len(rows):,} rows")

# =============================================================================
# DISTRIBUTION
# =============================================================================

print()
print("=" * 80)
print("PROBABILITY DISTRIBUTION")
print("=" * 80)

# Define buckets
buckets = [
    (0.99, 1.00),
    (0.95, 0.99),
    (0.90, 0.95),
    (0.80, 0.90),
    (0.70, 0.80),
    (0.60, 0.70),
    (0.50, 0.60),
    (0.40, 0.50),
    (0.30, 0.40),
    (0.20, 0.30),
    (0.10, 0.20),
    (0.05, 0.10),
    (0.01, 0.05),
    (0.00, 0.01),
]

bucket_rows = {b: [] for b in buckets}

for row in rows:
    p = row["review_prob"]
    for low, high in buckets:
        if low <= p < high or (high == 1.00 and p == 1.00):
            bucket_rows[(low, high)].append(row)
            break

print()
print(f"{'Bucket':<15} {'Count':>10} {'Pct':>8}  Bar")
print("-" * 60)

for low, high in buckets:
    count = len(bucket_rows[(low, high)])
    pct = 100 * count / len(rows)
    bar = "#" * int(pct / 2)
    print(f"{low:.2f} - {high:.2f}    {count:>10,} {pct:>7.2f}%  {bar}")

print()
print(
    f"Total predicted as review (prob >= 0.5): {sum(1 for r in rows if r['review_prob'] >= 0.5):,}"
)
print(
    f"Total predicted as not review (prob < 0.5): {sum(1 for r in rows if r['review_prob'] < 0.5):,}"
)

# =============================================================================
# SAMPLE FROM INTERESTING BUCKETS
# =============================================================================


def truncate(text, max_len):
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# Focus on: high confidence reviews, boundary cases, high confidence not-reviews
interesting_buckets = [
    (0.99, 1.00, "HIGH CONFIDENCE REVIEWS (0.99-1.00)"),
    (0.90, 0.95, "CONFIDENT REVIEWS (0.90-0.95)"),
    (0.50, 0.60, "BORDERLINE - LEANING REVIEW (0.50-0.60)"),
    (0.40, 0.50, "BORDERLINE - LEANING NOT REVIEW (0.40-0.50)"),
    (0.10, 0.20, "CONFIDENT NOT REVIEWS (0.10-0.20)"),
    (0.01, 0.05, "HIGH CONFIDENCE NOT REVIEWS (0.01-0.05)"),
    (0.00, 0.01, "VERY HIGH CONFIDENCE NOT REVIEWS (0.00-0.01)"),
]

for low, high, label in interesting_buckets:
    bucket = bucket_rows.get((low, high), [])

    print()
    print("=" * 80)
    print(f"{label} - {len(bucket):,} total")
    print("=" * 80)

    if len(bucket) == 0:
        print("(no samples in this bucket)")
        continue

    # Sample evenly from bucket
    step = max(1, len(bucket) // SAMPLES_PER_BUCKET)
    samples = bucket[::step][:SAMPLES_PER_BUCKET]

    for i, row in enumerate(samples, 1):
        prob = row["review_prob"]
        text = truncate(row["document"], MAX_TEXT_LENGTH)

        print()
        print(f"--- Sample {i} | prob={prob:.4f} ---")
        print(text)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
