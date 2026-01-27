"""
Sample documents at different probability checkpoints for manual review
"""

import json

# =============================================================================
# CONFIG
# =============================================================================

INPUT_PATH = "data/descriptors_fineweb-edu_harmonized_labelled_with_review_pred.jsonl"
MAX_TEXT_LENGTH = 500  # Truncate long docs for readability

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading data...")
rows = []
with open(INPUT_PATH, "r") as f:
    for line in f:
        rows.append(json.loads(line))

print(f"Loaded {len(rows):,} rows")

# Sort by probability
rows_sorted = sorted(rows, key=lambda x: x["review_prob"], reverse=True)

# =============================================================================
# DEFINE CHECKPOINTS
# =============================================================================

# Top 10, bottom 10, and 10 evenly spaced checkpoints in between
n = len(rows_sorted)
checkpoints = [
    ("TOP 10 (highest prob)", 0, 10),
    ("90th percentile", int(n * 0.10), int(n * 0.10) + 10),
    ("80th percentile", int(n * 0.20), int(n * 0.20) + 10),
    ("70th percentile", int(n * 0.30), int(n * 0.30) + 10),
    ("60th percentile", int(n * 0.40), int(n * 0.40) + 10),
    ("50th percentile (median)", int(n * 0.50), int(n * 0.50) + 10),
    ("40th percentile", int(n * 0.60), int(n * 0.60) + 10),
    ("30th percentile", int(n * 0.30), int(n * 0.70) + 10),
    ("20th percentile", int(n * 0.80), int(n * 0.80) + 10),
    ("10th percentile", int(n * 0.90), int(n * 0.90) + 10),
    ("BOTTOM 10 (lowest prob)", n - 10, n),
]

# =============================================================================
# PRINT SAMPLES
# =============================================================================


def truncate(text, max_len):
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


for label, start, end in checkpoints:
    print()
    print("=" * 80)
    print(f"{label}")
    print("=" * 80)

    for i, row in enumerate(rows_sorted[start:end], 1):
        prob = row["review_prob"]
        pred = "REVIEW" if row["is_review"] else "NOT REVIEW"
        text = truncate(row["document"], MAX_TEXT_LENGTH)

        print()
        print(f"--- Sample {i} | prob={prob:.4f} | pred={pred} ---")
        print(text)

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total rows: {len(rows):,}")
print(f"Predicted as review: {sum(1 for r in rows if r['is_review']):,}")
print(
    f"Probability range: {rows_sorted[-1]['review_prob']:.4f} - {rows_sorted[0]['review_prob']:.4f}"
)
