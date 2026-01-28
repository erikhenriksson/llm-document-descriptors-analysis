"""
Evaluate descriptor-based filtering for large-scale data reduction
Goal: Find top-K descriptors that capture most reviews while keeping a small % of data
"""

import json
import pickle
import random

import numpy as np

# =============================================================================
# CONFIG
# =============================================================================

MODEL_PATH = "output/descriptor_model/model.pkl"
VOCAB_PATH = "output/descriptor_model/vocab.json"
FINEWEB_PATH = "data/descriptors_fineweb-edu_harmonized_labelled_with_review_pred.jsonl"

# Same seed and sample size as training to identify and exclude training data
SEED = 42
N_FINEWEB_USED_IN_TRAINING = 11100

# K values to test
TOP_K_VALUES = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]

random.seed(SEED)

# =============================================================================
# STEP 1: LOAD MODEL AND VOCAB
# =============================================================================

print("=" * 60)
print("STEP 1: Loading model and vocab")
print("=" * 60)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

idx_to_desc = {i: desc for desc, i in vocab.items()}
coefficients = model.coef_[0]
sorted_indices = np.argsort(coefficients)[::-1]

print(f"Loaded model with {len(vocab)} features")

# =============================================================================
# STEP 2: SHOW TOP DESCRIPTORS
# =============================================================================

print()
print("=" * 60)
print("STEP 2: Top 30 'review' descriptors")
print("=" * 60)

print()
for i in range(30):
    idx = sorted_indices[i]
    desc = idx_to_desc[idx]
    coef = coefficients[idx]
    print(f"  {i + 1:2}. {desc:<35} coef={coef:.4f}")

# =============================================================================
# STEP 3: LOAD FINEWEB AND EXCLUDE TRAINING DATA
# =============================================================================

print()
print("=" * 60)
print("STEP 3: Loading FineWeb (excluding training data)")
print("=" * 60)

all_fineweb = []
with open(FINEWEB_PATH, "r") as f:
    for line in f:
        all_fineweb.append(json.loads(line))

print(f"Total FineWeb documents: {len(all_fineweb)}")

# Reproduce the same shuffle to identify training samples
all_fineweb_shuffled = all_fineweb.copy()
random.seed(SEED)
random.shuffle(all_fineweb_shuffled)

training_docs = set(
    row["document"] for row in all_fineweb_shuffled[:N_FINEWEB_USED_IN_TRAINING]
)
eval_data = [row for row in all_fineweb if row["document"] not in training_docs]

print(f"Excluded {len(all_fineweb) - len(eval_data)} training documents")
print(f"Evaluation set: {len(eval_data)} documents")

n_total = len(eval_data)
n_reviews = sum(1 for row in eval_data if row["is_review"])
review_rate = n_reviews / n_total

print(f"Reviews in eval set: {n_reviews} ({100 * review_rate:.2f}%)")

# =============================================================================
# STEP 4: EVALUATE FILTERING AT DIFFERENT K
# =============================================================================

print()
print("=" * 60)
print("STEP 4: Evaluating filtering thresholds")
print("=" * 60)

print()
print(
    f"{'K':>5} {'Recall':>10} {'Kept':>10} {'Kept%':>10} {'Enrichment':>12} {'Reviews in bucket':>18}"
)
print("-" * 75)

results = []

for k in TOP_K_VALUES:
    # Top-K descriptors
    top_descriptors = set(idx_to_desc[sorted_indices[i]] for i in range(k))

    # Filter: keep docs with at least 1 top-K descriptor
    bucket = []
    for row in eval_data:
        doc_descriptors = set(row["harmonized_descriptors"])
        if doc_descriptors & top_descriptors:
            bucket.append(row)

    n_kept = len(bucket)
    n_reviews_in_bucket = sum(1 for row in bucket if row["is_review"])

    recall = n_reviews_in_bucket / n_reviews if n_reviews > 0 else 0
    kept_pct = n_kept / n_total

    # Enrichment: how much richer is the bucket in reviews vs original?
    bucket_review_rate = n_reviews_in_bucket / n_kept if n_kept > 0 else 0
    enrichment = bucket_review_rate / review_rate if review_rate > 0 else 0

    print(
        f"{k:>5} {recall:>10.2%} {n_kept:>10,} {kept_pct:>10.2%} {enrichment:>12.1f}x {n_reviews_in_bucket:>10} / {n_reviews}"
    )

    results.append(
        {
            "k": k,
            "recall": recall,
            "n_kept": n_kept,
            "kept_pct": kept_pct,
            "enrichment": enrichment,
            "n_reviews_in_bucket": n_reviews_in_bucket,
        }
    )

# =============================================================================
# STEP 5: MATCH COUNT DISTRIBUTION (for best K)
# =============================================================================

print()
print("=" * 60)
print("STEP 5: Match count distribution (K=20)")
print("=" * 60)

k = 500
top_descriptors = set(idx_to_desc[sorted_indices[i]] for i in range(k))

match_counts_reviews = []
match_counts_not_reviews = []

for row in eval_data:
    doc_descriptors = set(row["harmonized_descriptors"])
    matches = len(doc_descriptors & top_descriptors)

    if row["is_review"]:
        match_counts_reviews.append(matches)
    else:
        match_counts_not_reviews.append(matches)

print()
print("Match count distribution for REVIEWS:")
print("-" * 40)
for i in range(max(match_counts_reviews) + 1):
    count = match_counts_reviews.count(i)
    pct = 100 * count / len(match_counts_reviews)
    bar = "#" * int(pct / 2)
    print(f"  {i:>2} matches: {count:>5} ({pct:>5.1f}%) {bar}")

print()
print("Match count distribution for NOT REVIEWS:")
print("-" * 40)
for i in range(min(10, max(match_counts_not_reviews) + 1)):
    count = match_counts_not_reviews.count(i)
    pct = 100 * count / len(match_counts_not_reviews)
    bar = "#" * int(pct / 2)
    print(f"  {i:>2} matches: {count:>5} ({pct:>5.1f}%) {bar}")

# =============================================================================
# STEP 6: RECOMMENDATIONS
# =============================================================================

print()
print("=" * 60)
print("STEP 6: Recommendations")
print("=" * 60)

print()
print("Choose K based on your needs:")
print()
for r in results:
    if r["recall"] >= 0.90:
        print(
            f"  K={r['k']:>3}: {r['recall']:.0%} recall, keeps {r['kept_pct']:.1%} of data ({r['enrichment']:.1f}x enrichment)"
        )

print()
print("Done!")
