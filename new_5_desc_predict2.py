"""
Evaluate filtering using full logistic regression probability scores
Goal: Rank documents by P(review), evaluate recall at different thresholds
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

SEED = 42
N_FINEWEB_USED_IN_TRAINING = 11100

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

print(f"Loaded model with {len(vocab)} features")

# =============================================================================
# STEP 2: LOAD FINEWEB AND EXCLUDE TRAINING DATA
# =============================================================================

print()
print("=" * 60)
print("STEP 2: Loading FineWeb (excluding training data)")
print("=" * 60)

all_fineweb = []
with open(FINEWEB_PATH, "r") as f:
    for line in f:
        all_fineweb.append(json.loads(line))

print(f"Total FineWeb documents: {len(all_fineweb)}")

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
# STEP 3: ENCODE AND SCORE ALL DOCUMENTS
# =============================================================================

print()
print("=" * 60)
print("STEP 3: Scoring all documents")
print("=" * 60)


def encode(descriptors, vocab):
    x = np.zeros(len(vocab), dtype=np.float32)
    for desc in descriptors:
        if desc in vocab:
            x[vocab[desc]] = 1.0
    return x


X = np.array([encode(row["harmonized_descriptors"], vocab) for row in eval_data])
probs = model.predict_proba(X)[:, 1]  # P(review)

# Attach scores to data
for i, row in enumerate(eval_data):
    row["descriptor_prob"] = probs[i]

print(f"Scored {len(eval_data)} documents")
print(f"Probability range: {probs.min():.4f} - {probs.max():.4f}")

# =============================================================================
# STEP 4: EVALUATE AT DIFFERENT PROBABILITY THRESHOLDS
# =============================================================================

print()
print("=" * 60)
print("STEP 4: Filtering by probability threshold")
print("=" * 60)

thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

print()
print(
    f"{'Threshold':>10} {'Recall':>10} {'Kept':>10} {'Kept%':>10} {'Enrichment':>12} {'Reviews':>15}"
)
print("-" * 75)

for thresh in thresholds:
    bucket = [row for row in eval_data if row["descriptor_prob"] >= thresh]

    n_kept = len(bucket)
    n_reviews_in_bucket = sum(1 for row in bucket if row["is_review"])

    recall = n_reviews_in_bucket / n_reviews if n_reviews > 0 else 0
    kept_pct = n_kept / n_total if n_total > 0 else 0

    bucket_review_rate = n_reviews_in_bucket / n_kept if n_kept > 0 else 0
    enrichment = bucket_review_rate / review_rate if review_rate > 0 else 0

    print(
        f"{thresh:>10.2f} {recall:>10.2%} {n_kept:>10,} {kept_pct:>10.2%} {enrichment:>12.1f}x {n_reviews_in_bucket:>6} / {n_reviews}"
    )

# =============================================================================
# STEP 5: EVALUATE BY TOP-N RANKING
# =============================================================================

print()
print("=" * 60)
print("STEP 5: Filtering by top-N ranked documents")
print("=" * 60)

# Sort by probability descending
eval_data_sorted = sorted(eval_data, key=lambda x: x["descriptor_prob"], reverse=True)

top_n_values = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]

print()
print(f"{'Top-N':>10} {'Recall':>10} {'Kept%':>10} {'Enrichment':>12} {'Reviews':>15}")
print("-" * 65)

for top_n in top_n_values:
    if top_n > n_total:
        continue

    bucket = eval_data_sorted[:top_n]
    n_reviews_in_bucket = sum(1 for row in bucket if row["is_review"])

    recall = n_reviews_in_bucket / n_reviews
    kept_pct = top_n / n_total

    bucket_review_rate = n_reviews_in_bucket / top_n
    enrichment = bucket_review_rate / review_rate

    print(
        f"{top_n:>10,} {recall:>10.2%} {kept_pct:>10.2%} {enrichment:>12.1f}x {n_reviews_in_bucket:>6} / {n_reviews}"
    )

# =============================================================================
# STEP 6: PROBABILITY DISTRIBUTION
# =============================================================================

print()
print("=" * 60)
print("STEP 6: Probability distribution")
print("=" * 60)

buckets = [
    (0.9, 1.0),
    (0.8, 0.9),
    (0.7, 0.8),
    (0.6, 0.7),
    (0.5, 0.6),
    (0.4, 0.5),
    (0.3, 0.4),
    (0.2, 0.3),
    (0.1, 0.2),
    (0.0, 0.1),
]

print()
print(f"{'Prob range':>12} {'Total':>10} {'Reviews':>10} {'% Reviews':>12}")
print("-" * 50)

for low, high in buckets:
    in_bucket = [row for row in eval_data if low <= row["descriptor_prob"] < high]
    n_in_bucket = len(in_bucket)
    n_reviews_in_bucket = sum(1 for row in in_bucket if row["is_review"])
    pct_reviews = 100 * n_reviews_in_bucket / n_in_bucket if n_in_bucket > 0 else 0

    print(
        f"{low:.1f} - {high:.1f}   {n_in_bucket:>10,} {n_reviews_in_bucket:>10} {pct_reviews:>11.1f}%"
    )

print()
print("Done!")
