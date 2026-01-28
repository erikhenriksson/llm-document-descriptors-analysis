"""
Evaluate filtering using descriptors ranked by lift
Lift = P(descriptor | review) / P(descriptor | not review)
"""

import json
import random
from collections import Counter

# =============================================================================
# CONFIG
# =============================================================================

IMDB_PATH = "data/descriptors_imdb_harmonized_labelled.jsonl"
FINEWEB_PATH = "data/descriptors_fineweb-edu_harmonized_labelled_with_review_pred.jsonl"

SEED = 42
N_FINEWEB_USED_IN_TRAINING = 11100

# Use same IMDB sample as training to compute lift
N_IMDB_TOTAL = 2100

TOP_K_VALUES = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]

random.seed(SEED)

# =============================================================================
# STEP 1: LOAD IMDB (reviews) FOR LIFT CALCULATION
# =============================================================================

print("=" * 60)
print("STEP 1: Loading IMDB data (reviews)")
print("=" * 60)

imdb_rows = []
with open(IMDB_PATH, "r") as f:
    for line in f:
        imdb_rows.append(json.loads(line))

random.shuffle(imdb_rows)
imdb_rows = imdb_rows[:N_IMDB_TOTAL]

print(f"Loaded {len(imdb_rows)} IMDB documents")

# =============================================================================
# STEP 2: LOAD FINEWEB AND SPLIT TRAIN/EVAL
# =============================================================================

print()
print("=" * 60)
print("STEP 2: Loading FineWeb data")
print("=" * 60)

all_fineweb = []
with open(FINEWEB_PATH, "r") as f:
    for line in f:
        all_fineweb.append(json.loads(line))

print(f"Total FineWeb documents: {len(all_fineweb)}")

# Reproduce shuffle to identify training samples
all_fineweb_shuffled = all_fineweb.copy()
random.seed(SEED)
random.shuffle(all_fineweb_shuffled)

# Training set (for computing lift on non-reviews)
fineweb_train = all_fineweb_shuffled[:N_FINEWEB_USED_IN_TRAINING]

# Eval set (exclude training)
training_docs = set(row["document"] for row in fineweb_train)
eval_data = [row for row in all_fineweb if row["document"] not in training_docs]

print(f"FineWeb for lift calculation: {len(fineweb_train)}")
print(f"FineWeb for evaluation: {len(eval_data)}")

n_total = len(eval_data)
n_reviews = sum(1 for row in eval_data if row["is_review"])
review_rate = n_reviews / n_total

print(f"Reviews in eval set: {n_reviews} ({100 * review_rate:.2f}%)")

# =============================================================================
# STEP 3: COMPUTE LIFT FOR EACH DESCRIPTOR
# =============================================================================

print()
print("=" * 60)
print("STEP 3: Computing lift for each descriptor")
print("=" * 60)

# Count descriptor occurrences in reviews (IMDB)
review_desc_counts = Counter()
for row in imdb_rows:
    review_desc_counts.update(row["harmonized_descriptors"])

# Count descriptor occurrences in non-reviews (FineWeb train)
non_review_desc_counts = Counter()
for row in fineweb_train:
    non_review_desc_counts.update(row["harmonized_descriptors"])

# Get all descriptors
all_descriptors = set(review_desc_counts.keys()) | set(non_review_desc_counts.keys())

# Compute lift
n_reviews_train = len(imdb_rows)
n_non_reviews_train = len(fineweb_train)

lifts = {}
for desc in all_descriptors:
    p_desc_given_review = review_desc_counts[desc] / n_reviews_train
    p_desc_given_not_review = non_review_desc_counts[desc] / n_non_reviews_train

    # Avoid division by zero - add small smoothing
    if p_desc_given_not_review == 0:
        lift = p_desc_given_review / (1 / n_non_reviews_train)  # assume 1 occurrence
    else:
        lift = p_desc_given_review / p_desc_given_not_review

    lifts[desc] = {
        "lift": lift,
        "p_review": p_desc_given_review,
        "p_not_review": p_desc_given_not_review,
        "count_review": review_desc_counts[desc],
        "count_not_review": non_review_desc_counts[desc],
    }

# Sort by lift descending
sorted_descriptors = sorted(lifts.keys(), key=lambda x: lifts[x]["lift"], reverse=True)

print(f"Total descriptors: {len(all_descriptors)}")

# =============================================================================
# STEP 4: SHOW TOP DESCRIPTORS BY LIFT
# =============================================================================

print()
print("=" * 60)
print("STEP 4: Top 30 descriptors by lift")
print("=" * 60)

print()
print(
    f"{'Rank':>4} {'Descriptor':<35} {'Lift':>8} {'P(rev)':>8} {'P(not)':>8} {'#rev':>6} {'#not':>6}"
)
print("-" * 90)

for i in range(30):
    desc = sorted_descriptors[i]
    info = lifts[desc]
    print(
        f"{i + 1:>4} {desc:<35} {info['lift']:>8.1f} {info['p_review']:>8.2%} {info['p_not_review']:>8.2%} {info['count_review']:>6} {info['count_not_review']:>6}"
    )

# =============================================================================
# STEP 5: EVALUATE FILTERING AT DIFFERENT K
# =============================================================================

print()
print("=" * 60)
print("STEP 5: Evaluating filtering with top-K descriptors by lift")
print("=" * 60)

print()
print(
    f"{'K':>5} {'Recall':>10} {'Kept':>10} {'Kept%':>10} {'Enrichment':>12} {'Reviews':>15}"
)
print("-" * 75)

results = []

for k in TOP_K_VALUES:
    top_descriptors = set(sorted_descriptors[:k])

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

    bucket_review_rate = n_reviews_in_bucket / n_kept if n_kept > 0 else 0
    enrichment = bucket_review_rate / review_rate if review_rate > 0 else 0

    print(
        f"{k:>5} {recall:>10.2%} {n_kept:>10,} {kept_pct:>10.2%} {enrichment:>12.1f}x {n_reviews_in_bucket:>6} / {n_reviews}"
    )

    results.append(
        {
            "k": k,
            "recall": recall,
            "n_kept": n_kept,
            "kept_pct": kept_pct,
            "enrichment": enrichment,
        }
    )

# =============================================================================
# STEP 6: MATCH COUNT DISTRIBUTION
# =============================================================================

print()
print("=" * 60)
print("STEP 6: Match count distribution (K=50 by lift)")
print("=" * 60)

k = 50
top_descriptors = set(sorted_descriptors[:k])

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
max_matches = max(match_counts_reviews) if match_counts_reviews else 0
for i in range(max_matches + 1):
    count = match_counts_reviews.count(i)
    pct = 100 * count / len(match_counts_reviews) if match_counts_reviews else 0
    bar = "#" * int(pct / 2)
    print(f"  {i:>2} matches: {count:>5} ({pct:>5.1f}%) {bar}")

print()
print("Match count distribution for NOT REVIEWS:")
print("-" * 40)
max_show = min(10, max(match_counts_not_reviews) + 1) if match_counts_not_reviews else 0
for i in range(max_show):
    count = match_counts_not_reviews.count(i)
    pct = 100 * count / len(match_counts_not_reviews) if match_counts_not_reviews else 0
    bar = "#" * int(pct / 2)
    print(f"  {i:>2} matches: {count:>5} ({pct:>5.1f}%) {bar}")

# =============================================================================
# STEP 7: RECOMMENDATIONS
# =============================================================================

print()
print("=" * 60)
print("STEP 7: Recommendations")
print("=" * 60)

print()
print("Configurations with >=80% recall:")
print()
for r in results:
    if r["recall"] >= 0.80:
        print(
            f"  K={r['k']:>3}: {r['recall']:.0%} recall, keeps {r['kept_pct']:.1%} of data ({r['enrichment']:.1f}x enrichment)"
        )

print()
print("Done!")
