"""
Simple logistic regression classifier using harmonized_descriptors
(descriptor model)
"""

import json
import pickle
import random
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# =============================================================================
# CONFIG
# =============================================================================

IMDB_PATH = "data/descriptors_imdb_harmonized_labelled.jsonl"
FINEWEB_PATH = "data/descriptors_fineweb-edu_harmonized_labelled.jsonl"
OUTPUT_MODEL_PATH = "output/descriptor_model/model.pkl"
OUTPUT_VOCAB_PATH = "output/descriptor_model/vocab.json"

# Same splits as DeBERTa model
N_IMDB_TRAIN = 1400
N_IMDB_DEV = 200
N_IMDB_TEST = 500
N_IMDB_TOTAL = N_IMDB_TRAIN + N_IMDB_DEV + N_IMDB_TEST  # 2100

N_FINEWEB_TRAIN = 1400
N_FINEWEB_DEV = 200
N_FINEWEB_TEST = 9500
N_FINEWEB_TOTAL = N_FINEWEB_TRAIN + N_FINEWEB_DEV + N_FINEWEB_TEST  # 11100

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# STEP 1: LOAD IMDB DATA
# =============================================================================

print("=" * 60)
print("STEP 1: Loading IMDB data")
print("=" * 60)

imdb_rows = []
with open(IMDB_PATH, "r") as f:
    for line in f:
        imdb_rows.append(json.loads(line))

print(f"Loaded {len(imdb_rows)} IMDB documents")

random.shuffle(imdb_rows)
imdb_rows = imdb_rows[:N_IMDB_TOTAL]
print(f"Sampled {len(imdb_rows)} IMDB documents")

# =============================================================================
# STEP 2: LOAD FINEWEB DATA
# =============================================================================

print()
print("=" * 60)
print("STEP 2: Loading FineWeb data")
print("=" * 60)

fineweb_rows = []
with open(FINEWEB_PATH, "r") as f:
    for line in f:
        fineweb_rows.append(json.loads(line))

print(f"Loaded {len(fineweb_rows)} FineWeb documents")

random.shuffle(fineweb_rows)
fineweb_rows = fineweb_rows[:N_FINEWEB_TOTAL]
print(f"Sampled {len(fineweb_rows)} FineWeb documents")

# =============================================================================
# STEP 3: CREATE TRAIN/DEV/TEST SPLITS
# =============================================================================

print()
print("=" * 60)
print("STEP 3: Creating stratified splits")
print("=" * 60)

# Split IMDB
imdb_train = imdb_rows[:N_IMDB_TRAIN]
imdb_dev = imdb_rows[N_IMDB_TRAIN : N_IMDB_TRAIN + N_IMDB_DEV]
imdb_test = imdb_rows[N_IMDB_TRAIN + N_IMDB_DEV :]

# Split FineWeb
fineweb_train = fineweb_rows[:N_FINEWEB_TRAIN]
fineweb_dev = fineweb_rows[N_FINEWEB_TRAIN : N_FINEWEB_TRAIN + N_FINEWEB_DEV]
fineweb_test = fineweb_rows[N_FINEWEB_TRAIN + N_FINEWEB_DEV :]

# Combine: label 1 = review (IMDB), label 0 = not review (FineWeb)
train_data = [(r["harmonized_descriptors"], 1) for r in imdb_train] + [
    (r["harmonized_descriptors"], 0) for r in fineweb_train
]
dev_data = [(r["harmonized_descriptors"], 1) for r in imdb_dev] + [
    (r["harmonized_descriptors"], 0) for r in fineweb_dev
]
test_data = [(r["harmonized_descriptors"], 1) for r in imdb_test] + [
    (r["harmonized_descriptors"], 0) for r in fineweb_test
]

random.shuffle(train_data)
random.shuffle(dev_data)
random.shuffle(test_data)

print(
    f"Train: {len(train_data):,} ({N_IMDB_TRAIN} reviews, {N_FINEWEB_TRAIN} not) - 50/50"
)
print(f"Dev:   {len(dev_data):,} ({N_IMDB_DEV} reviews, {N_FINEWEB_DEV} not) - 50/50")
print(f"Test:  {len(test_data):,} ({N_IMDB_TEST} reviews, {N_FINEWEB_TEST} not) - 5/95")

# =============================================================================
# STEP 4: BUILD VOCABULARY (from train only)
# =============================================================================

print()
print("=" * 60)
print("STEP 4: Building vocabulary")
print("=" * 60)

descriptor_counts = Counter()
for descriptors, label in train_data:
    descriptor_counts.update(descriptors)

vocab = {desc: i for i, desc in enumerate(sorted(descriptor_counts.keys()))}

print(f"Vocabulary size: {len(vocab)}")
print(f"Top 10 descriptors: {descriptor_counts.most_common(10)}")

# =============================================================================
# STEP 5: ENCODE FEATURES (multi-hot)
# =============================================================================

print()
print("=" * 60)
print("STEP 5: Encoding features")
print("=" * 60)


def encode(data, vocab):
    X = np.zeros((len(data), len(vocab)), dtype=np.float32)
    y = np.zeros(len(data), dtype=np.int32)

    for i, (descriptors, label) in enumerate(data):
        for desc in descriptors:
            if desc in vocab:
                X[i, vocab[desc]] = 1.0
        y[i] = label

    return X, y


X_train, y_train = encode(train_data, vocab)
X_dev, y_dev = encode(dev_data, vocab)
X_test, y_test = encode(test_data, vocab)

print(f"X_train shape: {X_train.shape}")
print(f"X_dev shape: {X_dev.shape}")
print(f"X_test shape: {X_test.shape}")

# =============================================================================
# STEP 6: TRAIN LOGISTIC REGRESSION
# =============================================================================

print()
print("=" * 60)
print("STEP 6: Training logistic regression")
print("=" * 60)

model = LogisticRegression(
    C=1.0,
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    random_state=SEED,
)

model.fit(X_train, y_train)

print("Training complete")

# =============================================================================
# STEP 7: EVALUATE ON DEV
# =============================================================================

print()
print("=" * 60)
print("STEP 7: Evaluating on dev set (50/50)")
print("=" * 60)

y_dev_pred = model.predict(X_dev)

print()
print(
    classification_report(
        y_dev,
        y_dev_pred,
        target_names=["not_review", "review"],
    )
)

# =============================================================================
# STEP 8: EVALUATE ON TEST
# =============================================================================

print()
print("=" * 60)
print("STEP 8: Evaluating on test set (5% reviews, 95% not)")
print("=" * 60)

y_test_pred = model.predict(X_test)

print()
print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["not_review", "review"],
    )
)

# =============================================================================
# STEP 9: SAVE MODEL
# =============================================================================

print()
print("=" * 60)
print("STEP 9: Saving model")
print("=" * 60)

import os

os.makedirs("output/descriptor_model", exist_ok=True)

with open(OUTPUT_MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(OUTPUT_VOCAB_PATH, "w") as f:
    json.dump(vocab, f)

print(f"Model saved to {OUTPUT_MODEL_PATH}")
print(f"Vocab saved to {OUTPUT_VOCAB_PATH}")
print()
print("Done!")
