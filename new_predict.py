"""
Predict review vs. not-review on FineWeb data using trained model
"""

import json

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =============================================================================
# CONFIG
# =============================================================================

MODEL_PATH = "output/review_classifier/final_model"
INPUT_PATH = "data/descriptors_fineweb-edu_harmonized_labelled.jsonl"
OUTPUT_PATH = "data/descriptors_fineweb-edu_harmonized_labelled_with_review_pred.jsonl"

BATCH_SIZE = 32
MAX_LENGTH = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# STEP 1: LOAD MODEL
# =============================================================================

print("=" * 60)
print("STEP 1: Loading model")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

print(f"Loaded model from {MODEL_PATH}")
print(f"Using device: {DEVICE}")

# =============================================================================
# STEP 2: LOAD DATA
# =============================================================================

print()
print("=" * 60)
print("STEP 2: Loading data")
print("=" * 60)

rows = []
with open(INPUT_PATH, "r") as f:
    for line in f:
        rows.append(json.loads(line))

print(f"Loaded {len(rows):,} rows")

# =============================================================================
# STEP 3: PREDICT IN BATCHES
# =============================================================================

print()
print("=" * 60)
print("STEP 3: Predicting")
print("=" * 60)

predictions = []

for i in tqdm(range(0, len(rows), BATCH_SIZE)):
    batch_rows = rows[i : i + BATCH_SIZE]
    texts = [r["document"] for r in batch_rows]

    inputs = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

    for j, row in enumerate(batch_rows):
        row["is_review"] = bool(preds[j].item())
        row["review_prob"] = round(probs[j, 1].item(), 4)
        predictions.append(row)

# =============================================================================
# STEP 4: SAVE
# =============================================================================

print()
print("=" * 60)
print("STEP 4: Saving results")
print("=" * 60)

with open(OUTPUT_PATH, "w") as f:
    for row in predictions:
        f.write(json.dumps(row) + "\n")

n_reviews = sum(1 for r in predictions if r["is_review"])
print(f"Saved {len(predictions):,} rows to {OUTPUT_PATH}")
print(f"Predicted reviews: {n_reviews:,} ({100 * n_reviews / len(predictions):.2f}%)")
print()
print("Done!")
