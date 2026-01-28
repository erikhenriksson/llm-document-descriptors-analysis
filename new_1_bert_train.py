"""
Simple DeBERTa classifier: movie review vs. not review
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# =============================================================================
# CONFIG
# =============================================================================

IMDB_PATH = "data/descriptors_imdb_harmonized_labelled.jsonl"
FINEWEB_CACHE_PATH = "data/fineweb_sample_11100.jsonl"
MODEL_NAME = "microsoft/deberta-v3-small"
OUTPUT_DIR = "output/review_classifier"

# Train/Dev: 50/50 balanced (for learning)
# Test: 5/95 imbalanced (realistic evaluation)
N_IMDB_TRAIN = 1400
N_IMDB_DEV = 200
N_IMDB_TEST = 500
N_IMDB_TOTAL = N_IMDB_TRAIN + N_IMDB_DEV + N_IMDB_TEST  # 2100

N_FINEWEB_TRAIN = 1400
N_FINEWEB_DEV = 200
N_FINEWEB_TEST = 9500
N_FINEWEB_TOTAL = N_FINEWEB_TRAIN + N_FINEWEB_DEV + N_FINEWEB_TEST  # 11100

FINEWEB_START_INDEX = 1_000_001

MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
PATIENCE = 5

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# STEP 1: LOAD IMDB DATA
# =============================================================================

print("=" * 60)
print("STEP 1: Loading IMDB data")
print("=" * 60)

imdb_docs = []
with open(IMDB_PATH, "r") as f:
    for line in f:
        row = json.loads(line)
        imdb_docs.append(row["document"])

print(f"Loaded {len(imdb_docs)} IMDB documents")

# Sample
random.shuffle(imdb_docs)
imdb_docs = imdb_docs[:N_IMDB_TOTAL]
print(f"Sampled {len(imdb_docs)} IMDB documents")

# =============================================================================
# STEP 2: LOAD OR CACHE FINEWEB DATA
# =============================================================================

print()
print("=" * 60)
print("STEP 2: Loading FineWeb data")
print("=" * 60)

fineweb_docs = []

if os.path.exists(FINEWEB_CACHE_PATH):
    print(f"Loading from cache: {FINEWEB_CACHE_PATH}")
    with open(FINEWEB_CACHE_PATH, "r") as f:
        for line in f:
            row = json.loads(line)
            fineweb_docs.append(row["text"])
    print(f"Loaded {len(fineweb_docs)} FineWeb documents from cache")

else:
    print(
        f"Downloading FineWeb (streaming, skipping first {FINEWEB_START_INDEX} records)..."
    )
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # Skip to start index and collect samples
    for i, example in enumerate(dataset):
        if i < FINEWEB_START_INDEX:
            if i % 100_000 == 0:
                print(f"  Skipping... {i:,} / {FINEWEB_START_INDEX:,}")
            continue

        fineweb_docs.append(example["text"])

        if len(fineweb_docs) >= N_FINEWEB_TOTAL:
            break

        if len(fineweb_docs) % 500 == 0:
            print(f"  Collected {len(fineweb_docs)} / {N_FINEWEB_TOTAL}")

    # Cache to disk
    print(f"Caching to {FINEWEB_CACHE_PATH}")
    Path(FINEWEB_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(FINEWEB_CACHE_PATH, "w") as f:
        for text in fineweb_docs:
            f.write(json.dumps({"text": text}) + "\n")

    print(f"Cached {len(fineweb_docs)} FineWeb documents")

# =============================================================================
# STEP 3: CREATE TRAIN/DEV/TEST SPLITS (stratified)
# =============================================================================

print()
print("=" * 60)
print("STEP 3: Creating stratified splits")
print("=" * 60)

# Split IMDB
imdb_train = imdb_docs[:N_IMDB_TRAIN]
imdb_dev = imdb_docs[N_IMDB_TRAIN : N_IMDB_TRAIN + N_IMDB_DEV]
imdb_test = imdb_docs[N_IMDB_TRAIN + N_IMDB_DEV :]

# Split FineWeb
fineweb_train = fineweb_docs[:N_FINEWEB_TRAIN]
fineweb_dev = fineweb_docs[N_FINEWEB_TRAIN : N_FINEWEB_TRAIN + N_FINEWEB_DEV]
fineweb_test = fineweb_docs[N_FINEWEB_TRAIN + N_FINEWEB_DEV :]

# Combine and label: 1 = review, 0 = not review
train_data = [{"text": t, "label": 1} for t in imdb_train] + [
    {"text": t, "label": 0} for t in fineweb_train
]
dev_data = [{"text": t, "label": 1} for t in imdb_dev] + [
    {"text": t, "label": 0} for t in fineweb_dev
]
test_data = [{"text": t, "label": 1} for t in imdb_test] + [
    {"text": t, "label": 0} for t in fineweb_test
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
# STEP 4: TOKENIZE
# =============================================================================

print()
print("=" * 60)
print("STEP 4: Tokenizing")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


# Convert to HF datasets
from datasets import Dataset

train_dataset = Dataset.from_list(train_data).map(tokenize, batched=True)
dev_dataset = Dataset.from_list(dev_data).map(tokenize, batched=True)
test_dataset = Dataset.from_list(test_data).map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

print("Tokenization complete")

# =============================================================================
# STEP 5: LOAD MODEL
# =============================================================================

print()
print("=" * 60)
print("STEP 5: Loading model")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

print(f"Loaded {MODEL_NAME}")

# =============================================================================
# STEP 6: TRAIN
# =============================================================================

print()
print("=" * 60)
print("STEP 6: Training")
print("=" * 60)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    logging_steps=50,
    report_to="none",
    seed=SEED,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
)

trainer.train()

print("Training complete")

# =============================================================================
# STEP 7: EVALUATE ON TEST SET
# =============================================================================

print()
print("=" * 60)
print("STEP 7: Evaluating on test set (5% reviews, 95% not)")
print("=" * 60)

# Get predictions
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

# Classification report
print()
print(
    classification_report(
        labels,
        preds,
        target_names=["not_review", "review"],
    )
)

# =============================================================================
# STEP 8: SAVE MODEL
# =============================================================================

print()
print("=" * 60)
print("STEP 8: Saving model")
print("=" * 60)

final_model_path = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"Model saved to {final_model_path}")
print()
print("Done!")
