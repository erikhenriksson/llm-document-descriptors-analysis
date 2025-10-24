import json
import os

# Set cache to local directory
os.environ["TRANSFORMERS_CACHE"] = "./model_cache"
os.environ["HF_HOME"] = "./model_cache"

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Load data
documents = []
with open("processed/bbc_harmonized_with_labels_threshold_1.jsonl", "r") as f:
    for line in f:
        doc = json.loads(line)
        # Skip if no text
        if "document" in doc and doc["document"].strip():
            documents.append(doc)

print(f"Loaded {len(documents)} documents with text")

# Extract text and labels
texts = [doc["document"] for doc in documents]
labels = [doc["label_text"] for doc in documents]

# Create label mapping
unique_labels = sorted(set(labels))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
label_ids = [label2id[label] for label in labels]

print(f"Categories: {unique_labels}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, label_ids, test_size=0.2, random_state=42, stratify=label_ids
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Load ModernBERT
model_name = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
)


# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=8192,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx]),
        }


train_dataset = TextDataset(X_train, y_train, tokenizer)
test_dataset = TextDataset(X_test, y_test, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    report_to="none",
)


# Metric function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}


# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("\nTraining ModernBERT...")
trainer.train()

# Evaluate
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(axis=1)

print("\n" + "=" * 60)
print("MODERNBERT RESULTS")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=unique_labels))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Categories:", unique_labels)
