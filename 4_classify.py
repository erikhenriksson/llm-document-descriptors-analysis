import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load data
documents = []
with open("processed/bbc_harmonized_with_labels_threshold_1.jsonl", "r") as f:
    for line in f:
        documents.append(json.loads(line))

print(f"Loaded {len(documents)} documents")

# Extract descriptors and labels, filter out docs with 0 descriptors
X_descriptors = []
y_labels = []

for doc in documents:
    # Get unique descriptors
    descriptors = set()
    for desc in doc["harmonized_descriptors"]:
        descriptors.add(desc.split(";")[0].strip())

    # Skip if no descriptors
    if len(descriptors) == 0:
        continue

    X_descriptors.append(descriptors)
    y_labels.append(doc["label_text"])

print(f"After filtering: {len(X_descriptors)} documents")

# Build vocabulary
vocabulary = set()
for descriptors in X_descriptors:
    vocabulary.update(descriptors)
vocabulary = sorted(list(vocabulary))
vocab_to_idx = {desc: idx for idx, desc in enumerate(vocabulary)}

print(f"Vocabulary size: {len(vocabulary)}")

# Multi-hot encoding
X = np.zeros((len(X_descriptors), len(vocabulary)))
for i, descriptors in enumerate(X_descriptors):
    for desc in descriptors:
        X[i, vocab_to_idx[desc]] = 1

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Train logistic regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print()

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
categories = sorted(set(y_labels))
print("Categories:", categories)
