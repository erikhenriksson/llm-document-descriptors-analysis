"""
Train a descriptor-based classifier using logistic regression.
Input: binary descriptor presence vectors
Output: class predictions
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MultiLabelBinarizer


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_splits(
    data: list[dict],
    label_field: str,
    test_size: float = 0.1,
    dev_size: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Create train/dev/test splits with stratification.
    Same logic as BERT training script.
    """
    np.random.seed(seed)

    labels = np.array([item[label_field] for item in data])
    indices = np.arange(len(data))

    # First split: test
    test_indices = []
    remaining_indices = []
    for label in np.unique(labels):
        label_idx = indices[labels == label]
        np.random.shuffle(label_idx)
        n_test = int(len(label_idx) * test_size)
        test_indices.extend(label_idx[:n_test])
        remaining_indices.extend(label_idx[n_test:])

    # Second split: dev from remaining
    remaining_indices = np.array(remaining_indices)
    remaining_labels = labels[remaining_indices]

    dev_indices = []
    train_indices = []
    adjusted_dev_size = dev_size / (1 - test_size)

    for label in np.unique(remaining_labels):
        label_idx = remaining_indices[remaining_labels == label]
        np.random.shuffle(label_idx)
        n_dev = int(len(label_idx) * adjusted_dev_size)
        dev_indices.extend(label_idx[:n_dev])
        train_indices.extend(label_idx[n_dev:])

    train_data = [data[i] for i in train_indices]
    dev_data = [data[i] for i in dev_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, dev_data, test_data


def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    training_config = config["training"]

    print(f"Dataset: {config['name']}")
    print(f"Loading data from {data_config['path']}")

    data = load_jsonl(data_config["path"])
    print(f"Loaded {len(data)} examples")

    # Get config values
    descriptor_field = data_config["descriptor_field"]
    label_field = data_config["label_field"]
    label_names = data_config["label_names"]

    # Split data
    train_data, dev_data, test_data = prepare_splits(
        data=data,
        label_field=label_field,
        test_size=data_config.get("test_size", 0.1),
        dev_size=data_config.get("dev_size", 0.1),
        seed=training_config.get("seed", 42),
    )
    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")

    # Build descriptor vocabulary from training data
    print("\nBuilding descriptor vocabulary...")
    mlb = MultiLabelBinarizer()

    train_descriptors = [item[descriptor_field] for item in train_data]
    X_train = mlb.fit_transform(train_descriptors)
    y_train = np.array([item[label_field] for item in train_data])

    print(f"Vocabulary size: {len(mlb.classes_)}")
    print(f"Feature matrix shape: {X_train.shape}")

    # Transform dev and test
    dev_descriptors = [item[descriptor_field] for item in dev_data]
    X_dev = mlb.transform(dev_descriptors)
    y_dev = np.array([item[label_field] for item in dev_data])

    test_descriptors = [item[descriptor_field] for item in test_data]
    X_test = mlb.transform(test_descriptors)
    y_test = np.array([item[label_field] for item in test_data])

    # Train logistic regression
    print("\nTraining logistic regression...")
    C = training_config.get("C", 1.0)
    max_iter = training_config.get("max_iter", 1000)

    model = LogisticRegression(
        C=C,
        penalty="l2",
        solver="lbfgs",
        max_iter=max_iter,
        random_state=training_config.get("seed", 42),
    )
    model.fit(X_train, y_train)

    # Evaluate on dev
    print("\n" + "=" * 50)
    print("DEV SET EVALUATION")
    print("=" * 50)

    y_dev_pred = model.predict(X_dev)
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    dev_precision, dev_recall, dev_f1, _ = precision_recall_fscore_support(
        y_dev, y_dev_pred, average="binary" if len(label_names) == 2 else "weighted"
    )

    print(f"Accuracy:  {dev_accuracy:.4f}")
    print(f"Precision: {dev_precision:.4f}")
    print(f"Recall:    {dev_recall:.4f}")
    print(f"F1:        {dev_f1:.4f}")

    # Evaluate on test
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="binary" if len(label_names) == 2 else "weighted"
    )

    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1:        {test_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=label_names))

    # Print top coefficients
    print("\n" + "=" * 50)
    print("TOP DESCRIPTOR COEFFICIENTS")
    print("=" * 50)

    if len(label_names) == 2:
        # Binary: single coefficient vector
        coef = model.coef_[0]
        top_positive_idx = np.argsort(coef)[-15:][::-1]
        top_negative_idx = np.argsort(coef)[:15]

        print(f"\nTop 15 descriptors for '{label_names[1]}':")
        for idx in top_positive_idx:
            print(f"  {coef[idx]:+.4f}  {mlb.classes_[idx]}")

        print(f"\nTop 15 descriptors for '{label_names[0]}':")
        for idx in top_negative_idx:
            print(f"  {coef[idx]:+.4f}  {mlb.classes_[idx]}")
    else:
        # Multiclass: one coefficient vector per class
        for i, name in enumerate(label_names):
            coef = model.coef_[i]
            top_idx = np.argsort(coef)[-10:][::-1]
            print(f"\nTop 10 descriptors for '{name}':")
            for idx in top_idx:
                print(f"  {coef[idx]:+.4f}  {mlb.classes_[idx]}")

    # Save model and vectorizer
    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / "model.joblib")
    joblib.dump(mlb, output_dir / "vectorizer.joblib")

    # Save config for reference
    with open(output_dir / "training_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save vocabulary
    with open(output_dir / "vocabulary.json", "w") as f:
        json.dump(list(mlb.classes_), f, indent=2)

    # Save metrics
    metrics = {
        "dev": {
            "accuracy": dev_accuracy,
            "precision": dev_precision,
            "recall": dev_recall,
            "f1": dev_f1,
        },
        "test": {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
        },
        "vocabulary_size": len(mlb.classes_),
        "train_size": len(train_data),
        "dev_size": len(dev_data),
        "test_size": len(test_data),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {output_dir}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train descriptor-based classifier")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    main(args)
