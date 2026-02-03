"""
Simple BERT classifier for document classification.
Trains on benchmark positive class + random FineWeb negative samples.
"""

import json
import random
from pathlib import Path

import numpy as np
import yaml
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path):
    """Load JSONL file into list of dictionaries."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def prepare_data(config):
    """Prepare training data from benchmark and FineWeb."""

    print("Loading benchmark data...")
    benchmark_data = load_jsonl(config["benchmark"]["path"])

    print("Loading FineWeb data...")
    fineweb_data = load_jsonl(config["fineweb"]["path"])

    # Get positive examples from benchmark
    positive_labels = config["benchmark"]["positive_labels"]
    positive_examples = [
        item for item in benchmark_data if item["label"] in positive_labels
    ]

    # Sample from positive examples
    samples_per_class = config["training"]["samples_per_class"]
    n_positive = min(samples_per_class, len(positive_examples))

    if n_positive < samples_per_class:
        print(
            f"Warning: Only {n_positive} positive examples available (requested {samples_per_class})"
        )

    positive_sample = random.sample(positive_examples, n_positive)

    # Sample negative examples from FineWeb
    n_negative = n_positive  # Keep balanced
    negative_sample = random.sample(fineweb_data, n_negative)

    # Extract documents and create labels
    positive_docs = [item["document"] for item in positive_sample]
    negative_docs = [item["document"] for item in negative_sample]

    documents = positive_docs + negative_docs
    labels = [1] * len(positive_docs) + [0] * len(negative_docs)

    print(
        f"Prepared {len(positive_docs)} positive and {len(negative_docs)} negative examples"
    )

    # Split into train and validation
    test_split = config["training"]["test_split"]
    train_docs, val_docs, train_labels, val_labels = train_test_split(
        documents, labels, test_size=test_split, random_state=42, stratify=labels
    )

    # Get remaining FineWeb for prediction
    used_indices = set(fineweb_data.index(item) for item in negative_sample)
    remaining_fineweb = [
        item for i, item in enumerate(fineweb_data) if i not in used_indices
    ]

    return train_docs, train_labels, val_docs, val_labels, remaining_fineweb


def compute_metrics(eval_pred):
    """Compute accuracy for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set random seed
    random.seed(42)

    # Prepare data
    train_docs, train_labels, val_docs, val_labels, remaining_fineweb = prepare_data(
        config
    )

    # Create HuggingFace datasets
    train_dataset = Dataset.from_dict({"text": train_docs, "label": train_labels})

    val_dataset = Dataset.from_dict({"text": val_docs, "label": val_labels})

    # Initialize tokenizer and model
    print(f"Loading model: {config['training']['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["training"]["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["training"]["model_name"], num_labels=2
    )

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["training"]["max_length"],
            padding="max_length",
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Setup training arguments
    model_dir = Path(config["output"]["model_dir"])

    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=str(Path(config["output"]["logs_dir"])),
        logging_steps=100,
        seed=42,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save best model
    best_model_path = model_dir / "best_model"
    trainer.save_model(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))

    print(f"\nTraining complete! Best model saved to {best_model_path}")

    # Evaluate
    metrics = trainer.evaluate()
    print(f"Final validation metrics: {metrics}")

    # Save remaining FineWeb data for prediction
    predictions_dir = Path(config["output"]["predictions_dir"])
    predictions_dir.mkdir(parents=True, exist_ok=True)

    with open(predictions_dir / "fineweb_to_predict.jsonl", "w") as f:
        for item in remaining_fineweb:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(remaining_fineweb)} FineWeb documents for prediction")


if __name__ == "__main__":
    main()
