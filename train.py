"""
Simple BERT fine-tuning for text classification.
Uses HuggingFace Transformers + Datasets.
Config-driven for multiple datasets.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_dataset(
    data: list[dict],
    text_field: str,
    label_field: str,
    label_map: dict | None,
    label_names: list[str],
    test_size: float,
    dev_size: float,
    seed: int,
) -> DatasetDict:
    """
    Create train/dev/test splits.
    """
    from datasets import ClassLabel, Features, Value

    # Extract text and labels
    texts = [item[text_field] for item in data]
    labels = [item[label_field] for item in data]

    # Map labels if needed
    if label_map:
        labels = [label_map[str(label)] for label in labels]

    # Ensure labels are plain ints
    labels = [int(label) for label in labels]

    # Create HuggingFace Dataset with explicit ClassLabel type
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=label_names),
        }
    )
    ds = Dataset.from_dict({"text": texts, "label": labels}, features=features)

    # First split: separate test set
    ds_split = ds.train_test_split(
        test_size=test_size, seed=seed, stratify_by_column="label"
    )

    # Second split: separate dev from train
    train_dev = ds_split["train"].train_test_split(
        test_size=dev_size / (1 - test_size), seed=seed, stratify_by_column="label"
    )

    return DatasetDict(
        {
            "train": train_dev["train"],
            "dev": train_dev["test"],
            "test": ds_split["test"],
        }
    )


def get_compute_metrics(num_labels: int, label_names: list[str]):
    """Return metrics function based on number of labels."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = accuracy_score(labels, predictions)

        # Binary vs multiclass
        average = "binary" if num_labels == 2 else "weighted"
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=average
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return compute_metrics


def main(config_path: str):
    # Load config
    config = load_config(config_path)

    # Extract config sections
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    print(f"Dataset: {config['name']}")
    print(f"Loading data from {data_config['path']}")

    data = load_jsonl(data_config["path"])
    print(f"Loaded {len(data)} examples")

    # Get label info
    label_map = data_config.get("label_map")
    label_names = data_config["label_names"]
    num_labels = len(label_names)

    # Create id2label and label2id
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    # Check label distribution
    label_field = data_config["label_field"]
    labels = [item[label_field] for item in data]
    if label_map:
        labels = [label_map[str(label)] for label in labels]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(unique, counts))}")

    # Prepare dataset
    dataset = prepare_dataset(
        data=data,
        text_field=data_config["text_field"],
        label_field=data_config["label_field"],
        label_map=label_map,
        label_names=label_names,
        test_size=data_config.get("test_size", 0.1),
        dev_size=data_config.get("dev_size", 0.1),
        seed=training_config.get("seed", 42),
    )
    print(
        f"Train: {len(dataset['train'])}, Dev: {len(dataset['dev'])}, Test: {len(dataset['test'])}"
    )

    # Load tokenizer and model
    model_name = model_config["name"]
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Tokenize
    max_length = model_config.get("max_length", 512)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    print("Tokenizing...")
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # Training arguments
    output_dir = training_config["output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=training_config.get("learning_rate", 2e-5),
        per_device_train_batch_size=training_config.get("batch_size", 16),
        per_device_eval_batch_size=training_config.get("batch_size", 16),
        num_train_epochs=training_config.get("epochs", 3),
        weight_decay=training_config.get("weight_decay", 0.01),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        seed=training_config.get("seed", 42),
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["dev"],
        compute_metrics=get_compute_metrics(num_labels, label_names),
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Evaluate on test set
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)

    test_results = trainer.evaluate(tokenized_dataset["test"])
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")

    # Detailed classification report
    predictions = trainer.predict(tokenized_dataset["test"])
    preds = np.argmax(predictions.predictions, axis=-1)

    print("\nClassification Report:")
    print(
        classification_report(
            tokenized_dataset["test"]["label"], preds, target_names=label_names
        )
    )

    # Save model
    final_path = f"{output_dir}/final"
    print(f"\nSaving model to {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    # Save config alongside model for reference
    config_save_path = f"{final_path}/training_config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT on text classification"
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
