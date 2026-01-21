"""
Apply trained descriptor classifier to new data.
Outputs predictions to JSONL.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from tqdm import tqdm


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str):
    """Save to JSONL file."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main(args):
    # Load model config
    config_path = Path(args.model_path) / "training_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        label_names = config["data"]["label_names"]
        descriptor_field = config["data"]["descriptor_field"]
    else:
        print("Warning: No training_config.yaml found, using defaults")
        label_names = ["negative", "positive"]
        descriptor_field = "harmonized_descriptors"

    # Override descriptor field if specified
    if args.descriptor_field:
        descriptor_field = args.descriptor_field

    print(f"Label names: {label_names}")
    print(f"Descriptor field: {descriptor_field}")

    # Load model and vectorizer
    print(f"Loading model from {args.model_path}")
    model = joblib.load(Path(args.model_path) / "model.joblib")
    mlb = joblib.load(Path(args.model_path) / "vectorizer.joblib")

    print(f"Vocabulary size: {len(mlb.classes_)}")

    # Load data
    print(f"Loading data from {args.data_path}")
    data = load_jsonl(args.data_path)
    print(f"Loaded {len(data)} documents")

    # Transform descriptors
    print("Transforming descriptors...")
    descriptors = [item[descriptor_field] for item in data]
    X = mlb.transform(descriptors)

    # Predict
    print("Predicting...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Add predictions to data
    results = []
    for i, item in enumerate(tqdm(data, desc="Building results")):
        result = item.copy()
        result["predicted_label"] = int(predictions[i])
        result["confidence"] = float(probabilities[i].max())
        for j, name in enumerate(label_names):
            result[f"prob_{name}"] = float(probabilities[i][j])
        results.append(result)

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(results, output_path)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with descriptor classifier"
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model directory"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to JSONL data to predict on"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path for output JSONL with predictions",
    )
    parser.add_argument(
        "--descriptor_field",
        type=str,
        default=None,
        help="Override descriptor field name",
    )

    args = parser.parse_args()
    main(args)
