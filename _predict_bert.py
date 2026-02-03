"""
Apply trained BERT classifier to predict on remaining FineWeb documents.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm
from transformers import pipeline


def load_jsonl(path):
    """Load JSONL file into list of dictionaries."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Predict with trained BERT classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    predictions_dir = Path(config["output"]["predictions_dir"])

    # Load documents to predict
    print("Loading documents to predict...")
    data = load_jsonl(predictions_dir / "fineweb_to_predict.jsonl")
    documents = [item["document"] for item in data]

    print(f"Loaded {len(documents)} documents")

    # Load model as pipeline
    model_dir = Path(config["output"]["model_dir"]) / "best_model"
    print(f"Loading model from {model_dir}")

    classifier = pipeline(
        "text-classification",
        model=str(model_dir),
        tokenizer=str(model_dir),
        device=0 if __import__("torch").cuda.is_available() else -1,
        truncation=True,
        max_length=config["training"]["max_length"],
    )

    # Generate predictions in batches
    print("\nGenerating predictions...")
    batch_size = config["training"]["batch_size"]

    all_predictions = []
    all_probabilities = []

    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i : i + batch_size]
        results = classifier(batch)

        for result in results:
            # Pipeline returns label as "LABEL_0" or "LABEL_1"
            pred_label = int(result["label"].split("_")[1])

            # Get probability for positive class (label 1)
            if pred_label == 1:
                prob = result["score"]
            else:
                prob = 1 - result["score"]

            all_predictions.append(pred_label)
            all_probabilities.append(prob)

    # Save results
    output_file = predictions_dir / "predictions.jsonl"

    with open(output_file, "w") as f:
        for item, pred, prob in zip(data, all_predictions, all_probabilities):
            result = {
                "document": item["document"],
                "harmonized_descriptors": item.get("harmonized_descriptors", []),
                "predicted_label": pred,
                "probability": float(prob),
            }
            f.write(json.dumps(result) + "\n")

    print(f"\nSaved predictions to {output_file}")

    # Print summary statistics
    n_positive = sum(all_predictions)
    n_total = len(all_predictions)
    print(f"\nPrediction Summary:")
    print(f"Total documents: {n_total}")
    print(f"Predicted positive: {n_positive} ({100 * n_positive / n_total:.2f}%)")
    print(
        f"Predicted negative: {n_total - n_positive} ({100 * (n_total - n_positive) / n_total:.2f}%)"
    )
    print(f"Average probability (positive class): {np.mean(all_probabilities):.4f}")


if __name__ == "__main__":
    main()
