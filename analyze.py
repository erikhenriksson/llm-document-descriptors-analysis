"""
Apply trained classifier to new data and analyze confidence distribution.
Use this to:
1. Pseudo-label FineWeb with a trained model
2. Investigate confidence distributions
3. Determine thresholds for filtering
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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


def predict_batch(
    texts: list[str],
    model,
    tokenizer,
    device: str,
    max_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict on a batch of texts.
    Returns (predicted_labels, probabilities).
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

    return preds, probs


def run_inference(
    data: list[dict],
    model,
    tokenizer,
    device: str,
    text_field: str,
    batch_size: int = 32,
    max_length: int = 512,
) -> list[dict]:
    """
    Run inference on all data, adding predictions and probabilities.
    """
    results = []

    for i in tqdm(range(0, len(data), batch_size), desc="Predicting"):
        batch = data[i : i + batch_size]
        texts = [item[text_field] for item in batch]

        preds, probs = predict_batch(texts, model, tokenizer, device, max_length)

        for j, item in enumerate(batch):
            result = item.copy()
            result["predicted_label"] = int(preds[j])
            result["confidence"] = float(probs[j].max())
            result["prob_class_0"] = float(probs[j][0])
            result["prob_class_1"] = float(probs[j][1])
            results.append(result)

    return results


def analyze_confidence_distribution(
    results: list[dict],
    label_names: list[str],
    output_dir: str,
):
    """
    Analyze and visualize confidence distributions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    confidences = np.array([r["confidence"] for r in results])
    predictions = np.array([r["predicted_label"] for r in results])

    # Basic stats
    print("\n" + "=" * 60)
    print("CONFIDENCE DISTRIBUTION ANALYSIS")
    print("=" * 60)

    print(f"\nTotal documents: {len(results)}")
    print(f"\nConfidence statistics:")
    print(f"  Mean:   {confidences.mean():.4f}")
    print(f"  Std:    {confidences.std():.4f}")
    print(f"  Min:    {confidences.min():.4f}")
    print(f"  Max:    {confidences.max():.4f}")
    print(f"  Median: {np.median(confidences):.4f}")

    # Prediction distribution
    print(f"\nPrediction distribution:")
    for i, name in enumerate(label_names):
        count = (predictions == i).sum()
        pct = count / len(predictions) * 100
        print(f"  {name}: {count} ({pct:.1f}%)")

    # Confidence thresholds analysis
    print(f"\nThreshold analysis (documents retained):")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    threshold_stats = []
    for thresh in thresholds:
        above = confidences >= thresh
        n_above = above.sum()
        pct_above = n_above / len(confidences) * 100

        # Class distribution among high-confidence predictions
        high_conf_preds = predictions[above]
        class_dist = {
            name: (high_conf_preds == i).sum() for i, name in enumerate(label_names)
        }

        print(f"  >= {thresh:.2f}: {n_above:>6} ({pct_above:>5.1f}%) | {class_dist}")

        threshold_stats.append(
            {
                "threshold": thresh,
                "n_documents": int(n_above),
                "pct_documents": pct_above,
                "class_distribution": class_dist,
            }
        )

    # Save threshold stats
    with open(output_dir / "threshold_analysis.json", "w") as f:
        json.dump(threshold_stats, f, indent=2)

    # Plot 1: Confidence histogram
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(confidences, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Overall Confidence Distribution")
    axes[0].axvline(x=0.5, color="red", linestyle="--", label="0.5")
    axes[0].axvline(x=0.9, color="orange", linestyle="--", label="0.9")
    axes[0].legend()

    # Plot 2: Confidence by predicted class
    for i, name in enumerate(label_names):
        mask = predictions == i
        axes[1].hist(confidences[mask], bins=50, alpha=0.5, label=name)
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Confidence by Predicted Class")
    axes[1].legend()

    # Plot 3: Cumulative distribution (what % of data is above threshold)
    sorted_conf = np.sort(confidences)[::-1]
    cumulative_pct = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf) * 100
    axes[2].plot(sorted_conf, cumulative_pct)
    axes[2].set_xlabel("Confidence Threshold")
    axes[2].set_ylabel("% Documents Above Threshold")
    axes[2].set_title("Cumulative Distribution")
    axes[2].axhline(y=10, color="red", linestyle="--", alpha=0.5, label="10%")
    axes[2].axhline(y=5, color="orange", linestyle="--", alpha=0.5, label="5%")
    axes[2].axhline(y=1, color="green", linestyle="--", alpha=0.5, label="1%")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "confidence_distribution.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to {output_dir / 'confidence_distribution.png'}")

    # Sample high and low confidence examples
    print("\n" + "=" * 60)
    print("SAMPLE DOCUMENTS")
    print("=" * 60)

    # Sort by confidence
    sorted_results = sorted(results, key=lambda x: x["confidence"], reverse=True)

    print("\n--- TOP 5 HIGHEST CONFIDENCE ---")
    for r in sorted_results[:5]:
        print(
            f"\nConfidence: {r['confidence']:.4f} | Predicted: {label_names[r['predicted_label']]}"
        )
        print(f"Text: {r['document'][:300]}...")

    print("\n--- 5 NEAR 0.5 CONFIDENCE (UNCERTAIN) ---")
    mid_conf = sorted(results, key=lambda x: abs(x["confidence"] - 0.5))[:5]
    for r in mid_conf:
        print(
            f"\nConfidence: {r['confidence']:.4f} | Predicted: {label_names[r['predicted_label']]}"
        )
        print(f"Text: {r['document'][:300]}...")

    return threshold_stats


def main(args):
    # Load model config to get label names
    config_path = Path(args.model_path) / "training_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        label_names = config["data"]["label_names"]
        max_length = config["model"].get("max_length", 512)
    else:
        print("Warning: No training_config.yaml found, using defaults")
        label_names = ["negative", "positive"]
        max_length = 512

    print(f"Label names: {label_names}")

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    # Load data
    print(f"Loading data from {args.data_path}")
    data = load_jsonl(args.data_path)
    print(f"Loaded {len(data)} documents")

    # Run inference
    results = run_inference(
        data=data,
        model=model,
        tokenizer=tokenizer,
        device=device,
        text_field=args.text_field,
        batch_size=args.batch_size,
        max_length=max_length,
    )

    # Save predictions
    output_path = Path(args.output_dir) / "predictions.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(results, output_path)
    print(f"\nPredictions saved to {output_path}")

    # Analyze
    analyze_confidence_distribution(results, label_names, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and analyze confidence")

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model directory"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to JSONL data to predict on"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for outputs (predictions, plots)",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="document",
        help="Key containing text in JSONL",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )

    args = parser.parse_args()
    main(args)
