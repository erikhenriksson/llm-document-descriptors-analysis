"""
Analyze predictions from predict.py.
Run this on the output JSONL to investigate confidence distributions and thresholds.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_label_names(data: list[dict]) -> list[str]:
    """Infer label names from prob_* keys."""
    sample = data[0]
    label_names = []
    for key in sample.keys():
        if key.startswith("prob_"):
            label_names.append(key[5:])  # Remove "prob_" prefix
    return sorted(label_names)


def analyze_confidence_distribution(
    results: list[dict],
    label_names: list[str],
    output_dir: str,
    text_field: str = "document",
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

    # Percentiles
    print(f"\nConfidence percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(confidences, p):.4f}")

    # Prediction distribution
    print(f"\nPrediction distribution:")
    for i, name in enumerate(label_names):
        count = (predictions == i).sum()
        pct = count / len(predictions) * 100
        mean_conf = (
            confidences[predictions == i].mean() if (predictions == i).any() else 0
        )
        print(f"  {name}: {count} ({pct:.1f}%) | mean confidence: {mean_conf:.4f}")

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
            name: int((high_conf_preds == i).sum())
            for i, name in enumerate(label_names)
        }

        print(f"  >= {thresh:.2f}: {n_above:>6} ({pct_above:>5.1f}%) | {class_dist}")

        threshold_stats.append(
            {
                "threshold": thresh,
                "n_documents": int(n_above),
                "pct_documents": float(pct_above),
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
        if mask.any():
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
        text = r.get(text_field, "NO TEXT FIELD")
        print(
            f"\nConfidence: {r['confidence']:.4f} | Predicted: {label_names[r['predicted_label']]}"
        )
        print(f"Text: {text[:300]}...")

    print("\n--- 5 NEAR 0.5 CONFIDENCE (UNCERTAIN) ---")
    mid_conf = sorted(results, key=lambda x: abs(x["confidence"] - 0.5))[:5]
    for r in mid_conf:
        text = r.get(text_field, "NO TEXT FIELD")
        print(
            f"\nConfidence: {r['confidence']:.4f} | Predicted: {label_names[r['predicted_label']]}"
        )
        print(f"Text: {text[:300]}...")

    # Save samples for manual inspection
    samples = {
        "highest_confidence": sorted_results[:20],
        "lowest_confidence": sorted_results[-20:],
        "near_threshold_0.5": sorted(results, key=lambda x: abs(x["confidence"] - 0.5))[
            :20
        ],
        "near_threshold_0.9": sorted(results, key=lambda x: abs(x["confidence"] - 0.9))[
            :20
        ],
    }

    with open(output_dir / "sample_documents.json", "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\nSample documents saved to {output_dir / 'sample_documents.json'}")

    return threshold_stats


def main(args):
    print(f"Loading predictions from {args.predictions_path}")
    results = load_jsonl(args.predictions_path)
    print(f"Loaded {len(results)} predictions")

    # Infer label names from data
    label_names = get_label_names(results)
    print(f"Detected labels: {label_names}")

    analyze_confidence_distribution(
        results=results,
        label_names=label_names,
        output_dir=args.output_dir,
        text_field=args.text_field,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze prediction confidence distribution"
    )

    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Path to predictions JSONL from predict.py",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory for analysis outputs"
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="document",
        help="Key containing text in JSONL (for sampling)",
    )

    args = parser.parse_args()
    main(args)
