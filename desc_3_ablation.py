"""
Evaluate retrieval performance using only top N descriptors (by coefficient magnitude).
Tests how many descriptors are needed for good retrieval.
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

# Suppress sklearn warning about unknown classes in MultiLabelBinarizer
warnings.filterwarnings("ignore", message="unknown class.*")


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def evaluate_retrieval(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Compute retrieval metrics."""
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    # Precision/Recall at K
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]

    total_positives = labels.sum()

    k_values = [100, 500, 1000, 2500]
    pr_at_k = {}
    for k in k_values:
        if k > len(labels):
            continue
        top_k_labels = sorted_labels[:k]
        tp = top_k_labels.sum()
        pr_at_k[k] = {
            "precision": tp / k,
            "recall": tp / total_positives,
        }

    # Threshold for 80% recall
    cumsum = np.cumsum(sorted_labels)
    target_tp = int(0.8 * total_positives)
    idx = np.searchsorted(cumsum, target_tp)
    pct_data_for_80_recall = (idx + 1) / len(labels) * 100

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_at_k": pr_at_k,
        "pct_data_for_80_recall": pct_data_for_80_recall,
    }


def main(args):
    # Load model and vectorizer
    print(f"Loading model from {args.model_path}")
    model = joblib.load(Path(args.model_path) / "model.joblib")
    mlb = joblib.load(Path(args.model_path) / "vectorizer.joblib")

    # Load config
    config_path = Path(args.model_path) / "training_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    descriptor_field = config["data"]["descriptor_field"]
    label_names = config["data"]["label_names"]

    vocabulary = list(mlb.classes_)
    print(f"Total descriptors: {len(vocabulary)}")

    # Get coefficients (absolute value for ranking importance)
    if len(label_names) == 2:
        # Binary: single coefficient vector
        coef = model.coef_[0]
    else:
        # Multiclass: use max absolute coefficient across classes
        coef = np.abs(model.coef_).max(axis=0)

    coef_magnitude = np.abs(coef)
    sorted_indices = np.argsort(coef_magnitude)[::-1]

    # Print top descriptors
    print("\nTop 20 descriptors by coefficient magnitude:")
    for i, idx in enumerate(sorted_indices[:20]):
        print(f"  {i + 1:3d}. {coef[idx]:+.4f}  {vocabulary[idx]}")

    # Load data
    print(f"\nLoading data from {args.data_path}")
    data = load_jsonl(args.data_path)
    print(f"Loaded {len(data)} documents")

    # Check for source field
    if "source" not in data[0]:
        raise ValueError("Data must have 'source' field (imdb/fineweb)")

    # Extract descriptors and labels
    descriptors = [item[descriptor_field] for item in data]
    X_full = mlb.transform(descriptors)
    labels = np.array([1 if item["source"] == "imdb" else 0 for item in data])

    n_imdb = labels.sum()
    print(f"IMDB: {n_imdb}, FineWeb: {len(labels) - n_imdb}")

    # Test different numbers of top descriptors
    if args.n_descriptors:
        n_values = [int(n) for n in args.n_descriptors.split(",")]
    else:
        n_values = [5, 10, 20, 50, 100, 200, 500, 1000, len(vocabulary)]

    n_values = [n for n in n_values if n <= len(vocabulary)]

    print(f"\nTesting descriptor counts: {n_values}")
    print("\n" + "=" * 80)

    results = []

    for n in n_values:
        # Get top N descriptor indices
        top_n_indices = sorted_indices[:n]

        # Create mask for these descriptors only
        X_subset = X_full[:, top_n_indices]
        coef_subset = coef[top_n_indices]
        intercept = model.intercept_[0] if len(label_names) == 2 else model.intercept_

        # Compute scores manually: sigmoid(X @ coef + intercept)
        logits = X_subset @ coef_subset + intercept
        scores = 1 / (1 + np.exp(-logits))  # Probability of positive class

        # For retrieval, use max(prob, 1-prob) as confidence
        confidence = np.maximum(scores, 1 - scores)

        # Evaluate
        metrics = evaluate_retrieval(confidence, labels)
        metrics["n_descriptors"] = n

        # Get the descriptors used
        descriptors_used = [vocabulary[i] for i in top_n_indices]
        metrics["top_descriptors"] = descriptors_used[:10]  # Save top 10 for reference

        results.append(metrics)

        print(f"\nTop {n} descriptors:")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:  {metrics['pr_auc']:.4f}")
        print(f"  % data for 80% recall: {metrics['pct_data_for_80_recall']:.1f}%")
        if 1000 in metrics["precision_at_k"]:
            print(
                f"  P@1000: {metrics['precision_at_k'][1000]['precision'] * 100:.1f}%"
            )

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"\n{'N':<8} {'ROC-AUC':<10} {'PR-AUC':<10} {'P@1000':<10} {'Data for 80% recall':<20}"
    )
    print("-" * 58)

    for r in results:
        p_at_1000 = r["precision_at_k"].get(1000, {}).get("precision", 0) * 100
        print(
            f"{r['n_descriptors']:<8} {r['roc_auc']:<10.4f} {r['pr_auc']:<10.4f} {p_at_1000:<10.1f} {r['pct_data_for_80_recall']:<20.1f}"
        )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    with open(output_dir / "descriptor_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save the ranked descriptor list with coefficients
    ranked_descriptors = [
        {"rank": i + 1, "descriptor": vocabulary[idx], "coefficient": float(coef[idx])}
        for i, idx in enumerate(sorted_indices)
    ]
    with open(output_dir / "ranked_descriptors.json", "w") as f:
        json.dump(ranked_descriptors, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    n_vals = [r["n_descriptors"] for r in results]
    roc_aucs = [r["roc_auc"] for r in results]
    pr_aucs = [r["pr_auc"] for r in results]
    data_for_80 = [r["pct_data_for_80_recall"] for r in results]

    axes[0].plot(n_vals, roc_aucs, marker="o")
    axes[0].set_xlabel("Number of descriptors")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("ROC-AUC vs Number of Descriptors")
    axes[0].set_xscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(n_vals, pr_aucs, marker="o")
    axes[1].set_xlabel("Number of descriptors")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("PR-AUC vs Number of Descriptors")
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(n_vals, data_for_80, marker="o")
    axes[2].set_xlabel("Number of descriptors")
    axes[2].set_ylabel("% data retained for 80% recall")
    axes[2].set_title("Filtering Efficiency vs Number of Descriptors")
    axes[2].set_xscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "descriptor_ablation.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {output_dir}")
    print(f"  - descriptor_ablation.json")
    print(f"  - ranked_descriptors.json")
    print(f"  - descriptor_ablation.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval with varying descriptor counts"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained descriptor model directory",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to mixed dataset JSONL (must have 'source' field)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory for output"
    )
    parser.add_argument(
        "--n_descriptors",
        type=str,
        default=None,
        help="Comma-separated list of N values to test (default: 5,10,20,50,100,200,500,1000,all)",
    )

    args = parser.parse_args()
    main(args)
