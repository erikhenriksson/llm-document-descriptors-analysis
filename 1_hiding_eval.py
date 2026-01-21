"""
Evaluate retrieval: can BERT confidence separate IMDB from FineWeb?
Run this on predictions from the mixed dataset.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_precision_recall_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k_values: list[int],
) -> dict:
    """
    Compute precision and recall at various K values.
    labels: 1 = IMDB (positive), 0 = FineWeb (negative)
    """
    # Sort by score descending
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]

    total_positives = labels.sum()

    results = {}
    for k in k_values:
        if k > len(labels):
            k = len(labels)

        top_k_labels = sorted_labels[:k]
        true_positives = top_k_labels.sum()

        precision = true_positives / k
        recall = true_positives / total_positives if total_positives > 0 else 0

        results[k] = {
            "precision": float(precision),
            "recall": float(recall),
            "true_positives": int(true_positives),
            "k": k,
        }

    return results


def compute_threshold_analysis(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    """
    At each threshold, compute how much data we keep and what fraction is IMDB.
    """
    total_imdb = labels.sum()
    total_fineweb = len(labels) - total_imdb

    results = []
    for thresh in thresholds:
        above = scores >= thresh
        n_above = above.sum()

        imdb_above = (labels[above] == 1).sum()
        fineweb_above = (labels[above] == 0).sum()

        results.append(
            {
                "threshold": thresh,
                "n_retained": int(n_above),
                "pct_retained": float(n_above / len(labels) * 100),
                "imdb_retained": int(imdb_above),
                "imdb_recall": float(imdb_above / total_imdb * 100)
                if total_imdb > 0
                else 0,
                "fineweb_retained": int(fineweb_above),
                "fineweb_removed_pct": float(
                    (total_fineweb - fineweb_above) / total_fineweb * 100
                )
                if total_fineweb > 0
                else 0,
                "precision": float(imdb_above / n_above * 100) if n_above > 0 else 0,
            }
        )

    return results


def find_threshold_for_recall(
    scores: np.ndarray,
    labels: np.ndarray,
    target_recall: float,
) -> dict:
    """
    Find the threshold that achieves a target recall for IMDB.
    """
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]
    sorted_scores = scores[sorted_indices]

    total_positives = labels.sum()
    target_tp = int(np.ceil(target_recall * total_positives))

    cumsum = np.cumsum(sorted_labels)
    idx = np.searchsorted(cumsum, target_tp)

    if idx >= len(sorted_scores):
        idx = len(sorted_scores) - 1

    threshold = sorted_scores[idx]

    # Compute stats at this threshold
    above = scores >= threshold
    n_above = above.sum()
    imdb_above = (labels[above] == 1).sum()

    return {
        "target_recall": target_recall,
        "threshold": float(threshold),
        "actual_recall": float(imdb_above / total_positives)
        if total_positives > 0
        else 0,
        "n_retained": int(n_above),
        "pct_data_retained": float(n_above / len(labels) * 100),
        "precision": float(imdb_above / n_above * 100) if n_above > 0 else 0,
    }


def main(args):
    print(f"Loading predictions from {args.predictions_path}")
    data = load_jsonl(args.predictions_path)
    print(f"Loaded {len(data)} documents")

    # Check for source field
    if "source" not in data[0]:
        raise ValueError("Predictions must have 'source' field (imdb/fineweb)")

    # Extract scores and labels
    # Score = max(prob_positive, prob_negative) = confidence that it's a review
    scores = np.array([d["confidence"] for d in data])
    labels = np.array([1 if d["source"] == "imdb" else 0 for d in data])

    n_imdb = labels.sum()
    n_fineweb = len(labels) - n_imdb

    print(f"\nDataset composition:")
    print(f"  IMDB: {n_imdb} ({n_imdb / len(labels) * 100:.2f}%)")
    print(f"  FineWeb: {n_fineweb} ({n_fineweb / len(labels) * 100:.2f}%)")

    # Compute metrics
    print("\n" + "=" * 60)
    print("RETRIEVAL METRICS")
    print("=" * 60)

    # ROC-AUC and PR-AUC
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    # Precision/Recall at K
    k_values = [100, 500, 1000, 2000, 2500, 3000, 5000, 10000]
    k_values = [k for k in k_values if k <= len(data)]

    print(f"\nPrecision/Recall at K:")
    print(f"{'K':<8} {'Precision':<12} {'Recall':<12} {'IMDB found':<12}")
    print("-" * 44)

    pr_at_k = compute_precision_recall_at_k(scores, labels, k_values)
    for k in k_values:
        r = pr_at_k[k]
        print(
            f"{k:<8} {r['precision'] * 100:<12.2f} {r['recall'] * 100:<12.2f} {r['true_positives']:<12}"
        )

    # Threshold analysis
    print(f"\nThreshold analysis:")
    print(
        f"{'Thresh':<8} {'Retained':<10} {'IMDB Recall':<12} {'FineWeb Removed':<16} {'Precision':<10}"
    )
    print("-" * 56)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    thresh_results = compute_threshold_analysis(scores, labels, thresholds)

    for r in thresh_results:
        print(
            f"{r['threshold']:<8.2f} {r['pct_retained']:<10.1f} {r['imdb_recall']:<12.1f} {r['fineweb_removed_pct']:<16.1f} {r['precision']:<10.1f}"
        )

    # Find thresholds for target recalls
    print(f"\nThresholds for target IMDB recall:")
    print(
        f"{'Target Recall':<15} {'Threshold':<12} {'Actual Recall':<15} {'Data Retained':<15} {'Precision':<10}"
    )
    print("-" * 67)

    target_recalls = [0.95, 0.90, 0.80, 0.70, 0.50]
    recall_thresholds = []
    for target in target_recalls:
        r = find_threshold_for_recall(scores, labels, target)
        recall_thresholds.append(r)
        print(
            f"{r['target_recall'] * 100:<15.0f} {r['threshold']:<12.4f} {r['actual_recall'] * 100:<15.1f} {r['pct_data_retained']:<15.1f} {r['precision']:<10.1f}"
        )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_recall_at_k": pr_at_k,
        "threshold_analysis": thresh_results,
        "recall_thresholds": recall_thresholds,
        "dataset": {
            "total": len(data),
            "imdb": int(n_imdb),
            "fineweb": int(n_fineweb),
        },
    }

    with open(output_dir / "retrieval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    axes[0].plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PR curve
    precision, recall, _ = precision_recall_curve(labels, scores)
    axes[1].plot(recall, precision, label=f"PR (AUC={pr_auc:.3f})")
    axes[1].axhline(
        y=n_imdb / len(labels),
        color="r",
        linestyle="--",
        alpha=0.5,
        label="Random baseline",
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Recall vs Data Retained
    thresh_fine = np.linspace(scores.min(), scores.max(), 100)
    recalls = []
    data_retained = []
    for t in thresh_fine:
        above = scores >= t
        imdb_above = (labels[above] == 1).sum()
        recalls.append(imdb_above / n_imdb * 100)
        data_retained.append(above.sum() / len(labels) * 100)

    axes[2].plot(data_retained, recalls)
    axes[2].set_xlabel("% Data Retained")
    axes[2].set_ylabel("% IMDB Recall")
    axes[2].set_title("IMDB Recall vs Data Retained")
    axes[2].axhline(y=90, color="r", linestyle="--", alpha=0.5, label="90% recall")
    axes[2].axhline(y=80, color="orange", linestyle="--", alpha=0.5, label="80% recall")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "retrieval_curves.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {output_dir}")
    print(f"  - retrieval_metrics.json")
    print(f"  - retrieval_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate IMDB retrieval from mixed dataset"
    )

    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Path to predictions JSONL (must have 'source' field)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for output metrics and plots",
    )

    args = parser.parse_args()
    main(args)
