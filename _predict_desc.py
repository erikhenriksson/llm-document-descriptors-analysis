"""
Train logistic regression on descriptors and evaluate retrieval performance.
"""

import argparse
import json
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Suppress sklearn warnings about unknown features
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def load_jsonl(path):
    """Load JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def prepare_data(config):
    """Prepare same train/test split as BERT training."""

    print("Loading benchmark data...")
    benchmark_data = load_jsonl(config["benchmark"]["path"])

    print("Loading FineWeb data...")
    fineweb_data = load_jsonl(config["fineweb"]["path"])

    # Get positive examples
    positive_labels = config["benchmark"]["positive_labels"]
    positive_examples = [
        item for item in benchmark_data if item["label"] in positive_labels
    ]

    # Sample same amounts as BERT
    samples_per_class = config["training"]["samples_per_class"]
    n_positive = min(samples_per_class, len(positive_examples))

    positive_sample = random.sample(positive_examples, n_positive)
    negative_sample = random.sample(fineweb_data, n_positive)

    # Extract descriptors and labels
    positive_descriptors = [item["harmonized_descriptors"] for item in positive_sample]
    negative_descriptors = [item["harmonized_descriptors"] for item in negative_sample]

    all_descriptors = positive_descriptors + negative_descriptors
    all_labels = [1] * len(positive_descriptors) + [0] * len(negative_descriptors)

    print(
        f"Prepared {len(positive_descriptors)} positive and {len(negative_descriptors)} negative examples"
    )

    # Same train/test split as BERT
    test_split = config["training"]["test_split"]
    train_desc, test_desc, train_labels, test_labels = train_test_split(
        all_descriptors,
        all_labels,
        test_size=test_split,
        random_state=42,
        stratify=all_labels,
    )

    return train_desc, train_labels, test_desc, test_labels


def create_feature_matrix(descriptors_list, mlb=None):
    """Convert list of descriptor lists to multi-hot encoded matrix."""
    if mlb is None:
        mlb = MultiLabelBinarizer()
        X = mlb.fit_transform(descriptors_list)
    else:
        X = mlb.transform(descriptors_list)

    return X, mlb


def evaluate_at_thresholds(y_true, y_proba, thresholds):
    """Evaluate precision, recall, F1 at different thresholds."""
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        # Handle edge cases
        if y_pred.sum() == 0:
            prec = 0.0
            rec = 0.0
            f1 = 0.0
        else:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

        # Dataset reduction
        kept = y_pred.sum()
        reduction = 100 * (1 - kept / len(y_pred))

        results.append(
            {
                "threshold": threshold,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "kept": kept,
                "reduction_pct": reduction,
            }
        )

    return results


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate descriptor-based retrieval")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set random seed
    random.seed(42)
    np.random.seed(42)

    # Prepare data
    train_desc, train_labels, test_desc, test_labels = prepare_data(config)

    # Convert to multi-hot features
    print("\nConverting descriptors to features...")
    X_train, mlb = create_feature_matrix(train_desc)
    X_test, _ = create_feature_matrix(test_desc, mlb)

    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Number of unique descriptors: {len(mlb.classes_)}")

    # Train logistic regression
    print("\nTraining logistic regression (C=1, L2)...")
    clf = LogisticRegression(C=1.0, penalty="l2", max_iter=1000, random_state=42)
    clf.fit(X_train, train_labels)

    # Predict probabilities
    print("Generating predictions...")
    y_proba = clf.predict_proba(X_test)[:, 1]  # Probability of positive class

    # Compute metrics
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)

    # Average Precision
    ap = average_precision_score(test_labels, y_proba)
    print(f"Average Precision: {ap:.4f}")

    # Evaluate at different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = evaluate_at_thresholds(test_labels, y_proba, thresholds)

    print("\n" + "=" * 60)
    print("PERFORMANCE AT DIFFERENT THRESHOLDS")
    print("=" * 60)
    print(
        f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Reduction %':<12}"
    )
    print("-" * 60)

    for r in results:
        print(
            f"{r['threshold']:<12.2f} {r['precision']:<12.3f} {r['recall']:<12.3f} "
            f"{r['f1']:<12.3f} {r['reduction_pct']:<12.1f}"
        )

    # Find best F1 threshold
    best_f1_idx = max(range(len(results)), key=lambda i: results[i]["f1"])
    best_result = results[best_f1_idx]

    print("\n" + "=" * 60)
    print("BEST F1 THRESHOLD")
    print("=" * 60)
    print(f"Threshold: {best_result['threshold']:.2f}")
    print(f"Precision: {best_result['precision']:.3f}")
    print(f"Recall: {best_result['recall']:.3f}")
    print(f"F1: {best_result['f1']:.3f}")
    print(f"Dataset reduction: {best_result['reduction_pct']:.1f}%")

    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(test_labels, y_proba)

    # Find recall at high precision
    print("\n" + "=" * 60)
    print("RECALL AT HIGH PRECISION")
    print("=" * 60)

    for target_prec in [0.95, 0.90, 0.85, 0.80]:
        idx = np.where(precision >= target_prec)[0]
        if len(idx) > 0:
            best_recall = recall[idx].max()
            print(f"At precision ≥ {target_prec:.2f}: recall = {best_recall:.3f}")
        else:
            print(f"At precision ≥ {target_prec:.2f}: not achievable")

    # Dataset reduction at high recall
    print("\n" + "=" * 60)
    print("DATASET REDUCTION AT HIGH RECALL")
    print("=" * 60)

    for target_recall in [0.95, 0.90, 0.85, 0.80]:
        idx = np.where(recall >= target_recall)[0]
        if len(idx) > 0:
            # Get threshold at this recall
            threshold_idx = idx[-1]
            if threshold_idx < len(pr_thresholds):
                threshold = pr_thresholds[threshold_idx]
                y_pred = (y_proba >= threshold).astype(int)
                reduction = 100 * (1 - y_pred.sum() / len(y_pred))
                prec = precision[threshold_idx]
                print(
                    f"To keep recall ≥ {target_recall:.2f}: can filter {reduction:.1f}% "
                    f"(precision = {prec:.3f})"
                )

    # Save results
    output_dir = Path(config["output"]["predictions_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    with open(output_dir / "descriptor_eval.json", "w") as f:
        json.dump(
            {
                "average_precision": float(ap),
                "best_f1_threshold": {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in best_result.items()
                },
                "threshold_results": [
                    {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in r.items()
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {output_dir / 'descriptor_eval.json'}")

    # Plot PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve (AP = {ap:.3f})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plot_path = output_dir / "pr_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"PR curve saved to {plot_path}")


if __name__ == "__main__":
    main()
