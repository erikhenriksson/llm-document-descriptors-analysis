"""
Sample documents from specific confidence bands for manual inspection.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


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
            label_names.append(key[5:])
    return sorted(label_names)


def sample_by_confidence_bands(
    results: list[dict],
    label_names: list[str],
    bands: list[tuple[float, float]],
    n_samples: int,
    text_field: str,
    seed: int = 42,
) -> dict:
    """
    Sample n documents from each confidence band, split by predicted class.
    """
    random.seed(seed)

    samples = {}

    for low, high in bands:
        band_name = f"{low:.2f}-{high:.2f}"

        # Filter to this band
        in_band = [r for r in results if low <= r["confidence"] < high]

        # Split by class
        by_class = defaultdict(list)
        for r in in_band:
            by_class[r["predicted_label"]].append(r)

        samples[band_name] = {
            "total_in_band": len(in_band),
            "by_class": {},
        }

        for label_idx, label_name in enumerate(label_names):
            class_docs = by_class[label_idx]
            n_to_sample = min(n_samples, len(class_docs))
            sampled = random.sample(class_docs, n_to_sample) if class_docs else []

            samples[band_name]["by_class"][label_name] = {
                "total": len(class_docs),
                "sampled": n_to_sample,
                "documents": sampled,
            }

    return samples


def print_samples(
    samples: dict,
    label_names: list[str],
    text_field: str,
    max_text_length: int = 500,
):
    """Pretty print samples for terminal inspection."""

    for band_name, band_data in samples.items():
        print("\n" + "=" * 80)
        print(f"CONFIDENCE BAND: {band_name}")
        print(f"Total documents in band: {band_data['total_in_band']}")
        print("=" * 80)

        for label_name in label_names:
            class_data = band_data["by_class"][label_name]
            print(
                f"\n--- {label_name.upper()} ({class_data['total']} total, showing {class_data['sampled']}) ---"
            )

            for i, doc in enumerate(class_data["documents"], 1):
                text = doc.get(text_field, "NO TEXT")
                text_preview = (
                    text[:max_text_length] + "..."
                    if len(text) > max_text_length
                    else text
                )

                print(f"\n[{i}] Confidence: {doc['confidence']:.4f}")
                print(f"Text: {text_preview}")
                print("-" * 40)


def main(args):
    print(f"Loading predictions from {args.predictions_path}")
    results = load_jsonl(args.predictions_path)
    print(f"Loaded {len(results)} predictions")

    label_names = get_label_names(results)
    print(f"Detected labels: {label_names}")

    # Define confidence bands
    if args.bands:
        # Parse custom bands like "0.5-0.6,0.6-0.7,0.9-1.0"
        bands = []
        for b in args.bands.split(","):
            low, high = b.split("-")
            bands.append((float(low), float(high)))
    else:
        # Default bands
        bands = [
            (0.50, 0.60),
            (0.60, 0.70),
            (0.70, 0.80),
            (0.80, 0.90),
            (0.90, 0.95),
            (0.95, 0.99),
            (0.99, 1.01),  # 1.01 to include 0.99-1.0
        ]

    print(f"\nSampling {args.n_samples} per class from bands: {bands}")

    samples = sample_by_confidence_bands(
        results=results,
        label_names=label_names,
        bands=bands,
        n_samples=args.n_samples,
        text_field=args.text_field,
        seed=args.seed,
    )

    # Print to terminal
    print_samples(samples, label_names, args.text_field, args.max_text_length)

    # Save to file
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)

        print(f"\n\nSamples saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"\n{'Band':<12} {'Total':<10} "
        + " ".join(f"{name:<12}" for name in label_names)
    )
    print("-" * (24 + 12 * len(label_names)))

    for band_name, band_data in samples.items():
        row = f"{band_name:<12} {band_data['total_in_band']:<10} "
        for label_name in label_names:
            count = band_data["by_class"][label_name]["total"]
            row += f"{count:<12}"
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample documents by confidence bands")

    parser.add_argument(
        "--predictions_path", type=str, required=True, help="Path to predictions JSONL"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save samples JSON (optional)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=5, help="Number of samples per class per band"
    )
    parser.add_argument(
        "--text_field", type=str, default="document", help="Key containing text"
    )
    parser.add_argument(
        "--max_text_length",
        type=int,
        default=500,
        help="Max characters to show per document",
    )
    parser.add_argument(
        "--bands", type=str, default=None, help="Custom bands, e.g. '0.5-0.6,0.9-1.0'"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    args = parser.parse_args()
    main(args)
