"""
List top N documents by confidence, separately for each class.
"""

import argparse
import json
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


def main(args):
    print(f"Loading predictions from {args.predictions_path}")
    results = load_jsonl(args.predictions_path)
    print(f"Loaded {len(results)} predictions")

    label_names = get_label_names(results)
    print(f"Detected labels: {label_names}")

    # Split by class
    by_class = {i: [] for i in range(len(label_names))}
    for r in results:
        by_class[r["predicted_label"]].append(r)

    # Sort each class by confidence descending
    for label_idx in by_class:
        by_class[label_idx].sort(key=lambda x: x["confidence"], reverse=True)

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label_idx, label_name in enumerate(label_names):
        docs = by_class[label_idx][: args.n]

        print(f"\n{'=' * 80}")
        print(
            f"TOP {args.n} {label_name.upper()} (total in class: {len(by_class[label_idx])})"
        )
        print(f"{'=' * 80}")

        # Save to file
        output_file = output_dir / f"top_{args.n}_{label_name}.txt"

        with open(output_file, "w") as f:
            f.write(f"TOP {args.n} {label_name.upper()} PREDICTIONS\n")
            f.write(f"Total in class: {len(by_class[label_idx])}\n")
            f.write("=" * 80 + "\n\n")

            for i, doc in enumerate(docs, 1):
                text = doc.get(args.text_field, "NO TEXT")
                conf = doc["confidence"]

                # Print preview to terminal
                if i <= 10:  # Only print first 10 to terminal
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"\n[{i}] Confidence: {conf:.4f}")
                    print(f"{preview}")

                # Write full text to file
                f.write(f"[{i}] Confidence: {conf:.4f}\n")
                f.write("-" * 40 + "\n")
                f.write(text + "\n")
                f.write("\n" + "=" * 80 + "\n\n")

        print(f"\n... Full list saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List top N documents by confidence per class"
    )

    parser.add_argument(
        "--predictions_path", type=str, required=True, help="Path to predictions JSONL"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Number of top documents per class"
    )
    parser.add_argument(
        "--text_field", type=str, default="document", help="Key containing text"
    )

    args = parser.parse_args()
    main(args)
