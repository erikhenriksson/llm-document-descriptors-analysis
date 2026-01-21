"""
Create mixed dataset: IMDB test set + FineWeb.
Adds 'source' field to distinguish origin.
"""

import argparse
import json
from pathlib import Path

import numpy as np


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


def extract_test_split(
    data: list[dict],
    test_size: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    """
    Extract test split using same logic as training script.
    """
    np.random.seed(seed)

    # Get labels for stratification
    labels = np.array([item["label"] for item in data])
    indices = np.arange(len(data))

    # Stratified split
    test_indices = []
    for label in np.unique(labels):
        label_indices = indices[labels == label]
        n_test = int(len(label_indices) * test_size)
        np.random.shuffle(label_indices)
        test_indices.extend(label_indices[:n_test])

    test_data = [data[i] for i in test_indices]
    return test_data


def main(args):
    # Load IMDB
    print(f"Loading IMDB from {args.imdb_path}")
    imdb_data = load_jsonl(args.imdb_path)
    print(f"Loaded {len(imdb_data)} IMDB documents")

    # Extract test split
    imdb_test = extract_test_split(imdb_data, test_size=args.test_size, seed=args.seed)
    print(f"Extracted {len(imdb_test)} IMDB test documents")

    # Add source field
    for item in imdb_test:
        item["source"] = "imdb"

    # Load FineWeb
    print(f"Loading FineWeb from {args.fineweb_path}")
    fineweb_data = load_jsonl(args.fineweb_path)
    print(f"Loaded {len(fineweb_data)} FineWeb documents")

    # Add source field
    for item in fineweb_data:
        item["source"] = "fineweb"

    # Mix
    mixed_data = imdb_test + fineweb_data
    print(f"\nMixed dataset: {len(mixed_data)} total")
    print(f"  - IMDB: {len(imdb_test)} ({len(imdb_test) / len(mixed_data) * 100:.2f}%)")
    print(
        f"  - FineWeb: {len(fineweb_data)} ({len(fineweb_data) / len(mixed_data) * 100:.2f}%)"
    )

    # Shuffle
    np.random.seed(args.seed)
    np.random.shuffle(mixed_data)

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(mixed_data, output_path)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create mixed IMDB test + FineWeb dataset"
    )

    parser.add_argument(
        "--imdb_path", type=str, required=True, help="Path to IMDB JSONL"
    )
    parser.add_argument(
        "--fineweb_path", type=str, required=True, help="Path to FineWeb JSONL"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path for output mixed JSONL"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Fraction of IMDB to use as test (default: 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (must match training script)"
    )

    args = parser.parse_args()
    main(args)
