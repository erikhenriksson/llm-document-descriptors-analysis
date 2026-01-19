import json
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

dataset = "core"
sample_n = 10000  # Set to an integer to sample n lines, None to use all


def main(filepath=f"data/descriptors_{dataset}_harmonized_labelled.jsonl"):
    # Load data - only extract fields we need to save memory
    descriptors_list = []
    labels = []

    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line)
            descriptors_list.append(data["harmonized_descriptors"])
            labels.append(data["label"])

    # Sample if requested
    if sample_n is not None and sample_n < len(descriptors_list):
        random.seed(42)
        indices = random.sample(range(len(descriptors_list)), sample_n)
        descriptors_list = [descriptors_list[i] for i in indices]
        labels = [labels[i] for i in indices]

    print(f"Loaded {len(descriptors_list)} documents")
    print(f"Unique labels: {sorted(set(labels))}")
    print(f"Label distribution: {Counter(labels)}")

    # Count all descriptor frequencies once
    all_descriptors = [d for doc in descriptors_list for d in doc]
    counts = Counter(all_descriptors)

    # Configs to test
    min_freqs = [0, 1, 5, 10]
    regularizations = [
        ("L2 C=10", {"penalty": "l2", "C": 10, "solver": "lbfgs"}),
        ("L2 C=1.0", {"penalty": "l2", "C": 1.0, "solver": "lbfgs"}),
        ("L2 C=0.1", {"penalty": "l2", "C": 0.1, "solver": "lbfgs"}),
        ("L1 C=10", {"penalty": "l1", "C": 10, "solver": "liblinear"}),
        ("L1 C=1.0", {"penalty": "l1", "C": 1.0, "solver": "liblinear"}),
        ("L1 C=0.1", {"penalty": "l1", "C": 0.1, "solver": "liblinear"}),
        ("None", {"penalty": None, "solver": "lbfgs"}),
    ]

    seeds = [42, 123, 456, 789, 1010]

    # Store results: {reg_name: [[acc for each min_freq] for each seed]}
    results = {name: [] for name, _ in regularizations}
    nonzero_counts = {name: [] for name, _ in regularizations}

    # Track descriptor counts for each min_freq
    descriptor_counts = []

    for seed in seeds:
        print(f"\nSeed {seed}")
        seed_results = {name: [] for name, _ in regularizations}
        seed_nonzero = {name: [] for name, _ in regularizations}

        for min_freq in min_freqs:
            # Filter descriptors
            valid_descriptors = {d for d, c in counts.items() if c > min_freq}
            filtered = [
                [d for d in doc if d in valid_descriptors] for doc in descriptors_list
            ]

            # Track counts only on first seed
            if seed == seeds[0]:
                descriptor_counts.append(len(valid_descriptors))

            # Convert to features
            mlb = MultiLabelBinarizer()
            X = mlb.fit_transform(filtered)
            y = labels

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )

            for name, config in regularizations:
                model = LogisticRegression(max_iter=1000, **config)
                model.fit(X_train, y_train)
                acc = accuracy_score(y_test, model.predict(X_test))
                seed_results[name].append(acc)

                # Count non-zero coefficients (across all classes)
                nonzero = np.sum(np.any(model.coef_ != 0, axis=0))
                seed_nonzero[name].append(nonzero)

                print(
                    f"  {name}: acc={acc:.4f}, nonzero={nonzero}, coef_shape={model.coef_.shape}, X_shape={X_train.shape}"
                )

        for name in seed_results:
            results[name].append(seed_results[name])
            nonzero_counts[name].append(seed_nonzero[name])

    # Convert to numpy for easy mean/std
    for name in results:
        results[name] = np.array(results[name])  # shape: (n_seeds, n_freqs)
        nonzero_counts[name] = np.array(nonzero_counts[name])

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Define styles: L2 solid, L1 dashed, None dotted
    # Colors by C value
    styles = {
        "L2 C=10": {"linestyle": "-", "color": "tab:blue"},
        "L2 C=1.0": {"linestyle": "-", "color": "tab:orange"},
        "L2 C=0.1": {"linestyle": "-", "color": "tab:green"},
        "L1 C=10": {"linestyle": "--", "color": "tab:blue"},
        "L1 C=1.0": {"linestyle": "--", "color": "tab:orange"},
        "L1 C=0.1": {"linestyle": "--", "color": "tab:green"},
        "None": {"linestyle": ":", "color": "tab:red", "linewidth": 2},
    }

    # Top plot: Accuracy
    for name, _ in regularizations:
        mean = results[name].mean(axis=0)
        std = results[name].std(axis=0)
        style = styles[name]

        ax1.plot(min_freqs, mean, marker="o", label=name, **style)
        ax1.fill_between(
            min_freqs, mean - std, mean + std, color=style["color"], alpha=0.1
        )

    ax1.set_ylabel("Accuracy")
    ax1.set_title(
        f"[{dataset}] Accuracy by Descriptor Frequency Threshold and Regularization\n(mean ± std over 5 seeds)"
    )
    ax1.legend()
    ax1.grid(True)

    # Secondary x-axis for descriptor counts on top plot
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(min_freqs)
    ax1_top.set_xticklabels([f"n={c}" for c in descriptor_counts])
    ax1_top.set_xlabel("Number of descriptors available")

    # Bottom plot: Non-zero coefficients
    for name, _ in regularizations:
        mean = nonzero_counts[name].mean(axis=0)
        std = nonzero_counts[name].std(axis=0)
        style = styles[name]

        ax2.plot(min_freqs, mean, marker="o", label=name, **style)
        ax2.fill_between(
            min_freqs, mean - std, mean + std, color=style["color"], alpha=0.1
        )

    ax2.set_xlabel("Minimum descriptor frequency (excluded if <= this)")
    ax2.set_ylabel("Non-zero coefficients")
    ax2.set_title("Number of Features Used by Model")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xticks(min_freqs)

    plt.tight_layout()
    plt.savefig(f"results_{dataset}.png")
    print(f"\nSaved plot to results_{dataset}.png")


if __name__ == "__main__":
    import sys

    filepath = (
        sys.argv[1]
        if len(sys.argv) > 1
        else f"data/descriptors_{dataset}_harmonized_labelled.jsonl"
    )
    main(filepath)
