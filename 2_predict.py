"""
Apply trained classifier to new data.
Outputs predictions to JSONL.
"""

import argparse
import json
from pathlib import Path

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
    label_names: list[str],
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
            # Add probability for each class
            for k, name in enumerate(label_names):
                result[f"prob_{name}"] = float(probs[j][k])
            results.append(result)

    return results


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
        label_names=label_names,
        batch_size=args.batch_size,
        max_length=max_length,
    )

    # Save predictions
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(results, output_path)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained model")

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model directory"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to JSONL data to predict on"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path for output JSONL with predictions",
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
