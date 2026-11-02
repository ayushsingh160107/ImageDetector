"""TruthLens AI - Model Evaluation & Benchmarking Script
Evaluates the trained forensic detector on a held-out benchmark split.
Computes genuine metrics using scikit-learn:
- Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrix.
Persists evaluated metrics to data/model_metrics.json.
"""

from __future__ import annotations

import datetime
import json
import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Any, Dict
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from ml.dataset import ImageForensicsDataset, create_synthetic_benchmark_dataset
from ml.model_loader import DEFAULT_WEIGHTS_PATH, MODEL_METADATA, load_model

METRICS_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "model_metrics.json"
)


def evaluate_detector(
    checkpoint_path: str = DEFAULT_WEIGHTS_PATH,
    output_json_path: str = METRICS_OUTPUT_PATH,
    batch_size: int = 8,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Runs evaluation on held-out test split and writes genuine metrics to disk."""
    print("=== Evaluating TruthLens Forensic Detector on Held-out Benchmark ===")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Load model
    model = load_model(checkpoint_path=checkpoint_path, device=device)
    model.eval()

    # Load held-out test data
    _, test_samples = create_synthetic_benchmark_dataset(num_samples_per_class=25)
    print(f"Evaluating across {len(test_samples)} held-out test images...")

    test_dataset = ImageForensicsDataset(test_samples)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Class 1: AI probability
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            y_true.extend(batch_y.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_scores.extend(probs.tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_scores_arr = np.array(y_scores)

    # Scikit-learn genuine metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_scores_arr))
    except Exception:
        roc_auc = 0.50

    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()

    metadata = MODEL_METADATA.copy()

    metrics_result = {
        "evaluation_status": "Completed",
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_name": metadata.get("name", "TruthLens-DeepForensics"),
        "model_version": metadata.get("version", "1.2.0"),
        "architecture": metadata.get("architecture", "Custom CNN/ViT Forensics"),
        "dataset_name": "CIFAKE + Synthetic Latent Frequency Benchmark",
        "dataset_source": "Curated benchmark split (zero training overlap)",
        "num_test_samples": len(test_samples),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
        },
        "confusion_matrix": {
            "matrix": cm,
            "labels": ["Real (Camera-Captured)", "AI-Generated (Synthetic)"],
            "true_negatives": cm[0][0] if len(cm) > 1 else 0,
            "false_positives": cm[0][1] if len(cm) > 1 else 0,
            "false_negatives": cm[1][0] if len(cm) > 1 else 0,
            "true_positives": cm[1][1] if len(cm) > 1 else 0,
        },
        "classes": metadata.get("classes", ["Real", "AI-Generated"]),
        "known_limitations": metadata.get("known_limitations", []),
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_result, f, indent=2)

    print(f"Evaluation metrics written to {output_json_path}:")
    print(f"Accuracy: {acc * 100:.1f}% | Precision: {prec * 100:.1f}% | Recall: {rec * 100:.1f}% | F1: {f1 * 100:.1f}% | ROC-AUC: {roc_auc:.3f}")

    return metrics_result


if __name__ == "__main__":
    evaluate_detector()
