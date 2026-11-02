"""TruthLens AI - Classifier Training Script
Trains or fine-tunes the forensic binary image classifier (Real vs AI-Generated)
using PyTorch, CrossEntropyLoss, and AdamW optimizer. Saves weights to checkpoints/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.classifier import TruthLensForensicNet
from ml.dataset import ImageForensicsDataset, create_synthetic_benchmark_dataset
from ml.model_loader import CHECKPOINT_DIR, DEFAULT_WEIGHTS_PATH, MODEL_METADATA


def train_classifier(
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cpu",
    output_checkpoint: str = DEFAULT_WEIGHTS_PATH,
) -> None:
    """Trains the forensic classifier and saves weights."""
    print(f"=== Starting TruthLens Classifier Training ===")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {lr} | Device: {device}")

    # Prepare datasets
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_samples, test_samples = create_synthetic_benchmark_dataset(num_samples_per_class=30)
    print(f"Loaded {len(train_samples)} training samples, {len(test_samples)} validation samples.")

    train_dataset = ImageForensicsDataset(train_samples)
    val_dataset = ImageForensicsDataset(test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TruthLensForensicNet(architecture="custom")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = val_correct / max(val_total, 1)
        duration = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.1f}% | "
            f"Val Acc: {val_acc * 100:.1f}% | Time: {duration:.1f}s"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": MODEL_METADATA,
                    "version": "1.2.0",
                    "best_val_acc": best_val_acc,
                },
                output_checkpoint,
            )

    print(f"Training completed! Best checkpoint saved to: {output_checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TruthLens Image Forensics Classifier")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train_classifier(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
