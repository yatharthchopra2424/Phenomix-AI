"""
trainer.py
==========
Offline training script for the VariantFunctionClassifier.

Run from the project root:

  python -m ml_models.trainer --csv path/to/training_data.csv --epochs 50

Expected CSV columns:
  chrom, pos, ref, alt, label

  label must be one of: normal_function, decreased_function,
                        increased_function, no_function

The script:
  1. Loads the CSV of annotated variants.
  2. One-hot encodes each variant's genomic window.
  3. Applies class-weighted cross-entropy loss to handle data imbalance.
  4. Trains the VariantFunctionClassifier for N epochs with an 80/20 split.
  5. Saves the best checkpoint to ml_models/weights/variant_classifier.pt.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from ml_models.architecture import VariantFunctionClassifier, CLASS_LABELS
from ml_models.feature_encoder import encode_variant

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

WEIGHTS_DIR  = Path(__file__).parent / "weights"
WEIGHTS_FILE = WEIGHTS_DIR / "variant_classifier.pt"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VariantDataset(Dataset):
    def __init__(self, records: List[Tuple]):
        self.records = records
        self.label_map = {lbl: i for i, lbl in enumerate(CLASS_LABELS)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        chrom, pos, ref, alt, label_str = self.records[idx]
        encoding = encode_variant(chrom, int(pos), ref, alt)  # (4, 101)
        label_idx = self.label_map.get(label_str, 0)
        return torch.from_numpy(encoding), torch.tensor(label_idx, dtype=torch.long)


def _load_csv(path: str) -> List[Tuple]:
    records = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append((
                row["chrom"],
                int(row["pos"]),
                row["ref"],
                row["alt"],
                row["label"].strip().lower(),
            ))
    return records


def _compute_class_weights(records: List[Tuple]) -> torch.Tensor:
    """Inverse-frequency class weights to handle data imbalance."""
    label_map = {lbl: i for i, lbl in enumerate(CLASS_LABELS)}
    counts = [0] * len(CLASS_LABELS)
    for r in records:
        idx = label_map.get(r[4], 0)
        counts[idx] += 1
    total = sum(counts)
    weights = [total / (c + 1e-6) for c in counts]
    weights_t = torch.tensor(weights, dtype=torch.float32)
    return weights_t / weights_t.sum() * len(CLASS_LABELS)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(csv_path: str, epochs: int = 50, batch_size: int = 32, lr: float = 1e-3):
    records = _load_csv(csv_path)
    logger.info("Loaded %d training records from %s", len(records), csv_path)

    dataset = VariantDataset(records)
    n_val   = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = VariantFunctionClassifier().to(device)

    class_weights = _compute_class_weights(records).to(device)
    criterion = nn.NLLLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X)
        train_loss /= n_train

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        correct     = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y  = X.to(device), y.to(device)
                out   = model(X)
                val_loss += criterion(out, y).item() * len(X)
                correct  += (out.argmax(1) == y).sum().item()
        val_loss /= n_val
        val_acc   = correct / n_val

        scheduler.step(val_loss)
        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f",
            epoch, epochs, train_loss, val_loss, val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(WEIGHTS_FILE))
            logger.info("  ✓ checkpoint saved → %s", WEIGHTS_FILE)

    logger.info("Training complete.  Best val_loss=%.4f", best_val_loss)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VariantFunctionClassifier")
    parser.add_argument("--csv",    required=True, help="Path to training CSV")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=32)
    parser.add_argument("--lr",     type=float, default=1e-3)
    args = parser.parse_args()
    train(args.csv, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
