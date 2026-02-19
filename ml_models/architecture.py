"""
architecture.py
===============
PyTorch neural network for pharmacogenomic variant function classification.

Architecture:

  Input  →  (batch, 4, 101)  [one-hot encoded 101-bp window]
         ↓
  Conv1D Block × 3            [local motif detection]
    Conv1d(in, out, kernel=7) → BatchNorm1d → ReLU → MaxPool1d(2)
    Channels: 4 → 64 → 128 → 256
         ↓
  BiLSTM × 2                  [long-range dependencies]
    hidden=128, bidirectional → output (batch, seq, 256)
    LayerNorm → Dropout(0.3)
         ↓
  Global Average Pool         [temporal aggregation]
         ↓
  Dense 256 → 128 → 4         [classification head]
    BatchNorm → ReLU → Dropout(0.3) between layers
         ↓
  LogSoftmax                  [output over 4 function classes]

Classes (index → label):
  0 → normal_function
  1 → decreased_function
  2 → increased_function
  3 → no_function
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ordered class labels — index must stay fixed (used in predictor.py)
CLASS_LABELS = [
    "normal_function",
    "decreased_function",
    "increased_function",
    "no_function",
]


class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int = 7):
        super().__init__()
        # Padding 'same' equivalent: (kernel - 1) // 2
        pad = (kernel - 1) // 2
        self.conv  = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=pad)
        self.bn    = nn.BatchNorm1d(out_channels)
        self.pool  = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class VariantFunctionClassifier(nn.Module):
    """
    1D-CNN + BiLSTM classifier for pharmacogenomic variant function prediction.

    Input  : (batch, 4, seq_len)  — one-hot encoded sequence
    Output : (batch, 4)           — log-softmax over CLASS_LABELS
    """

    def __init__(self, seq_len: int = 101, dropout: float = 0.3):
        super().__init__()

        # ── Convolutional front-end ──────────────────────────────────────────
        self.conv_blocks = nn.Sequential(
            ConvBlock(4,   64,  kernel=7),   # (batch, 64,  50)
            ConvBlock(64,  128, kernel=7),   # (batch, 128, 25)
            ConvBlock(128, 256, kernel=7),   # (batch, 256, 12)
        )

        # ── Bi-directional LSTM ──────────────────────────────────────────────
        # After 3 MaxPool(2) layers: seq_len → 101 // 8 ≈ 12
        self.bilstm = nn.LSTM(
            input_size   = 256,
            hidden_size  = 128,
            num_layers   = 2,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout,
        )
        self.layer_norm = nn.LayerNorm(256)  # 128 * 2 directions
        self.dropout    = nn.Dropout(dropout)

        # ── Classification head ──────────────────────────────────────────────
        self.dense1 = nn.Linear(256, 128)
        self.bn1    = nn.BatchNorm1d(128)
        self.dense2 = nn.Linear(128, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 4, seq_len)
        features = self.conv_blocks(x)       # (batch, 256, reduced_len)

        # LSTM expects (batch, seq, features)
        lstm_in = features.permute(0, 2, 1)  # (batch, reduced_len, 256)
        lstm_out, _ = self.bilstm(lstm_in)   # (batch, reduced_len, 256)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        # Global average pooling over the temporal dimension
        pooled = lstm_out.mean(dim=1)         # (batch, 256)

        # Classification
        out = F.relu(self.bn1(self.dense1(pooled)))
        out = self.dropout(out)
        logits = self.dense2(out)            # (batch, 4)
        return F.log_softmax(logits, dim=-1)
