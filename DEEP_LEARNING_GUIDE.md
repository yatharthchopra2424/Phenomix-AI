# Deep Learning Guide (Simple and Concise)

This file is intentionally split in order:

1. Basic terms and beginner questions
2. PharmaGuard project explanation

---

## 1) Basic Terms

- **Machine Learning (ML):** systems that learn patterns from data.
- **Deep Learning (DL):** ML using multi-layer neural networks.
- **Neural Network:** a model that maps input to output using learned weights.
- **Classification:** predicting one class out of many classes.
- **Epoch:** one full pass over the training data.
- **Batch size:** number of samples processed per step.
- **Learning rate:** size of each update step.
- **Loss function:** value the model tries to minimize during training.
- **NLLLoss:** loss used with log-probability outputs.
- **Overfitting:** model performs well on training data but poorly on new data.
- **Dropout:** random deactivation of neurons during training to reduce overfitting.
- **BatchNorm / LayerNorm:** normalization methods to stabilize training.
- **BiLSTM:** sequence layer that reads context in both directions.
- **Checkpoint:** saved model weights from training.

---

## 2) Beginner Questions (Quick Answers)

### Why use deep learning?
Because DNA sequence patterns are complex. DL can learn local motifs and broader context automatically.

### What is the model input?
A variant-centered DNA window encoded as one-hot matrix of shape `(4, 101)`.

### What is the model output?
One of 4 classes:

- `normal_function`
- `decreased_function`
- `increased_function`
- `no_function`

### Why not only rule-based tables?
Tables are strong for known variants, but cannot generalize well to unknown variants.

### Why not only ML?
Known PGx variants should still use curated clinical references for reliability.

---

## 3) PharmaGuard Deep Learning (Project Part)

PharmaGuard uses a **hybrid pipeline**:

1. Known variants → rule-based PGx lookup
2. Unknown variants in PGx regions → deep learning fallback

The DL part is in `ml_models/`.

### Model architecture (`ml_models/architecture.py`)

- Conv1D blocks for local sequence motifs
- BiLSTM layers for long-range context
- Dense classifier head
- `log_softmax` output over 4 classes

### Feature encoding (`ml_models/feature_encoder.py`)

- Builds a 101 bp centered window
- One-hot encodes to `(4, 101)`
- Current prototype uses deterministic synthetic flanks for reproducibility

### Training (`ml_models/trainer.py`)

- Reads CSV: `chrom, pos, ref, alt, label`
- Uses class-weighted `NLLLoss` for imbalance
- Uses Adam + scheduler + gradient clipping
- Saves best checkpoint to `ml_models/weights/variant_classifier.pt`

### Inference (`ml_models/predictor.py`)

- Loads model once at backend startup
- Uses GPU if available, else CPU
- If checkpoint missing: demo mode with capped confidence (`0.50`)

### API integration (`backend/api/predict.py`)

- Model is called only for unresolved PGx-region variants
- Predictions feed into diplotype, phenotype, and risk pipeline

---

## 4) Why These Choices (Simple)

- **CNN:** good for short local DNA motifs.
- **BiLSTM:** captures wider sequence context.
- **Class weighting:** handles imbalanced PGx labels.
- **Checkpoint by validation loss:** keeps best generalizing model.
- **Startup loading:** faster API inference.
- **Demo confidence cap:** safer behavior when trained weights are absent.

---

## 5) Current Limits and Next Steps

### Current limits

- Synthetic flank sequence in prototype mode
- Performance depends on training data quality and coverage

### Next upgrades

1. Use real hg38 sequence retrieval
2. Add confidence calibration
3. Add external validation benchmarks

---

## 6) One-Line Summary

PharmaGuard’s deep learning is a practical fallback model inside a clinical pipeline: rule-based for known variants, DL for unknown PGx-region variants.