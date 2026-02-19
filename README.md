# PharmaGuard

PharmaGuard is an AI-powered pharmacogenomics pipeline that predicts patient-specific drug risk from VCF files.
It combines:

- Rule-based PGx annotation (known variants)
- A deep learning fallback model (unknown PGx-region variants)
- Gene-to-phenotype and phenotype-to-risk clinical logic
- Retrieval-Augmented Generation (RAG) for guideline-grounded explanations

---

## Deep Learning Implementation (What I built)

The deep learning system is implemented in the `ml_models/` package and integrated directly into the API pipeline.

### 1) Model architecture

Defined in `ml_models/architecture.py` as `VariantFunctionClassifier`:

- **Input:** one-hot encoded genomic sequence window `(4, 101)`
- **Feature extractor:** 3 stacked 1D CNN blocks (`Conv1d + BatchNorm + ReLU + MaxPool`)
- **Context modeling:** 2-layer **BiLSTM** (bidirectional) for long-range sequence dependencies
- **Classifier head:** dense layers with BatchNorm + Dropout
- **Output:** `log_softmax` over 4 pharmacogenomic function classes:
  - `normal_function`
  - `decreased_function`
  - `increased_function`
  - `no_function`

This hybrid CNN+BiLSTM design captures both local motifs and broader sequence context.

### 2) Feature encoding pipeline

Implemented in `ml_models/feature_encoder.py`:

- Encodes a variant-centered sequence window into one-hot channels
- Uses deterministic synthetic flanking sequence generation in prototype mode
- Keeps center nucleotide aligned to the candidate variant allele

> In production, `_fetch_sequence()` is designed to be swapped for real hg38 sequence retrieval (e.g., pyfaidx/Ensembl).

### 3) Training pipeline

Implemented in `ml_models/trainer.py`:

- Reads labeled training CSV (`chrom, pos, ref, alt, label`)
- Builds PyTorch dataset and loaders
- Uses class-weighted `NLLLoss` to handle class imbalance
- Trains with Adam + LR scheduler + gradient clipping
- Tracks validation loss/accuracy and saves best checkpoint to:
  - `ml_models/weights/variant_classifier.pt`

### 4) Inference integration

Implemented in `ml_models/predictor.py` and wired into backend flow:

- Loads model once at startup in `backend/main.py` lifespan handler
- Performs variant function prediction with confidence score
- Supports CUDA when available; otherwise CPU fallback
- Exposes `predict_variant_function()` for downstream annotation

### 5) End-to-end API usage

In `backend/api/predict.py`, the model is used as an **ML fallback**:

1. Known PGx variants are matched from reference tables
2. Unmatched variants within PGx gene regions are flagged
3. Deep model predicts function class for those flagged variants
4. Predictions feed into diplotype assembly and risk classification

This means deep learning is not isolated—it is part of real clinical decision flow.

---

## Why this stands out

Compared with many prototype PGx tools, this implementation stands out for architectural and system-level reasons:

### 1) Hybrid intelligence (rules + deep learning)

Most systems choose one path:

- only static rule tables (high precision, low coverage), or
- only ML (high flexibility, lower interpretability for known variants)

PharmaGuard combines both:

- deterministic, reference-grade behavior for known variants
- model-based generalization for novel/unlisted PGx-region variants

### 2) Sequence-native modeling

Instead of manually engineered tabular features, the model learns directly from sequence context using CNN+BiLSTM.
This is closer to how regulatory and coding signals are actually embedded in DNA neighborhoods.

### 3) Production-aware lifecycle

The DL stack is not just a notebook experiment:

- dedicated architecture module
- reusable trainer
- checkpoint loader
- startup singleton initialization
- API-level operational fallback modes

### 4) Safe demo behavior

If weights are missing, predictor enters controlled demo mode with capped confidence.
This is a practical safety mechanism to avoid over-trusting random-weight predictions.

### 5) Clinically grounded output layer

Model predictions are only one component of final output.
They are merged into diplotype, phenotype, and guideline-based risk classification before user-facing recommendations are generated.

---

## System Architecture (High level)

1. **Upload VCF**
2. Parse and validate variants
3. Annotate against PGx reference
4. Run DL fallback for unresolved PGx-region variants
5. Build diplotypes and activity score
6. Infer phenotype and classify drug risk
7. Retrieve guideline context (RAG)
8. Generate clinician-readable explanation

For a full Mermaid tree diagram of the complete flow, see `SYSTEM_TREE.md`.

---

## Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Training the deep model

From project root:

```bash
python -m ml_models.trainer --csv path/to/training_data.csv --epochs 50 --batch 32 --lr 1e-3
```

Expected CSV columns:

- `chrom`
- `pos`
- `ref`
- `alt`
- `label`

Where `label` is one of:

- `normal_function`
- `decreased_function`
- `increased_function`
- `no_function`

---

## Current constraints and next improvements

Current prototype constraints:

- synthetic context sequence (not real reference genome)
- model quality depends on training data diversity and label quality

Recommended next upgrades:

1. Replace synthetic sequence fetcher with hg38-backed sequence extraction
2. Add calibration and per-class confidence reporting
3. Add external validation benchmarks per gene/drug cohort
4. Add model versioning and reproducibility metadata in API response

---

## Tech stack

- **Backend:** FastAPI, Pydantic, Python
- **Deep Learning:** PyTorch, NumPy
- **Genomics Pipeline:** custom PGx annotation + diplotype/phenotype logic
- **RAG:** ChromaDB, embeddings, LLM explanation layer
- **Frontend:** Next.js (TypeScript)

---

## Summary

PharmaGuard’s deep learning model is implemented as a real, integrated clinical-AI subsystem—
not just a standalone classifier.

Its key differentiator is the **hybrid PGx strategy**:

- trust known biology where available,
- generalize with deep sequence modeling where needed,
- and always surface decisions through clinically interpretable outputs.