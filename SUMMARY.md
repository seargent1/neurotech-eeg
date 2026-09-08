# NT@B FA26 Project Summary

## What I Built

A complete Python pipeline for left vs. right motor imagery classification on PhysioNet EEGMMIDB:

1. **Data loader** (`src/ntab_bci/data.py`): Downloads only requested runs via MNE (~50-100 MB, not 1.9 GB)
2. **Classifier** (`src/ntab_bci/model.py`): CSP + LDA with regularization
3. **Evaluation** (`src/ntab_bci/evaluation.py`): Within/cross-subject validation
4. **Training script** (`scripts/train.py`): Full evaluation pipeline
5. **Prediction CLI** (`scripts/predict.py`): Inference on new EDF files

## Quick Test Results (3 subjects)

```
Test Accuracy:  58.3%
Train Accuracy: 74.8%
Overfitting gap: 16.4%
```

**Interpretation:**
- Above chance (50%) ✓
- Moderate overfitting (regularization needed)
- Small sample size (135 epochs) limits conclusions

## How This Addresses the Brief

### 1. Data Processing Pipeline
- Band-pass filter 8-30 Hz (mu/beta sensorimotor rhythms)
- Epoch extraction 0.5-3.5s post-cue
- Average reference
- Stratified train/test splits

### 2. Model Capabilities
**Can do:**
- Distinguish left/right imagery within-subject (60-70% expected)
- Extract interpretable spatial patterns (CSP filters)

**Cannot do:**
- Generalize well to new subjects (cross-subject ~10-15% drop)
- Handle high inter-subject variability
- Work with artifact-heavy data

### 3. Not an Artifact
- Stratified splits maintain class balance
- Per-subject splits prevent session leakage
- Leave-one-subject-out tests cross-person generalization

### 4. Baselines
- **Chance:** 50% (binary classification)
- **Permutation test:** Shuffle labels, retrain → builds null distribution
- **Subject-specific variance:** Range shows best/worst performers

### 5. Person vs. Task
- **Within-subject:** Tests task learning
- **Cross-subject:** Tests person-independent features
- **Gap between them:** Quantifies person-specific variance

### 6. Learning vs. Memorizing
- Train/test split within-subject
- Gap of 10-20% is typical (small samples)
- CSP regularization + LDA shrinkage combat overfitting

## Weakest Point

**Small sample size:** ~40 epochs per subject × 3 runs = high variance.

A single "good" or "bad" subject can skew results. With only 10 subjects, one outlier (e.g., 90% accuracy due to head movement artifact) could make the model look better than it is.

**Validation:** Would need 50+ subjects to be confident.

## Something You Didn't Ask About

**Executed vs. Imagined:** The brief lets us choose. Imagery is harder (weaker signal) but more BCI-relevant. Execution has stronger ERD/ERS but includes proprioceptive feedback.

**Interesting question:** Does a model trained on executed movement transfer to imagined movement? This tests whether the model learned "motor planning" vs. "muscle feedback."

## Next Steps (With More Time)

1. **Full dataset:** 109 subjects for robust cross-subject evaluation
2. **Artifact rejection:** ICA for eye blinks, autoreject for bad epochs
3. **Hyperparameter search:** Grid search CSP components, frequency bands
4. **Deep learning:** EEGNet or Shallow ConvNet for end-to-end learning
5. **Transfer learning:** Train on executed, fine-tune on imagined
6. **Subject-specific calibration:** Few-shot adaptation for new users

## AI Tool Usage

**Used:**
- Claude (Cursor) for scaffolding, documentation, evaluation design
- GitHub Copilot for boilerplate code completion

**One specific choice I own:**
CSP regularization (`reg=0.05`) is empirical. Too low → overfits to noise. Too high → underfits. 0.05 is a typical default but should be tuned per-subject.

**Why:** Small sample size (40 epochs) means covariance matrices are noisy. Regularization shrinks toward the identity, stabilizing filters.

## Files Delivered

```
ntab-eeg-bci/
├── README.md               ← Setup instructions
├── SUMMARY.md             ← This file
├── config.yaml            ← Experiment configuration
├── requirements.txt       ← Dependencies
├── pyproject.toml         ← Packaging
├── .gitignore
├── scripts/
│   ├── train.py           ← Full training + evaluation
│   ├── predict.py         ← Prediction CLI ⭐
│   ├── test_download.py   ← Smoke test (1 subject)
│   └── quick_test.py      ← Pipeline test (3 subjects)
├── src/ntab_bci/
│   ├── data.py            ← PhysioNet loader
│   ├── model.py           ← CSP + LDA classifier
│   └── evaluation.py      ← Within/cross-subject eval
└── notebooks/
    └── analysis.ipynb     ← Exploratory analysis
```

## Running the Code

```bash
# Setup
cd ~/Documents/ntab-eeg-bci
source .venv/bin/activate

# Quick test (3 subjects, ~2 min)
python scripts/quick_test.py

# Full training (10 subjects, ~10 min)
python scripts/train.py

# Predict on new EDF
python scripts/predict.py data/eegmmidb/S001/S001R04.edf
```

## Time Spent

~3 hours so far:
- 30 min: Understanding brief + dataset structure
- 60 min: Data pipeline + CSP/LDA implementation
- 45 min: Evaluation framework (within/cross-subject)
- 30 min: CLI, README, testing
- 15 min: This summary

Ready for full 10-subject run and video demo.
