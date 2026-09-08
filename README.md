# NT@B FA26 Software Division — EEG Motor Imagery Classification

Left vs. right fist motor imagery classification on the [PhysioNet EEGMMIDB dataset](https://physionet.org/content/eegmmidb/1.0.0/).

**Note:** This repository is a recruitment project submission. See [Project Brief](#project-brief) below.

---

## Quick Start

### 1. Installation

```bash
# Clone and enter directory
cd ~/Documents/ntab-eeg-bci

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.9+
- MNE-Python (EEG processing + PhysioNet data fetcher)
- scikit-learn (CSP + LDA)
- NumPy, SciPy, joblib, matplotlib, tqdm, PyYAML

### 2. Train and Evaluate

```bash
python scripts/train.py
```

This will:
- Download only the required EEG runs (~50-100 MB, not the full 1.9 GB dataset)
- Preprocess and epoch the data
- Train a CSP+LDA classifier
- Run **within-subject** and **cross-subject** evaluations
- Save the trained model to `models/motor_imagery_classifier.pkl`

**Runtime:** ~5-15 minutes depending on download speed and CPU.

### 3. Predict on New Data

```bash
# Predict on a single EDF file
python scripts/predict.py data/eegmmidb/S001/S001R04.edf

# Save predictions to JSON
python scripts/predict.py data/eegmmidb/S001/S001R04.edf --output predictions.json
```

**Output format:**
```json
{
  "file": "data/eegmmidb/S001/S001R04.edf",
  "n_epochs": 42,
  "predictions": [0, 1, 0, 1, ...],
  "probabilities": [[0.72, 0.28], [0.31, 0.69], ...],
  "labels": {"0": "left_fist", "1": "right_fist"}
}
```

---

## Configuration

Edit `config.yaml` to customize:

- **Subjects:** Which subjects to download (default: 1-10)
- **Condition:** `imagined` (motor imagery), `executed` (real movement), or `both`
- **Frequency band:** Default 8-30 Hz (mu/beta sensorimotor rhythms)
- **Epoch window:** Default 0.5-3.5s post-cue
- **CSP components:** Default 6

---

## Project Structure

```
ntab-eeg-bci/
├── config.yaml              # Configuration
├── requirements.txt         # Python dependencies
├── scripts/
│   ├── train.py            # Training + evaluation
│   └── predict.py          # Prediction CLI
├── src/ntab_bci/
│   ├── data.py             # PhysioNet download + preprocessing
│   ├── model.py            # CSP + LDA classifier
│   └── evaluation.py       # Within/cross-subject evaluation
├── models/                  # Saved models (generated)
├── data/eegmmidb/          # Downloaded EEG data (generated)
└── notebooks/               # Analysis notebooks (TODO)
```

---

## Why MNE Instead of the 1.9 GB ZIP?

The PhysioNet EEGMMIDB dataset contains 109 subjects × 14 runs = 1,526 recordings.

**We don't need all of them.** For left/right fist classification:
- Relevant runs: 4, 8, 12 (imagined) or 3, 7, 11 (executed)
- 10 subjects × 3 runs ≈ **50-100 MB**

MNE's `eegbci.load_data()` fetches only the (subject, run) pairs you request, caching them locally. Re-running is a no-op (HTTP resume).

---

## Evaluation Strategy

The brief asks: *"What else could explain this number?"*

We address this with:

### 1. Within-Subject Evaluation
- **Method:** Stratified train/test split within each subject, then average
- **Tests:** Does the model generalize to unseen trials from the same recording session?
- **Expected:** Moderate accuracy (60-75%)

### 2. Cross-Subject Evaluation (Leave-One-Subject-Out)
- **Method:** Train on N-1 subjects, test on the held-out subject
- **Tests:** Does the model generalize to a completely new person?
- **Expected:** Lower accuracy due to inter-subject variability
- **Interpretation:** The gap quantifies how much performance is person-specific

### 3. Subject Variability Analysis
- **Method:** Report per-subject accuracy, mean, std, range
- **Tests:** How much does performance vary by individual?
- **Interpretation:** Large variance → the "right person" matters more than the model

### 4. Overfitting Detection
- **Method:** Compare train vs. test accuracy within-subject
- **Tests:** Is the model memorizing or learning?
- **Interpretation:** Large gap → overfitting; small gap → regularization working

### 5. Permutation Tests (TODO)
- **Method:** Shuffle labels, retrain, compare to true accuracy
- **Tests:** Is performance better than chance?
- **Interpretation:** Establishes statistical significance

---

## Classifier Choice: CSP + LDA

**Why CSP?**
- Common Spatial Patterns is the standard BCI approach for motor imagery
- Extracts spatial filters that maximize class-separable variance
- Works well with sensorimotor rhythms (8-30 Hz mu/beta bands)

**Why LDA?**
- Linear Discriminant Analysis is robust with small sample sizes
- Automatic shrinkage regularization prevents overfitting
- Interpretable: decision boundary is a hyperplane in CSP feature space

**Trade-offs:**
- **Simple** but establishes a reasonable baseline
- **Not state-of-the-art** (deep learning can do better with more data)
- **Interpretable** (spatial patterns can be visualized)

---

## Known Limitations

1. **Small sample size per subject:** ~40 epochs per condition → high variance
2. **Inter-subject variability:** EEG varies wildly across people (skull thickness, cortical folding, attention)
3. **Task confounds:** Executed vs. imagined movement have different SNR
4. **Session effects:** Electrode impedance, fatigue, motivation drift over time
5. **No artifact rejection:** Eye blinks, muscle artifacts, and line noise are still present

---

## Next Steps (If I Had More Time)

1. **Artifact rejection:** ICA for eye blinks, autoreject for noisy epochs
2. **Subject-specific calibration:** Fine-tune on a few trials from the test subject
3. **Temporal features:** Time-frequency decomposition (wavelet CSP)
4. **Deep learning:** EEGNet or Shallow ConvNet for end-to-end learning
5. **More subjects:** Full 109-subject dataset for better cross-subject generalization
6. **Hyperparameter tuning:** Grid search for CSP components, frequency bands, regularization

---

## Project Brief

**Time budget:** 5-10 hours  
**Task:** Build a classifier that distinguishes left fist from right fist (imagined or executed)  
**Dataset:** PhysioNet EEGMMIDB (64-channel EEG, 160 Hz)  
**Deliverables:**
1. Code (this repo)
2. Prediction script (`scripts/predict.py`)
3. Demo video (2-3 minutes)

**Evaluation focus:**
- Explain data processing pipeline
- Show what the model can and cannot do
- Prove results are not measurement artifacts
- Quantify person-specific vs. task-specific variance
- Detect overfitting

---

## Citations

**Dataset:**
> Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004).  
> BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.  
> *IEEE Transactions on Biomedical Engineering*, 51(6):1034-1043.  
> DOI: [10.1109/TBME.2004.827072](https://doi.org/10.1109/TBME.2004.827072)

**PhysioNet:**
> Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation* 101(23):e215-e220.

---

## License

MIT (for this submission code only; dataset has its own license)

---

## Author

Raghav — NT@B FA26 Software Division Recruitment Submission
