# EEG Motor Imagery Classification

Left vs right fist motor imagery classification using the PhysioNet EEGMMIDB dataset.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+. Dependencies are pinned in `requirements.txt` to match the shipped model.

## Usage

### Predict on new data

```bash
python scripts/predict.py path/to/recording.edf
```

Uses the shipped CSP+LDA weights in `models/motor_imagery_classifier.pkl`. Works on any EEGMMIDB-style EDF with T1/T2 left/right cues (including subjects not used in training).

```bash
python scripts/predict.py path/to/recording.edf --output predictions.json
```

Output:
```json
{
  "file": "path/to/recording.edf",
  "n_epochs": 42,
  "predictions": [0, 1, 0, 1],
  "probabilities": [[0.72, 0.28], [0.31, 0.69]],
  "labels": {"0": "left_fist", "1": "right_fist"}
}
```

### Train and compare models

```bash
python scripts/train_compare.py              # uses n_subjects from config.yaml
python scripts/train_compare.py --subjects 10
python scripts/train_compare.py --subjects 30
```

Downloads EEG data for subjects 1..N, trains CSP+LDA and Riemannian, runs within/cross-subject eval, saves to `results/comparison.json` and `models/`.

### Generate figures

```bash
python scripts/visualize.py
```

Creates charts in `figures/`.

## Configuration

Edit `config.yaml`:

| Setting | Default | Notes |
|---------|---------|-------|
| n_subjects | 30 | Use subjects 1..N (override with `--subjects`) |
| primary_condition | imagined | imagined, executed, or both |
| l_freq / h_freq | 8-30 Hz | Band-pass filter |
| tmin / tmax | 0.5-3.5 s | Epoch window |
| n_csp_components | 6 | CSP spatial filters |

## Project layout

```
.
├── config.yaml
├── requirements.txt
├── RESULTS.md            # experiment log
├── scripts/
│   ├── train_compare.py  # training and evaluation
│   ├── predict.py        # inference CLI
│   └── visualize.py      # generate figures
├── src/ntab_bci/
│   ├── data.py           # download and preprocessing
│   ├── model.py          # CSP + LDA
│   ├── models_advanced.py # Riemannian
│   └── evaluation.py     # within/cross-subject eval
├── models/               # saved models
├── results/              # JSON results
├── figures/              # charts
└── data/                 # downloaded EEG
```

## Data

EEGMMIDB has 109 subjects, 14 runs each. We use the left/right fist runs:

- Imagined: runs 4, 8, 12
- Executed: runs 3, 7, 11

T1 = left fist, T2 = right fist.

MNE downloads only the runs we need, not the full 1.9 GB archive.

## Preprocessing

1. Band-pass 8-30 Hz (mu and beta rhythms)
2. Average reference
3. Epoch 0.5-3.5 s after cue
4. Linear detrend

No ICA or artifact rejection.

## Models

**CSP + LDA**: 6 spatial filters with regularization, shrinkage LDA. Simple baseline.

**Riemannian**: Covariance matrices projected to tangent space, logistic regression. Sometimes better cross-subject.

See `RESULTS.md` for comparison.

## Evaluation

- **Within-subject**: 80/20 split per person, pooled model. Tests generalization to new trials.
- **Cross-subject (LOSO)**: Train on N-1, test on held-out person. Tests generalization to new people.
- **Per-subject tracking**: Identify which subjects work/fail.
- **Train/test gap**: Detect overfitting.

## Limitations

- ~45 epochs per subject
- No artifact rejection
- Hyperparameters not tuned
- 30 subjects, not all 109

## AI tools

Parts of this project were developed with AI coding assistance (Cursor). Design choices, experiment decisions, evaluation, and interpretation of results were directed and reviewed by me. Code and writeups were checked and edited before submission.

## Citations

Schalk et al. (2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE TBME 51(6):1034-1043.

Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23):e215-e220.

## License

MIT
