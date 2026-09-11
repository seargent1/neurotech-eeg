# Experiment Results

## Dataset

- PhysioNet EEGMMIDB, subjects 1-30
- Imagined left/right fist (runs 4, 8, 12)
- 1350 epochs total, ~45 per subject
- 64 channels, 160 Hz
- 683 left, 667 right

## Preprocessing

| Step | Setting |
|------|---------|
| Band-pass | 8-30 Hz |
| Reference | Average |
| Epoch window | 0.5-3.5s after cue |
| Detrend | Linear |

No ICA, no bad epoch rejection.

---

## CSP + LDA

6 components, reg=0.05, shrinkage LDA.

| Metric | Value |
|--------|-------|
| Within-subject | 52.5% |
| Train accuracy | 56.4% |
| Cross-subject | 54.7% |
| Overfitting gap | 3.9% |

Per-subject cross-subject accuracy ranged from 44% to 78%.

---

## Riemannian

Ledoit-Wolf covariance, tangent space, logistic regression.

| Metric | Value |
|--------|-------|
| Within-subject | 65.8% |
| Train accuracy | 99.8% |
| Cross-subject | 56.3% |
| Overfitting gap | 34.0% |

Per-subject cross-subject accuracy ranged from 40% to 73%.

---

## Comparison (30 subjects)

| Model | Within | Cross | Gap |
|-------|--------|-------|-----|
| CSP + LDA | 52.5% | 54.7% | -2.2% |
| Riemannian | 65.8% | 56.3% | 9.5% |

Best cross-subject: Riemannian (56.3%)

---

## Previous run (10 subjects)

| Model | Within | Cross | Gap |
|-------|--------|-------|-----|
| CSP + LDA | 63.7% | 55.3% | 8.4% |
| Riemannian | 76.2% | 56.2% | 20.0% |

---

## What worked

1. **CSP + LDA**: minimal overfitting (3.9% gap), stable
2. **Riemannian**: slightly better cross-subject (56.3% vs 54.7%)
3. **Regularization**: CSP reg=0.05 and LDA shrinkage kept train/test gap small

## What didn't work

1. **Filter Bank CSP**: too many features (16) for small sample, hurt accuracy
2. **Ensemble**: averaging weak learners made it worse
3. **More subjects didn't boost accuracy**: went from 10 to 30 subjects, cross-subject stayed around 55%

## Key numbers

| | Value |
|---|---|
| Chance | 50% |
| Best cross-subject | 56.3% (Riemannian) |
| CSP cross-subject | 54.7% |
| Subject variability | 8-10% std |
| Subject range | 40-78% |
| CSP overfitting | 3.9% |
| Riemannian overfitting | 34.0% |

## Takeaways

- Cross-subject accuracy ~55%, about 5 points above chance
- Riemannian overfits heavily within-subject but still edges out CSP on new people
- Person-to-person variance is huge (some at 40%, some at 78%)
- More subjects gave tighter estimates but didn't improve accuracy
- Signal is weak; bottleneck is the task, not the model
