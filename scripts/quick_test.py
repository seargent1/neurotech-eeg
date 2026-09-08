#!/usr/bin/env python3
"""Quick training test with just 3 subjects to verify full pipeline."""

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ntab_bci.data import load_dataset
from ntab_bci.model import MotorImageryClassifier
from ntab_bci.evaluation import within_subject_evaluation


# Test config: only 3 subjects
test_config = {
    "subjects": [1, 2, 3],
    "primary_condition": "imagined",
    "data_dir": "data/eegmmidb",
    "tmin": 0.5,
    "tmax": 3.5,
    "l_freq": 8.0,
    "h_freq": 30.0,
    "n_csp_components": 6,
    "csp_reg": 0.05,
    "random_seed": 42,
}

print("Quick pipeline test with 3 subjects...")
print()

# Load data
print("Loading data...")
X, y, subject_ids = load_dataset(
    subjects=test_config["subjects"],
    condition=test_config["primary_condition"],
    data_dir=test_config["data_dir"],
    tmin=test_config["tmin"],
    tmax=test_config["tmax"],
    l_freq=test_config["l_freq"],
    h_freq=test_config["h_freq"],
)
print(f"✓ Loaded {len(y)} epochs from {len(test_config['subjects'])} subjects")
print(f"  Shape: {X.shape}")
print()

# Train and evaluate
print("Training classifier...")
def make_model():
    return MotorImageryClassifier(
        n_components=test_config["n_csp_components"],
        reg=test_config["csp_reg"],
        random_state=test_config["random_seed"],
    )

result = within_subject_evaluation(
    X, y, subject_ids, make_model, random_state=test_config["random_seed"]
)

print(f"✓ Training complete")
print()
print("Results:")
print(f"  Test Accuracy:  {result.accuracy:.3f}")
print(f"  Train Accuracy: {result.train_acc:.3f}")
print(f"  Overfitting gap: {result.train_acc - result.accuracy:.3f}")
print()
print("Confusion Matrix:")
print(result.confusion)
print()

if result.accuracy > 0.5:
    print("✓ Model performs above chance!")
    print("  Full pipeline is working correctly.")
else:
    print("⚠ Model at or below chance. This can happen with very few subjects.")
    print("  Try running with more subjects in config.yaml")
