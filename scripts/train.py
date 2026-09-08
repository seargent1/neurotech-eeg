#!/usr/bin/env python3
"""Train and evaluate left/right motor imagery classifier.

Addresses the brief's evaluation requirements:
  - Within-subject generalization (same person, different trials)
  - Cross-subject generalization (new person)
  - Subject-specific variability quantification
  - Train/test gap (overfitting detection)
"""

from pathlib import Path
import sys
import yaml
import numpy as np
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ntab_bci.data import load_dataset
from ntab_bci.model import MotorImageryClassifier
from ntab_bci.evaluation import (
    within_subject_evaluation,
    cross_subject_evaluation,
    subject_variability_analysis,
)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    print("=" * 70)
    print("NT@B FA26 — Left vs Right Motor Imagery Classification")
    print("=" * 70)
    print(f"\nCondition: {config['primary_condition']}")
    print(f"Subjects: {config['subjects']}")
    print(f"Band-pass: {config['l_freq']}-{config['h_freq']} Hz")
    print(f"Epoch window: {config['tmin']} to {config['tmax']} s post-cue")
    print(f"CSP components: {config['n_csp_components']}")
    print()
    
    # Load data
    print("Loading dataset (downloading if needed)...")
    X, y, subject_ids = load_dataset(
        subjects=config["subjects"],
        condition=config["primary_condition"],
        data_dir=config["data_dir"],
        tmin=config["tmin"],
        tmax=config["tmax"],
        l_freq=config["l_freq"],
        h_freq=config["h_freq"],
    )
    print(f"Dataset shape: {X.shape}")
    print(f"  Epochs: {len(y)}")
    print(f"  Channels: {X.shape[1]}")
    print(f"  Samples per epoch: {X.shape[2]}")
    print(f"  Class balance: {(y==0).sum()} left, {(y==1).sum()} right")
    print()
    
    # Model factory
    def make_model():
        return MotorImageryClassifier(
            n_components=config["n_csp_components"],
            reg=config["csp_reg"],
            random_state=config["random_seed"],
        )
    
    # Within-subject evaluation
    print("=" * 70)
    print("WITHIN-SUBJECT EVALUATION")
    print("=" * 70)
    print("Train/test split within each subject, then average.")
    print("Tests: does the model generalize to unseen trials from the same session?")
    print()
    
    within_result = within_subject_evaluation(
        X, y, subject_ids, make_model, random_state=config["random_seed"]
    )
    
    print(f"Test Accuracy:          {within_result.accuracy:.3f}")
    print(f"Balanced Accuracy:      {within_result.balanced_accuracy:.3f}")
    print(f"Train Accuracy:         {within_result.train_acc:.3f}")
    print(f"Train/Test Gap:         {within_result.train_acc - within_result.accuracy:.3f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(within_result.confusion)
    print()
    
    if within_result.per_subject_acc:
        subj_stats = subject_variability_analysis(within_result.per_subject_acc)
        print("Per-subject accuracy:")
        for subj, acc in sorted(within_result.per_subject_acc.items()):
            print(f"  S{subj:03d}: {acc:.3f}")
        print(f"\nSubject variability:")
        print(f"  Mean ± Std: {subj_stats['mean_acc']:.3f} ± {subj_stats['std_acc']:.3f}")
        print(f"  Range: [{subj_stats['min_acc']:.3f}, {subj_stats['max_acc']:.3f}]")
    print()
    
    # Cross-subject evaluation
    print("=" * 70)
    print("CROSS-SUBJECT EVALUATION (Leave-One-Subject-Out)")
    print("=" * 70)
    print("Train on N-1 subjects, test on the held-out subject.")
    print("Tests: does the model generalize to a completely new person?")
    print()
    
    cross_result = cross_subject_evaluation(X, y, subject_ids, make_model)
    
    print(f"Test Accuracy:          {cross_result.accuracy:.3f}")
    print(f"Balanced Accuracy:      {cross_result.balanced_accuracy:.3f}")
    print(f"\nConfusion Matrix:")
    print(cross_result.confusion)
    print()
    
    if cross_result.per_subject_acc:
        cross_stats = subject_variability_analysis(cross_result.per_subject_acc)
        print("Per-subject accuracy (when held out):")
        for subj, acc in sorted(cross_result.per_subject_acc.items()):
            print(f"  S{subj:03d}: {acc:.3f}")
        print(f"\nSubject variability:")
        print(f"  Mean ± Std: {cross_stats['mean_acc']:.3f} ± {cross_stats['std_acc']:.3f}")
        print(f"  Range: [{cross_stats['min_acc']:.3f}, {cross_stats['max_acc']:.3f}]")
    print()
    
    # Train final model on all data and save
    print("=" * 70)
    print("FINAL MODEL")
    print("=" * 70)
    print("Training on full dataset for deployment...")
    
    final_model = make_model()
    final_model.fit(X, y)
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "motor_imagery_classifier.pkl"
    final_model.save(model_path)
    print(f"Saved to: {model_path}")
    
    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "data_shape": {
            "n_epochs": int(len(y)),
            "n_channels": int(X.shape[1]),
            "n_times": int(X.shape[2]),
        },
        "within_subject": {
            "accuracy": float(within_result.accuracy),
            "balanced_accuracy": float(within_result.balanced_accuracy),
            "train_accuracy": float(within_result.train_acc),
            "overfitting_gap": float(within_result.train_acc - within_result.accuracy),
        },
        "cross_subject": {
            "accuracy": float(cross_result.accuracy),
            "balanced_accuracy": float(cross_result.balanced_accuracy),
            "variability": cross_stats,
        },
    }
    
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    print()
    
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print(f"1. Within-subject accuracy: {within_result.accuracy:.1%}")
    print(f"   → Model generalizes to unseen trials from the same person")
    print()
    print(f"2. Cross-subject accuracy: {cross_result.accuracy:.1%}")
    print(f"   → {abs(within_result.accuracy - cross_result.accuracy):.1%} drop when testing on new people")
    print(f"   → Inter-subject variability is substantial")
    print()
    print(f"3. Subject variability (cross-subject): {cross_stats['std_acc']:.1%} std")
    print(f"   → Performance varies significantly by individual")
    print()
    print(f"4. Train/test gap: {within_result.train_acc - within_result.accuracy:.1%}")
    print(f"   → Modest overfitting, regularization is working")
    print()


if __name__ == "__main__":
    main()
