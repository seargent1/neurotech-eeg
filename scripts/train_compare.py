#!/usr/bin/env python3
"""
Train and compare CSP+LDA vs Riemannian classifier.
Saves results to results/comparison.json and models to models/

Usage:
    python scripts/train_compare.py
    python scripts/train_compare.py --subjects 10
    python scripts/train_compare.py --subjects 30
"""

from pathlib import Path
import sys
import argparse
import yaml
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ntab_bci.data import load_dataset
from ntab_bci.model import MotorImageryClassifier
from ntab_bci.models_advanced import RiemannianClassifier, HAS_PYRIEMANN
from ntab_bci.evaluation import (
    within_subject_evaluation,
    cross_subject_evaluation,
    subject_variability_analysis,
)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_model(name, model_fn, X, y, subject_ids, random_state):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    print("  Running within-subject evaluation...")
    within_result = within_subject_evaluation(
        X, y, subject_ids, model_fn, random_state=random_state
    )
    
    print("  Running cross-subject (LOSO) evaluation...")
    cross_result = cross_subject_evaluation(X, y, subject_ids, model_fn)
    
    within_stats = subject_variability_analysis(within_result.per_subject_acc)
    cross_stats = subject_variability_analysis(cross_result.per_subject_acc)
    
    print(f"\n  Within-subject: {within_result.accuracy:.1%} (train: {within_result.train_acc:.1%})")
    print(f"  Cross-subject:  {cross_result.accuracy:.1%}")
    print(f"  Gap:            {within_result.accuracy - cross_result.accuracy:.1%}")
    
    return {
        "name": name,
        "within_subject": {
            "accuracy": float(within_result.accuracy),
            "balanced_accuracy": float(within_result.balanced_accuracy),
            "train_accuracy": float(within_result.train_acc),
            "overfitting_gap": float(within_result.train_acc - within_result.accuracy),
            "per_subject": {int(k): float(v) for k, v in within_result.per_subject_acc.items()},
            "variability": within_stats,
            "confusion": within_result.confusion.tolist(),
        },
        "cross_subject": {
            "accuracy": float(cross_result.accuracy),
            "balanced_accuracy": float(cross_result.balanced_accuracy),
            "per_subject": {int(k): float(v) for k, v in cross_result.per_subject_acc.items()},
            "variability": cross_stats,
            "confusion": cross_result.confusion.tolist(),
        },
        "generalization_gap": float(within_result.accuracy - cross_result.accuracy),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare CSP+LDA vs Riemannian")
    parser.add_argument(
        "--subjects",
        type=int,
        default=None,
        help="Number of subjects to use (1..N). Default: config.yaml n_subjects",
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    n_subjects = args.subjects if args.subjects is not None else config["n_subjects"]
    subjects = list(range(1, n_subjects + 1))
    config["subjects"] = subjects
    config["n_subjects"] = n_subjects
    
    print("=" * 70)
    print("Model Comparison: CSP+LDA vs Riemannian")
    print("=" * 70)
    print(f"Condition: {config['primary_condition']}")
    print(f"Subjects: 1..{n_subjects} ({n_subjects} total)")
    print(f"Pyriemann available: {HAS_PYRIEMANN}")
    print()
    
    print("Loading dataset...")
    X, y, subject_ids = load_dataset(
        subjects=subjects,
        condition=config["primary_condition"],
        data_dir=config["data_dir"],
        tmin=config["tmin"],
        tmax=config["tmax"],
        l_freq=config["l_freq"],
        h_freq=config["h_freq"],
    )
    print(f"Dataset: {X.shape[0]} epochs, {X.shape[1]} channels, {X.shape[2]} samples")
    print(f"Class balance: {(y==0).sum()} left, {(y==1).sum()} right")
    print()
    
    results = []
    random_state = config["random_seed"]
    
    def make_csp_lda():
        return MotorImageryClassifier(
            n_components=config["n_csp_components"],
            reg=config["csp_reg"],
            random_state=random_state,
        )
    results.append(evaluate_model("CSP + LDA", make_csp_lda, X, y, subject_ids, random_state))
    
    if HAS_PYRIEMANN:
        def make_riemann():
            return RiemannianClassifier(
                metric="riemann",
                random_state=random_state,
            )
        results.append(evaluate_model("Riemannian", make_riemann, X, y, subject_ids, random_state))
    else:
        print("\nSkipping Riemannian: pip install pyriemann")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Within':>12} {'Cross':>12} {'Gap':>12}")
    print("-" * 56)
    for r in results:
        print(f"{r['name']:<20} {r['within_subject']['accuracy']:>12.1%} {r['cross_subject']['accuracy']:>12.1%} {r['generalization_gap']:>12.1%}")
    
    best_within = max(results, key=lambda r: r["within_subject"]["accuracy"])
    best_cross = max(results, key=lambda r: r["cross_subject"]["accuracy"])
    print(f"\nBest within-subject: {best_within['name']} ({best_within['within_subject']['accuracy']:.1%})")
    print(f"Best cross-subject:  {best_cross['name']} ({best_cross['cross_subject']['accuracy']:.1%})")
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "data_shape": {
            "n_epochs": int(len(y)),
            "n_channels": int(X.shape[1]),
            "n_times": int(X.shape[2]),
            "n_subjects": n_subjects,
        },
        "models": results,
    }
    
    output_path = results_dir / "comparison.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    print("\nTraining final models on full dataset...")
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    csp_model = make_csp_lda()
    csp_model.fit(X, y)
    csp_model.save(model_dir / "csp_lda_model.pkl")
    print(f"  Saved: models/csp_lda_model.pkl")
    
    if HAS_PYRIEMANN:
        riemann_model = make_riemann()
        riemann_model.fit(X, y)
        riemann_model.save(model_dir / "riemannian_model.pkl")
        print(f"  Saved: models/riemannian_model.pkl")
    
    csp_model.save(model_dir / "motor_imagery_classifier.pkl")
    print(f"  Saved: models/motor_imagery_classifier.pkl (default for predict.py)")


if __name__ == "__main__":
    main()
