#!/usr/bin/env python3
"""
Transfer evaluation: train on one condition, test on the other.

Tests whether patterns learned from imagined movement transfer to executed (and vice versa).
"""

from pathlib import Path
import sys
import argparse
import yaml
import json
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ntab_bci.data import load_dataset
from ntab_bci.model import MotorImageryClassifier
from ntab_bci.models_advanced import RiemannianClassifier, HAS_PYRIEMANN


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def transfer_eval(model, X_test, y_test):
    """Evaluate a trained model on test data."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "confusion": confusion_matrix(y_test, y_pred).tolist(),
        "n_samples": len(y_test),
    }


def main():
    parser = argparse.ArgumentParser(description="Transfer eval: imagined <-> executed")
    parser.add_argument("--subjects", type=int, default=10, help="Number of subjects (1..N)")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    n_subjects = args.subjects
    subjects = list(range(1, n_subjects + 1))
    
    print("=" * 70)
    print("Transfer Evaluation: Imagined <-> Executed")
    print("=" * 70)
    print(f"Subjects: 1..{n_subjects}")
    print()
    
    # Load both conditions
    print("Loading imagined data...")
    X_imag, y_imag, _ = load_dataset(
        subjects=subjects,
        condition="imagined",
        data_dir=config["data_dir"],
        tmin=config["tmin"],
        tmax=config["tmax"],
        l_freq=config["l_freq"],
        h_freq=config["h_freq"],
    )
    print(f"  Imagined: {len(y_imag)} epochs")
    
    print("Loading executed data...")
    X_exec, y_exec, _ = load_dataset(
        subjects=subjects,
        condition="executed",
        data_dir=config["data_dir"],
        tmin=config["tmin"],
        tmax=config["tmax"],
        l_freq=config["l_freq"],
        h_freq=config["h_freq"],
    )
    print(f"  Executed: {len(y_exec)} epochs")
    print()
    
    random_state = config["random_seed"]
    results = {}
    
    # CSP + LDA
    print("=" * 60)
    print("  CSP + LDA")
    print("=" * 60)
    
    # Train on imagined
    csp_imag = MotorImageryClassifier(
        n_components=config["n_csp_components"],
        reg=config["csp_reg"],
        random_state=random_state,
    )
    csp_imag.fit(X_imag, y_imag)
    
    # Train on executed
    csp_exec = MotorImageryClassifier(
        n_components=config["n_csp_components"],
        reg=config["csp_reg"],
        random_state=random_state,
    )
    csp_exec.fit(X_exec, y_exec)
    
    # Same-condition baselines
    imag_on_imag = transfer_eval(csp_imag, X_imag, y_imag)
    exec_on_exec = transfer_eval(csp_exec, X_exec, y_exec)
    
    # Transfer
    imag_on_exec = transfer_eval(csp_imag, X_exec, y_exec)
    exec_on_imag = transfer_eval(csp_exec, X_imag, y_imag)
    
    print(f"\n  Train imagined -> test imagined: {imag_on_imag['accuracy']:.1%} (baseline)")
    print(f"  Train executed -> test executed: {exec_on_exec['accuracy']:.1%} (baseline)")
    print(f"  Train imagined -> test executed: {imag_on_exec['accuracy']:.1%} (transfer)")
    print(f"  Train executed -> test imagined: {exec_on_imag['accuracy']:.1%} (transfer)")
    
    results["csp_lda"] = {
        "imag_on_imag": imag_on_imag,
        "exec_on_exec": exec_on_exec,
        "imag_on_exec": imag_on_exec,
        "exec_on_imag": exec_on_imag,
    }
    
    # Riemannian
    if HAS_PYRIEMANN:
        print("\n" + "=" * 60)
        print("  Riemannian")
        print("=" * 60)
        
        riem_imag = RiemannianClassifier(metric="riemann", random_state=random_state)
        riem_imag.fit(X_imag, y_imag)
        
        riem_exec = RiemannianClassifier(metric="riemann", random_state=random_state)
        riem_exec.fit(X_exec, y_exec)
        
        imag_on_imag_r = transfer_eval(riem_imag, X_imag, y_imag)
        exec_on_exec_r = transfer_eval(riem_exec, X_exec, y_exec)
        imag_on_exec_r = transfer_eval(riem_imag, X_exec, y_exec)
        exec_on_imag_r = transfer_eval(riem_exec, X_imag, y_imag)
        
        print(f"\n  Train imagined -> test imagined: {imag_on_imag_r['accuracy']:.1%} (baseline)")
        print(f"  Train executed -> test executed: {exec_on_exec_r['accuracy']:.1%} (baseline)")
        print(f"  Train imagined -> test executed: {imag_on_exec_r['accuracy']:.1%} (transfer)")
        print(f"  Train executed -> test imagined: {exec_on_imag_r['accuracy']:.1%} (transfer)")
        
        results["riemannian"] = {
            "imag_on_imag": imag_on_imag_r,
            "exec_on_exec": exec_on_exec_r,
            "imag_on_exec": imag_on_exec_r,
            "exec_on_imag": exec_on_imag_r,
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Setup':<35} {'CSP+LDA':>12}", end="")
    if HAS_PYRIEMANN:
        print(f" {'Riemannian':>12}", end="")
    print()
    print("-" * 60)
    
    rows = [
        ("Train imag -> test imag (baseline)", "imag_on_imag"),
        ("Train exec -> test exec (baseline)", "exec_on_exec"),
        ("Train imag -> test exec (transfer)", "imag_on_exec"),
        ("Train exec -> test imag (transfer)", "exec_on_imag"),
    ]
    for label, key in rows:
        csp_acc = results["csp_lda"][key]["accuracy"]
        print(f"{label:<35} {csp_acc:>12.1%}", end="")
        if HAS_PYRIEMANN:
            riem_acc = results["riemannian"][key]["accuracy"]
            print(f" {riem_acc:>12.1%}", end="")
        print()
    
    print(f"\n{'Chance':<35} {'50.0%':>12}")
    
    # Save
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_subjects": n_subjects,
        "n_imagined": len(y_imag),
        "n_executed": len(y_exec),
        "models": results,
    }
    
    output_path = results_dir / "transfer_eval.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
