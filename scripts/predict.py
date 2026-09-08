#!/usr/bin/env python3
"""Prediction CLI for trained motor imagery classifier.

Usage:
    python scripts/predict.py path/to/recording.edf

Returns:
    JSON with predicted labels and probabilities for each epoch.
"""

import sys
import json
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ntab_bci.model import MotorImageryClassifier
from ntab_bci.data import load_raw_edf, epochs_from_raw
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def predict_from_edf(
    edf_path: str | Path,
    model_path: str | Path,
    config: dict,
) -> dict:
    """Load EDF, extract epochs, predict labels.
    
    Returns:
        {
            "file": str,
            "n_epochs": int,
            "predictions": list[int],
            "probabilities": list[list[float]],
            "labels": {"0": "left", "1": "right"}
        }
    """
    # Load model
    model = MotorImageryClassifier.load(model_path)
    
    # Load and preprocess EDF
    raw = load_raw_edf(edf_path, preload=True)
    epochs = epochs_from_raw(
        raw,
        tmin=config["tmin"],
        tmax=config["tmax"],
        l_freq=config["l_freq"],
        h_freq=config["h_freq"],
    )
    
    # Extract data
    X = epochs.get_data(copy=True)  # (n_epochs, n_channels, n_times)
    
    # Predict
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    return {
        "file": str(edf_path),
        "n_epochs": int(len(y_pred)),
        "predictions": [int(p) for p in y_pred],
        "probabilities": [[float(p) for p in row] for row in y_proba],
        "labels": {"0": "left_fist", "1": "right_fist"},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict left/right motor imagery from EEG recording"
    )
    parser.add_argument("edf_path", type=str, help="Path to .edf file")
    parser.add_argument(
        "--model",
        type=str,
        default="models/motor_imagery_classifier.pkl",
        help="Path to trained model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: stdout)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    edf_path = Path(args.edf_path)
    if not edf_path.exists():
        print(f"Error: EDF file not found: {edf_path}", file=sys.stderr)
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        print("Hint: Run 'python scripts/train.py' first", file=sys.stderr)
        sys.exit(1)
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load config
    config = load_config(args.config)
    
    # Predict
    print(f"Loading model from {model_path}...", file=sys.stderr)
    print(f"Processing {edf_path}...", file=sys.stderr)
    
    result = predict_from_edf(edf_path, model_path, config)
    
    print(f"Found {result['n_epochs']} epochs", file=sys.stderr)
    print(f"Predictions: {sum(p == 0 for p in result['predictions'])} left, "
          f"{sum(p == 1 for p in result['predictions'])} right", file=sys.stderr)
    
    # Output
    output_json = json.dumps(result, indent=2)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Saved predictions to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
