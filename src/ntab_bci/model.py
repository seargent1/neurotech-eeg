"""CSP + LDA classifier for left/right motor imagery.

Why CSP?
  Common Spatial Patterns extracts spatial filters that maximize variance ratio
  between the two classes. Standard BCI approach for sensorimotor rhythms.
  Simple, interpretable, and establishes a reasonable baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from mne.decoding import CSP


class MotorImageryClassifier:
    """CSP feature extraction + LDA classifier.
    
    Attributes:
        pipeline: sklearn Pipeline with CSP transformer and LDA classifier
        is_fitted: whether the model has been trained
    """

    def __init__(
        self,
        n_components: int = 6,
        reg: float = 0.05,
        random_state: int = 42,
    ):
        """
        Args:
            n_components: Number of CSP components to extract (must be even)
            reg: CSP regularization to prevent overfitting on small samples
            random_state: Random seed for reproducibility
        """
        csp = CSP(
            n_components=n_components,
            reg=reg,
            log=True,
            cov_est="concat",
            norm_trace=False,
        )
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        
        self.pipeline = Pipeline([("CSP", csp), ("LDA", lda)])
        self.random_state = random_state
        self.is_fitted = False
        self._feature_names: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> MotorImageryClassifier:
        """Train on epoched EEG data.
        
        Args:
            X: shape (n_epochs, n_channels, n_times)
            y: shape (n_epochs,) with binary labels 0/1
        """
        self.pipeline.fit(X, y)
        self.is_fitted = True
        # Store CSP pattern info for interpretation
        csp_transformer = self.pipeline.named_steps["CSP"]
        n_comp = csp_transformer.filters_.shape[0]
        self._feature_names = [f"CSP_{i}" for i in range(n_comp)]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (0=left, 1=right).
        
        Args:
            X: shape (n_epochs, n_channels, n_times)
        
        Returns:
            predictions: shape (n_epochs,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict()")
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Returns:
            probabilities: shape (n_epochs, 2) for [P(left), P(right)]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba()")
        return self.pipeline.predict_proba(X)

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Extract CSP features without classification.
        
        Returns:
            features: shape (n_epochs, n_components)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform_features()")
        return self.pipeline.named_steps["CSP"].transform(X)

    def save(self, path: str | Path) -> None:
        """Serialize model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> MotorImageryClassifier:
        """Load serialized model."""
        model = joblib.load(path)
        if not isinstance(model, MotorImageryClassifier):
            raise TypeError(f"Expected MotorImageryClassifier, got {type(model)}")
        return model

    def get_csp_patterns(self) -> np.ndarray | None:
        """Get CSP spatial patterns for visualization.
        
        Returns:
            patterns: shape (n_components, n_channels) or None if not fitted
        """
        if not self.is_fitted:
            return None
        return self.pipeline.named_steps["CSP"].patterns_
