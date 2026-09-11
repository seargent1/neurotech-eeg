"""
CSP + LDA classifier for motor imagery.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mne.decoding import CSP


class MotorImageryClassifier:
    """CSP feature extraction followed by LDA."""

    def __init__(
        self,
        n_components: int = 6,
        reg: float = 0.05,
        random_state: int = 42,
    ):
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
        """X: (n_epochs, n_channels, n_times), y: binary 0/1"""
        self.pipeline.fit(X, y)
        self.is_fitted = True
        csp_transformer = self.pipeline.named_steps["CSP"]
        n_comp = csp_transformer.filters_.shape[0]
        self._feature_names = [f"CSP_{i}" for i in range(n_comp)]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 0 (left) or 1 (right)."""
        if not self.is_fitted:
            raise RuntimeError("fit() first")
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns [P(left), P(right)] per epoch."""
        if not self.is_fitted:
            raise RuntimeError("fit() first")
        return self.pipeline.predict_proba(X)

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Get CSP features without classifying."""
        if not self.is_fitted:
            raise RuntimeError("fit() first")
        return self.pipeline.named_steps["CSP"].transform(X)

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("nothing to save")
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> MotorImageryClassifier:
        model = joblib.load(path)
        if not isinstance(model, MotorImageryClassifier):
            raise TypeError(f"expected MotorImageryClassifier, got {type(model)}")
        return model

    def get_csp_patterns(self) -> np.ndarray | None:
        """Spatial patterns for plotting."""
        if not self.is_fitted:
            return None
        return self.pipeline.named_steps["CSP"].patterns_
