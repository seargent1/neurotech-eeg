"""
Additional classifiers: Filter Bank CSP and Riemannian.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False


class FilterBankCSP(BaseEstimator, ClassifierMixin):
    """
    Run CSP on multiple frequency bands, concatenate features.
    Bands: 4-8, 8-12, 12-20, 20-30 Hz by default.
    """
    
    def __init__(
        self,
        bands: list[tuple[float, float]] | None = None,
        n_components_per_band: int = 4,
        reg: float = 0.05,
        sfreq: float = 160.0,
        random_state: int = 42,
    ):
        self.bands = bands if bands is not None else [(4, 8), (8, 12), (12, 20), (20, 30)]
        self.n_components_per_band = n_components_per_band
        self.reg = reg
        self.sfreq = sfreq
        self.random_state = random_state
        self.band_csps_ = {}
        self.classifier_ = None
        self.scaler_ = None
        self.is_fitted = False
    
    def _extract_band_features(self, X: np.ndarray, fit: bool = False, y: np.ndarray = None) -> np.ndarray:
        all_features = []
        
        for (l_freq, h_freq) in self.bands:
            band_key = f"{l_freq}_{h_freq}"
            
            nyq = self.sfreq / 2
            low = l_freq / nyq
            high = min(h_freq / nyq, 0.99)
            b, a = butter(5, [low, high], btype='band')
            
            X_filt = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    X_filt[i, j, :] = filtfilt(b, a, X[i, j, :])
            
            if fit:
                csp = CSP(
                    n_components=self.n_components_per_band,
                    reg=self.reg,
                    log=True,
                    cov_est="concat",
                    norm_trace=False,
                )
                csp.fit(X_filt, y)
                self.band_csps_[band_key] = csp
            
            features = self.band_csps_[band_key].transform(X_filt)
            all_features.append(features)
        
        return np.hstack(all_features)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        features = self._extract_band_features(X, fit=True, y=y)
        
        self.scaler_ = StandardScaler()
        features_scaled = self.scaler_.fit_transform(features)
        
        self.classifier_ = LogisticRegression(
            C=1.0, 
            solver="lbfgs", 
            max_iter=1000,
            random_state=self.random_state
        )
        self.classifier_.fit(features_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        features = self._extract_band_features(X, fit=False)
        features_scaled = self.scaler_.transform(features)
        return self.classifier_.predict(features_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        features = self._extract_band_features(X, fit=False)
        features_scaled = self.scaler_.transform(features)
        return self.classifier_.predict_proba(features_scaled)
    
    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("nothing to save")
        joblib.dump(self, path)
    
    @staticmethod
    def load(path: str | Path) -> "FilterBankCSP":
        model = joblib.load(path)
        if not isinstance(model, FilterBankCSP):
            raise TypeError(f"expected FilterBankCSP, got {type(model)}")
        return model


class RiemannianClassifier(BaseEstimator, ClassifierMixin):
    """
    Covariance -> tangent space -> logistic regression.
    Needs pyriemann: pip install pyriemann
    """
    
    def __init__(
        self,
        metric: Literal["riemann", "logeuclid", "euclid"] = "riemann",
        reg: float = 1e-6,
        random_state: int = 42,
    ):
        if not HAS_PYRIEMANN:
            raise ImportError("pip install pyriemann")
        
        self.metric = metric
        self.reg = reg
        self.random_state = random_state
        self.pipeline = None
        self.is_fitted = False
    
    def _build_pipeline(self):
        self.pipeline = Pipeline([
            ("covariances", Covariances(estimator="lwf")),
            ("tangent_space", TangentSpace(metric=self.metric)),
            ("classifier", LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_state,
            )),
        ])
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self._build_pipeline()
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)
    
    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("nothing to save")
        joblib.dump(self, path)
    
    @staticmethod
    def load(path: str | Path) -> "RiemannianClassifier":
        model = joblib.load(path)
        if not isinstance(model, RiemannianClassifier):
            raise TypeError(f"expected RiemannianClassifier, got {type(model)}")
        return model


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Soft voting over CSP, Filter Bank CSP, and Riemannian."""
    
    def __init__(
        self,
        use_riemannian: bool = True,
        n_csp_components: int = 6,
        reg: float = 0.05,
        sfreq: float = 160.0,
        random_state: int = 42,
    ):
        self.use_riemannian = use_riemannian and HAS_PYRIEMANN
        self.n_csp_components = n_csp_components
        self.reg = reg
        self.sfreq = sfreq
        self.random_state = random_state
        self.models = {}
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        from .model import MotorImageryClassifier
        
        self.models["csp_lda"] = MotorImageryClassifier(
            n_components=self.n_csp_components,
            reg=self.reg,
            random_state=self.random_state,
        )
        self.models["csp_lda"].fit(X, y)
        
        self.models["fbcsp"] = FilterBankCSP(
            n_components_per_band=4,
            reg=self.reg,
            sfreq=self.sfreq,
            random_state=self.random_state,
        )
        self.models["fbcsp"].fit(X, y)
        
        if self.use_riemannian:
            self.models["riemann"] = RiemannianClassifier(
                metric="riemann",
                random_state=self.random_state,
            )
            self.models["riemann"].fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = []
        for name, model in self.models.items():
            probas.append(model.predict_proba(X))
        return np.mean(probas, axis=0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)
    
    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("nothing to save")
        joblib.dump(self, path)
    
    @staticmethod
    def load(path: str | Path) -> "EnsembleClassifier":
        model = joblib.load(path)
        if not isinstance(model, EnsembleClassifier):
            raise TypeError(f"expected EnsembleClassifier, got {type(model)}")
        return model
