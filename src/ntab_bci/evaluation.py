"""Evaluation strategies that address the brief's skepticism.

The brief explicitly asks:
  - What your model can and cannot do
  - That the result is not an artifact of measurement
  - Appropriate baselines (not just chance)
  - How much is about the person vs the task
  - Whether the model learned or memorized

We implement:
  1. Within-subject (train/test split per subject, avg across subjects)
  2. Cross-subject (leave-one-subject-out)
  3. Subject-specific analysis to quantify inter-subject variability
  4. Permutation tests to establish non-chance baselines
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from tqdm import tqdm

from .model import MotorImageryClassifier


@dataclass
class EvalResult:
    """Container for evaluation metrics."""
    accuracy: float
    balanced_accuracy: float
    confusion: np.ndarray
    per_subject_acc: dict[int, float] | None = None
    train_acc: float | None = None  # For overfitting analysis


def stratified_split_indices(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split maintaining class balance within each subject.
    
    Args:
        y: labels
        groups: subject IDs
        test_size: fraction for test set
        random_state: random seed
    
    Returns:
        train_idx, test_idx
    """
    rng = np.random.RandomState(random_state)
    train_idx_list, test_idx_list = [], []
    
    for subj in np.unique(groups):
        subj_mask = groups == subj
        subj_indices = np.where(subj_mask)[0]
        y_subj = y[subj_mask]
        
        # Stratified split within this subject
        unique_labels = np.unique(y_subj)
        for label in unique_labels:
            label_indices = subj_indices[y_subj == label]
            n_test = max(1, int(len(label_indices) * test_size))
            rng.shuffle(label_indices)
            test_idx_list.extend(label_indices[:n_test])
            train_idx_list.extend(label_indices[n_test:])
    
    return np.array(train_idx_list), np.array(test_idx_list)


def within_subject_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    model_fn: Callable[[], MotorImageryClassifier],
    test_size: float = 0.2,
    random_state: int = 42,
) -> EvalResult:
    """Train/test split within each subject, average results.
    
    This tests: does the model generalize to unseen trials from the same recording session?
    """
    train_idx, test_idx = stratified_split_indices(
        y, subject_ids, test_size=test_size, random_state=random_state
    )
    
    model = model_fn()
    model.fit(X[train_idx], y[train_idx])
    
    y_pred = model.predict(X[test_idx])
    y_train_pred = model.predict(X[train_idx])
    
    per_subj: dict[int, float] = {}
    for subj in np.unique(subject_ids):
        subj_test_mask = np.isin(test_idx, np.where(subject_ids == subj)[0])
        if subj_test_mask.sum() > 0:
            per_subj[int(subj)] = accuracy_score(
                y[test_idx][subj_test_mask],
                y_pred[subj_test_mask],
            )
    
    return EvalResult(
        accuracy=accuracy_score(y[test_idx], y_pred),
        balanced_accuracy=balanced_accuracy_score(y[test_idx], y_pred),
        confusion=confusion_matrix(y[test_idx], y_pred),
        per_subject_acc=per_subj,
        train_acc=accuracy_score(y[train_idx], y_train_pred),
    )


def cross_subject_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    model_fn: Callable[[], MotorImageryClassifier],
) -> EvalResult:
    """Leave-one-subject-out cross-validation.
    
    This tests: does the model generalize to a completely new person?
    Much harder than within-subject; quantifies person-specific variance.
    """
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []
    per_subj: dict[int, float] = {}
    
    for train_idx, test_idx in tqdm(
        logo.split(X, y, groups=subject_ids),
        total=len(np.unique(subject_ids)),
        desc="LOSO-CV",
    ):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(y_pred)
        
        held_out_subj = int(subject_ids[test_idx][0])
        per_subj[held_out_subj] = accuracy_score(y[test_idx], y_pred)
    
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    return EvalResult(
        accuracy=accuracy_score(y_true_all, y_pred_all),
        balanced_accuracy=balanced_accuracy_score(y_true_all, y_pred_all),
        confusion=confusion_matrix(y_true_all, y_pred_all),
        per_subject_acc=per_subj,
    )


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    model_fn: Callable[[], MotorImageryClassifier],
    n_permutations: int = 100,
    random_state: int = 42,
) -> tuple[float, np.ndarray]:
    """Permutation test to establish significance vs. chance.
    
    Shuffles labels and retrains to build a null distribution.
    
    Returns:
        p_value: fraction of permutations that matched or exceeded true accuracy
        null_distribution: accuracies from permuted labels
    """
    # True accuracy
    true_result = within_subject_evaluation(
        X, y, subject_ids, model_fn, random_state=random_state
    )
    true_acc = true_result.accuracy
    
    # Null distribution
    rng = np.random.RandomState(random_state)
    null_accs = []
    for _ in tqdm(range(n_permutations), desc="Permutation test"):
        y_perm = rng.permutation(y)
        perm_result = within_subject_evaluation(
            X, y_perm, subject_ids, model_fn, random_state=random_state
        )
        null_accs.append(perm_result.accuracy)
    
    null_accs = np.array(null_accs)
    p_value = (null_accs >= true_acc).sum() / n_permutations
    
    return p_value, null_accs


def subject_variability_analysis(
    per_subject_acc: dict[int, float],
) -> dict[str, float]:
    """Quantify how much performance varies across subjects."""
    accs = np.array(list(per_subject_acc.values()))
    return {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "min_acc": float(np.min(accs)),
        "max_acc": float(np.max(accs)),
        "range_acc": float(np.max(accs) - np.min(accs)),
    }
