"""
Evaluation functions for motor imagery classification.

Two main tests:
  - within_subject: train/test split per person, pooled model
  - cross_subject: leave-one-subject-out (LOSO)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm

from .model import MotorImageryClassifier


@dataclass
class EvalResult:
    accuracy: float
    balanced_accuracy: float
    confusion: np.ndarray
    per_subject_acc: dict[int, float] | None = None
    train_acc: float | None = None


def stratified_split_indices(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """80/20 split within each subject, keeps class balance."""
    rng = np.random.RandomState(random_state)
    train_idx_list, test_idx_list = [], []
    
    for subj in np.unique(groups):
        subj_mask = groups == subj
        subj_indices = np.where(subj_mask)[0]
        y_subj = y[subj_mask]
        
        for label in np.unique(y_subj):
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
    """
    Train on 80% of each person's trials, test on 20%.
    Tests generalization to new trials from known people.
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
    """
    Leave-one-subject-out: train on N-1 people, test on held-out person.
    Tests generalization to completely new people.
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
    """
    Shuffle labels, retrain, see how often shuffled beats real.
    Returns p-value and null distribution.
    """
    true_result = within_subject_evaluation(
        X, y, subject_ids, model_fn, random_state=random_state
    )
    true_acc = true_result.accuracy
    
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
    """Mean, std, min, max across subjects."""
    accs = np.array(list(per_subject_acc.values()))
    return {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "min_acc": float(np.min(accs)),
        "max_acc": float(np.max(accs)),
        "range_acc": float(np.max(accs) - np.min(accs)),
    }
