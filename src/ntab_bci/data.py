"""
Download and preprocess PhysioNet EEGMMIDB data.
Uses MNE to fetch only the runs we need (~50-100 MB) instead of the full 1.9 GB zip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Sequence

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import BaseRaw, read_raw_edf

Condition = Literal["imagined", "executed", "both"]

# run numbers for left/right fist task
EXECUTED_LR = (3, 7, 11)
IMAGINED_LR = (4, 8, 12)
BASELINE = (1, 2)
EXECUTED_BOTH = (5, 9, 13)
IMAGINED_BOTH = (6, 10, 14)

# T1 = left fist, T2 = right fist (for LR runs only)
EVENT_ID = {"left": 1, "right": 2}


def runs_for_condition(condition: Condition) -> list[int]:
    if condition == "imagined":
        return list(IMAGINED_LR)
    if condition == "executed":
        return list(EXECUTED_LR)
    if condition == "both":
        return list(EXECUTED_LR) + list(IMAGINED_LR)
    raise ValueError(f"unknown condition: {condition}")


def ensure_data_path(data_dir: str | Path) -> Path:
    path = Path(data_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_subject_runs(
    subject: int,
    runs: Sequence[int],
    data_dir: str | Path,
) -> list[Path]:
    """Fetch EDF files for one subject. Cached after first download."""
    path = ensure_data_path(data_dir)
    files = eegbci.load_data(subject, runs, path=path, update_path=False)
    return [Path(f) for f in files]


def download_subjects(
    subjects: Iterable[int],
    runs: Sequence[int],
    data_dir: str | Path,
) -> dict[int, list[Path]]:
    out: dict[int, list[Path]] = {}
    for subj in subjects:
        out[int(subj)] = download_subject_runs(int(subj), runs, data_dir)
    return out


def standardize_raw(raw: BaseRaw) -> BaseRaw:
    """Fix channel names and add 10-10 montage."""
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    return raw


def load_raw_edf(edf_path: str | Path, preload: bool = True) -> BaseRaw:
    raw = read_raw_edf(str(edf_path), preload=preload, verbose="error")
    return standardize_raw(raw)


def preprocess_raw(
    raw: BaseRaw,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
) -> BaseRaw:
    """Band-pass filter and average reference."""
    raw = raw.copy()
    raw.pick(picks="eeg")
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose="error")
    raw.set_eeg_reference("average", projection=False, verbose="error")
    return raw


def events_from_annotations(raw: BaseRaw) -> tuple[np.ndarray, dict[str, int]]:
    """Pull T1/T2 cues and map to left/right."""
    events, _ = mne.events_from_annotations(
        raw,
        event_id={"T1": EVENT_ID["left"], "T2": EVENT_ID["right"]},
        verbose="error",
    )
    return events, EVENT_ID


def epochs_from_raw(
    raw: BaseRaw,
    tmin: float = 0.5,
    tmax: float = 3.5,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
) -> mne.Epochs:
    raw = preprocess_raw(raw, l_freq=l_freq, h_freq=h_freq)
    events, event_id = events_from_annotations(raw)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        detrend=1,
        verbose="error",
    )
    return epochs


def load_subject_epochs(
    subject: int,
    runs: Sequence[int],
    data_dir: str | Path,
    tmin: float = 0.5,
    tmax: float = 3.5,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
) -> mne.Epochs:
    paths = download_subject_runs(subject, runs, data_dir)
    epoch_list = []
    for p in paths:
        raw = load_raw_edf(p)
        epoch_list.append(
            epochs_from_raw(raw, tmin=tmin, tmax=tmax, l_freq=l_freq, h_freq=h_freq)
        )
    return mne.concatenate_epochs(epoch_list, verbose="error")


def load_dataset(
    subjects: Sequence[int],
    condition: Condition,
    data_dir: str | Path,
    tmin: float = 0.5,
    tmax: float = 3.5,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        X: (n_epochs, n_channels, n_times)
        y: 0 = left, 1 = right
        subject_ids: which person each epoch came from
    """
    runs = runs_for_condition(condition)
    Xs, ys, sids = [], [], []
    for subj in subjects:
        epochs = load_subject_epochs(
            subj, runs, data_dir, tmin=tmin, tmax=tmax, l_freq=l_freq, h_freq=h_freq
        )
        X = epochs.get_data(copy=True)
        y = (epochs.events[:, 2] == EVENT_ID["right"]).astype(int)
        Xs.append(X)
        ys.append(y)
        sids.append(np.full(len(y), subj, dtype=int))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(sids)
