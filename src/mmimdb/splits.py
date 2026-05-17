"""Reproducible multilabel split creation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mmimdb.constants import GENRE_LABELS
from mmimdb.data import load_labels
from mmimdb.utils import ensure_dir, save_json


def multilabel_train_val_test_split(
    y: np.ndarray,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split sizes must sum to 1.0, got {total}")

    indices = np.arange(y.shape[0])
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        first = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_val_idx, test_idx = next(first.split(indices, y))

        relative_val = val_size / (train_size + val_size)
        second = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=relative_val,
            random_state=random_state,
        )
        train_rel, val_rel = next(second.split(train_val_idx, y[train_val_idx]))
        train_idx = train_val_idx[train_rel]
        val_idx = train_val_idx[val_rel]
    except ImportError:
        from sklearn.model_selection import train_test_split

        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        relative_val = val_size / (train_size + val_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val,
            random_state=random_state,
            shuffle=True,
        )

    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def save_splits(
    hdf5_path: str | Path,
    output_dir: str | Path,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    y = load_labels(hdf5_path)
    train_idx, val_idx, test_idx = multilabel_train_val_test_split(
        y,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    out = ensure_dir(output_dir)
    np.save(out / "train_indices.npy", train_idx)
    np.save(out / "val_indices.npy", val_idx)
    np.save(out / "test_indices.npy", test_idx)

    meta = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "random_state": random_state,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "genre_labels": GENRE_LABELS,
        "train_label_counts": y[train_idx].sum(axis=0).astype(int).tolist(),
        "val_label_counts": y[val_idx].sum(axis=0).astype(int).tolist(),
        "test_label_counts": y[test_idx].sum(axis=0).astype(int).tolist(),
    }
    save_json(meta, out / "split_metadata.json")
    return meta


def load_split_indices(split_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = Path(split_dir)
    return (
        np.load(p / "train_indices.npy"),
        np.load(p / "val_indices.npy"),
        np.load(p / "test_indices.npy"),
    )


def multilabel_kfold_split(
    indices: np.ndarray,
    y: np.ndarray,
    n_splits: int = 3,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create reproducible train/validation folds from an existing index pool."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("indices must be a one-dimensional array.")
    if len(indices) < 2:
        raise ValueError("At least two samples are required for cross-validation.")

    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_splits > len(indices):
        n_splits = len(indices)

    y_subset = np.asarray(y)[indices]
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        splitter = MultilabelStratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        split_iter = splitter.split(np.zeros(len(indices)), y_subset)
    except Exception:
        from sklearn.model_selection import KFold

        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(indices)

    folds = []
    for train_rel, val_rel in split_iter:
        folds.append((np.sort(indices[train_rel]), np.sort(indices[val_rel])))
    return folds
