"""HDF5 and metadata access for MM-IMDb."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import h5py
import numpy as np

from mmimdb.constants import GENRE_LABELS
from mmimdb.utils import resolve_path


@dataclass(frozen=True)
class DatasetPaths:
    hdf5: Path
    metadata: Path
    article_pdf: Path | None = None

    @classmethod
    def from_config(cls, config: dict) -> "DatasetPaths":
        paths = config["paths"]
        return cls(
            hdf5=resolve_path(paths["hdf5"]),
            metadata=resolve_path(paths["metadata"]),
            article_pdf=resolve_path(paths["article_pdf"]) if paths.get("article_pdf") else None,
        )


def load_metadata(metadata_path: str | Path) -> dict:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"dtype\(\): align.*")
        return np.load(resolve_path(metadata_path), allow_pickle=True).item()


def h5_summary(hdf5_path: str | Path) -> list[dict]:
    rows = []
    with h5py.File(resolve_path(hdf5_path), "r") as f:
        for key in f.keys():
            ds = f[key]
            rows.append(
                {
                    "key": key,
                    "shape": tuple(ds.shape),
                    "dtype": str(ds.dtype),
                }
            )
    return rows


def load_labels(hdf5_path: str | Path) -> np.ndarray:
    with h5py.File(resolve_path(hdf5_path), "r") as f:
        return f["genres"][:].astype(np.int8)


def load_imdb_ids(hdf5_path: str | Path) -> list[str]:
    with h5py.File(resolve_path(hdf5_path), "r") as f:
        return [x.decode("utf-8") for x in f["imdb_ids"][:]]


def read_h5_rows(dataset: h5py.Dataset, indices: Iterable[int]) -> np.ndarray:
    """Read arbitrary rows from an HDF5 dataset while preserving caller order."""
    idx = np.asarray(list(indices), dtype=np.int64)
    if idx.size == 0:
        return np.empty((0,), dtype=dataset.dtype)
    order = np.argsort(idx)
    sorted_idx = idx[order]
    data_sorted = dataset[sorted_idx]
    inverse = np.argsort(order)
    return data_sorted[inverse]


def genre_counts(y: np.ndarray) -> dict[str, int]:
    counts = y.sum(axis=0).astype(int)
    return {label: int(count) for label, count in zip(GENRE_LABELS, counts)}


def dataset_label_stats(y: np.ndarray) -> dict:
    labels_per_movie = y.sum(axis=1)
    return {
        "num_samples": int(y.shape[0]),
        "num_labels": int(y.shape[1]),
        "zero_label_rows": int((labels_per_movie == 0).sum()),
        "labels_per_movie_min": int(labels_per_movie.min()),
        "labels_per_movie_mean": float(labels_per_movie.mean()),
        "labels_per_movie_median": float(np.median(labels_per_movie)),
        "labels_per_movie_max": int(labels_per_movie.max()),
        "genre_counts": genre_counts(y),
    }
