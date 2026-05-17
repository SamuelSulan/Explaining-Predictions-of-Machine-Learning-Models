"""Text reconstruction and tensor preparation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mmimdb.constants import PAD_TOKEN, UNK_TOKEN


def sequence_to_tokens(sequence: Sequence[int], ix_to_word: dict[int, str]) -> list[str]:
    return [ix_to_word.get(int(ix), UNK_TOKEN) for ix in sequence]


def sequence_to_text(sequence: Sequence[int], ix_to_word: dict[int, str]) -> str:
    return " ".join(sequence_to_tokens(sequence, ix_to_word))


def clean_token_for_display(token: str) -> str:
    if token == "N":
        return "\n"
    return token


def prepare_token_ids(
    sequence: Sequence[int],
    vocab_size: int,
    max_length: int,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a raw HDF5 sequence to padded token IDs and mask.

    Original token IDs are preserved when they are within the vocabulary. Values
    outside the vocabulary are mapped to `_UNK_` index 0. Padding uses a new ID.
    """
    arr = np.asarray(sequence, dtype=np.int64)
    arr = np.where((arr >= 0) & (arr < vocab_size), arr, 0)
    arr = arr[:max_length]
    mask = np.ones(len(arr), dtype=np.float32)

    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, (0, pad_width), constant_values=pad_id)
        mask = np.pad(mask, (0, pad_width), constant_values=0.0)

    return arr.astype(np.int64), mask.astype(np.float32)


def build_embedding_matrix(metadata: dict, seed: int = 42) -> np.ndarray:
    """Create an embedding matrix with an added padding row.

    `metadata["lookup"]` covers the Word2Vec-intersected vocabulary. The full
    `word_to_ix` may be larger, so missing rows are initialized with small random
    values. The padding row is all zeros.
    """
    vocab_size = int(metadata["vocab_size"])
    lookup = metadata["lookup"].astype(np.float32)
    embedding_dim = int(lookup.shape[1])

    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.02, size=(vocab_size + 1, embedding_dim)).astype(np.float32)
    matrix[: lookup.shape[0]] = lookup
    matrix[vocab_size] = 0.0
    return matrix


def pad_token_id(metadata: dict) -> int:
    return int(metadata["vocab_size"])


def describe_sequence_lengths(sequences) -> dict:
    lengths = np.array([len(sequences[i]) for i in range(sequences.shape[0])], dtype=np.int32)
    return {
        "min": int(lengths.min()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "max": int(lengths.max()),
    }
