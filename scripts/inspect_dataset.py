from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmimdb.data import DatasetPaths, dataset_label_stats, h5_summary, load_labels, load_metadata
from mmimdb.text_utils import describe_sequence_lengths
from mmimdb.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MM-IMDb HDF5 and metadata files.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = DatasetPaths.from_config(config)

    print("HDF5 summary")
    pprint(h5_summary(paths.hdf5))

    metadata = load_metadata(paths.metadata)
    print("\nMetadata")
    print("keys:", sorted(metadata.keys()))
    print("vocab_size:", metadata["vocab_size"])
    print("lookup shape:", metadata["lookup"].shape)

    y = load_labels(paths.hdf5)
    print("\nLabel stats")
    pprint(dataset_label_stats(y))

    with h5py.File(paths.hdf5, "r") as f:
        print("\nSequence lengths")
        pprint(describe_sequence_lengths(f["sequences"]))


if __name__ == "__main__":
    main()
