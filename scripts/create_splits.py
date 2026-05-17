from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmimdb.data import DatasetPaths
from mmimdb.splits import save_splits
from mmimdb.utils import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible multilabel train/val/test splits.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = DatasetPaths.from_config(config)
    split_cfg = config["splits"]
    metadata = save_splits(
        paths.hdf5,
        resolve_path(split_cfg["output_dir"]),
        train_size=float(split_cfg["train_size"]),
        val_size=float(split_cfg["val_size"]),
        test_size=float(split_cfg["test_size"]),
        random_state=int(split_cfg["random_state"]),
    )
    pprint(metadata)


if __name__ == "__main__":
    main()
