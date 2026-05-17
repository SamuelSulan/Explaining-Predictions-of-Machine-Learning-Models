"""Minimal example for running the project code from a notebook cell.

In Jupyter, the same imports work after:

    pip install -e .

or after launching Jupyter from the project root with `src` on PYTHONPATH.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmimdb.data import DatasetPaths, load_labels
from mmimdb.splits import save_splits
from mmimdb.utils import load_config, resolve_path


config = load_config("configs/default.yaml")
paths = DatasetPaths.from_config(config)
y = load_labels(paths.hdf5)
print(y.shape)

save_splits(
    paths.hdf5,
    resolve_path(config["splits"]["output_dir"]),
    train_size=config["splits"]["train_size"],
    val_size=config["splits"]["val_size"],
    test_size=config["splits"]["test_size"],
    random_state=config["splits"]["random_state"],
)
