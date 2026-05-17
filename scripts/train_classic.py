from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmimdb.data import DatasetPaths
from mmimdb.model_registry import update_best_model
from mmimdb.models.classic import ClassicConfig, train_classic_multimodal
from mmimdb.splits import load_split_indices, save_splits
from mmimdb.utils import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multimodal classic ML baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test sample limit.")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = DatasetPaths.from_config(config)
    split_dir = resolve_path(config["splits"]["output_dir"])

    if not (split_dir / "train_indices.npy").exists():
        save_splits(
            paths.hdf5,
            split_dir,
            train_size=float(config["splits"]["train_size"]),
            val_size=float(config["splits"]["val_size"]),
            test_size=float(config["splits"]["test_size"]),
            random_state=int(config["splits"]["random_state"]),
        )

    train_idx, val_idx, test_idx = load_split_indices(split_dir)
    result = train_classic_multimodal(
        paths.hdf5,
        paths.metadata,
        train_idx,
        val_idx,
        test_idx,
        output_dir=resolve_path(config["project"]["output_dir"]) / "models",
        cfg=ClassicConfig.from_config(config),
        limit=args.limit,
    )
    result["best_registry"] = update_best_model(
        result["model_path"],
        result,
        config.get("model_registry", {}),
        model_kind="classic",
        is_full_run=args.limit is None,
    )
    pprint(result)


if __name__ == "__main__":
    main()
