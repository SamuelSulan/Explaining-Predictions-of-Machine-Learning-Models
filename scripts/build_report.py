from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmimdb.data import DatasetPaths
from mmimdb.reporting import build_dataset_report
from mmimdb.utils import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build thesis-style dataset and preprocessing report.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="docs/dataset_preprocessing_report.md")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = DatasetPaths.from_config(config)
    split_dir = resolve_path(config["splits"]["output_dir"])
    output = build_dataset_report(
        paths.hdf5,
        paths.metadata,
        paths.article_pdf,
        output_path=args.output,
        figures_dir=resolve_path(config["project"]["output_dir"]) / "figures",
        split_metadata_path=split_dir / "split_metadata.json",
    )
    print(output)


if __name__ == "__main__":
    main()
