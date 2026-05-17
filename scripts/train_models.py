from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmimdb.model_selection import run_training_process
from mmimdb.utils import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cross-validation model selection and final training for MM-IMDb models."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-type", choices=["classic", "neural", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test sample limit.")
    parser.add_argument("--folds", type=int, default=None, help="Override configured CV fold count.")
    parser.add_argument("--epochs", type=int, default=None, help="Override neural epoch count for this run.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override neural batch size for this run.")
    parser.add_argument(
        "--no-pretrained-image",
        action="store_true",
        help="Disable pretrained torchvision image weights for quick neural smoke tests.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config.setdefault("neural", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("neural", {})["batch_size"] = args.batch_size
    if args.no_pretrained_image:
        config.setdefault("neural", {})["pretrained_image"] = False
    summary = run_training_process(
        config,
        model_type=args.model_type,
        limit=args.limit,
        folds_override=args.folds,
    )
    pprint(
        {
            "metric": summary["metric"],
            "n_folds": summary["n_folds"],
            "overall_best": summary.get("overall_best"),
            "summary_path": str(
                resolve_path(config.get("training", {}).get("output_dir", "outputs/model_selection"))
                / "training_summary.json"
            ),
        }
    )


if __name__ == "__main__":
    main()
