from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmimdb.final_analysis import run_final_xai_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved best models on the test split and generate final XAI analysis artifacts."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/final_xai_analysis")
    parser.add_argument("--xai-limit", type=int, default=None)
    parser.add_argument("--xai-model-type", choices=["classic", "neural", "both"], default="both")
    parser.add_argument("--skip-local-xai", action="store_true")
    args = parser.parse_args()

    summary = run_final_xai_analysis(
        config_path=args.config,
        output_dir=args.output_dir,
        xai_limit=args.xai_limit,
        xai_model_type=args.xai_model_type,
        skip_local_xai=args.skip_local_xai,
    )
    pprint(
        {
            "test_count": summary["test_count"],
            "best_model": summary["best_model"],
            "html_report_path": summary["html_report_path"],
        }
    )


if __name__ == "__main__":
    main()
