from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmimdb.global_xai import run_global_neural_xai


def main() -> None:
    parser = argparse.ArgumentParser(description="Run global XAI for the canonical best neural MM-IMDb model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/global_xai/best_neural")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--samples-per-label", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--token-candidates-per-label", type=int, default=20)
    parser.add_argument("--token-occlusion-top-k", type=int, default=10)
    parser.add_argument("--image-occlusion-grid", type=int, default=4)
    parser.add_argument("--local-xai-dir", default=None)
    parser.add_argument("--no-token-occlusion", action="store_true")
    parser.add_argument("--no-image-occlusion", action="store_true")
    args = parser.parse_args()

    summary = run_global_neural_xai(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        samples_per_label=args.samples_per_label,
        seed=args.seed,
        batch_size=args.batch_size,
        token_candidates_per_label=args.token_candidates_per_label,
        token_occlusion_top_k=args.token_occlusion_top_k,
        image_occlusion_grid=args.image_occlusion_grid,
        enable_token_occlusion=not args.no_token_occlusion,
        enable_image_occlusion=not args.no_image_occlusion,
        local_xai_dir=args.local_xai_dir,
    )
    pprint(
        {
            "output_dir": args.output_dir,
            "summary_path": str(Path(args.output_dir) / "global_xai_summary.json"),
            "selected_sample_count": summary["selected_sample_count"],
            "runtime_seconds": summary["runtime_seconds"],
            "warnings": summary["warnings"][:5],
        }
    )


if __name__ == "__main__":
    main()

