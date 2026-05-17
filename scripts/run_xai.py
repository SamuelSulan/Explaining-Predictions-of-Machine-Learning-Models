from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmimdb.data import DatasetPaths
from mmimdb.splits import load_split_indices
from mmimdb.utils import load_config, resolve_path
from mmimdb.xai import explain_classic_samples, explain_neural_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XAI explanations for saved multimodal models.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-type", choices=["classic", "neural", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--classic-model", default=None)
    parser.add_argument("--neural-checkpoint", default=None)
    parser.add_argument("--target-genre", default=None)
    parser.add_argument("--target-genres", default=None, help="Comma-separated explicit target genres.")
    parser.add_argument(
        "--target-policy",
        choices=["top", "top_k", "predicted", "all"],
        default=None,
        help="How to choose labels to explain when explicit target genres are not provided.",
    )
    parser.add_argument("--target-top-k", type=int, default=None, help="Number of labels for target-policy=top_k.")
    parser.add_argument("--max-targets-per-sample", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--image-occlusion-grid", type=int, default=None)
    parser.add_argument("--no-occlusion", action="store_true")
    parser.add_argument("--no-set-explanation", action="store_true")
    parser.add_argument("--no-performance", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    xai_cfg = config.get("xai", {})
    paths = DatasetPaths.from_config(config)
    train_idx, val_idx, test_idx = load_split_indices(resolve_path(config["splits"]["output_dir"]))
    split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    split_name = args.split or xai_cfg.get("split", "test")
    limit = int(args.limit if args.limit is not None else xai_cfg.get("limit", 10))
    target_genre = args.target_genre if args.target_genre is not None else xai_cfg.get("target_genre")
    target_genres = args.target_genres if args.target_genres is not None else xai_cfg.get("target_genres", [])
    target_policy = args.target_policy if args.target_policy is not None else xai_cfg.get("target_policy", "top")
    target_top_k = int(args.target_top_k if args.target_top_k is not None else xai_cfg.get("target_top_k", 1))
    max_targets_per_sample = (
        args.max_targets_per_sample
        if args.max_targets_per_sample is not None
        else xai_cfg.get("max_targets_per_sample")
    )
    ensure_at_least_one_target = bool(xai_cfg.get("ensure_at_least_one_target", True))
    enable_experimental_set_explanation = (
        bool(xai_cfg.get("enable_experimental_set_explanation", False)) and not args.no_set_explanation
    )
    n_steps = int(args.n_steps if args.n_steps is not None else xai_cfg.get("n_steps", 8))
    top_k = int(args.top_k if args.top_k is not None else xai_cfg.get("top_k", 12))
    measure_performance = bool(xai_cfg.get("measure_performance", True)) and not args.no_performance
    sample_interval = float(xai_cfg.get("performance_sample_interval_seconds", 0.02))
    neural_xai_cfg = xai_cfg.get("neural", {})
    classic_xai_cfg = xai_cfg.get("classic", {})
    image_occlusion_grid = int(
        args.image_occlusion_grid
        if args.image_occlusion_grid is not None
        else neural_xai_cfg.get("image_occlusion_grid", 4)
    )
    occlusion_batch_size = int(neural_xai_cfg.get("occlusion_batch_size", 32))

    indices = split_map[split_name][:limit]
    out_dir = resolve_path(xai_cfg.get("output_dir", "outputs/xai"))

    results = {}
    if args.model_type in {"classic", "both"}:
        classic_model = resolve_path(
            args.classic_model
            or xai_cfg.get("classic", {}).get("model_path", "outputs/models/best/classic_multimodal_best.joblib")
        )
        if classic_model.exists():
            results["classic"] = explain_classic_samples(
                classic_model,
                paths.hdf5,
                paths.metadata,
                indices,
                out_dir / "classic",
                target_genre=target_genre,
                target_genres=target_genres,
                target_policy=target_policy,
                target_top_k=target_top_k,
                max_targets_per_sample=max_targets_per_sample,
                ensure_at_least_one_target=ensure_at_least_one_target,
                top_k=top_k,
                measure_performance=measure_performance,
                performance_sample_interval_seconds=sample_interval,
                enable_modality_shapley=bool(classic_xai_cfg.get("enable_modality_shapley", True)),
                enable_experimental_set_explanation=enable_experimental_set_explanation,
            )
        else:
            print(f"Classic model not found, skipping: {classic_model}")

    if args.model_type in {"neural", "both"}:
        if args.neural_checkpoint:
            neural_checkpoint = resolve_path(args.neural_checkpoint)
        else:
            neural_checkpoint = resolve_path(
                xai_cfg.get("neural", {}).get("checkpoint_path", "outputs/models/best/neural_multimodal_best.pt")
            )
        if neural_checkpoint.exists():
            results["neural"] = explain_neural_samples(
                neural_checkpoint,
                paths.hdf5,
                paths.metadata,
                indices,
                out_dir / "neural",
                target_genre=target_genre,
                target_genres=target_genres,
                target_policy=target_policy,
                target_top_k=target_top_k,
                max_targets_per_sample=max_targets_per_sample,
                ensure_at_least_one_target=ensure_at_least_one_target,
                top_k=top_k,
                n_steps=n_steps,
                measure_performance=measure_performance,
                performance_sample_interval_seconds=sample_interval,
                enable_layer_integrated_gradients_text=bool(
                    neural_xai_cfg.get("enable_layer_integrated_gradients_text", True)
                ),
                enable_integrated_gradients_image=bool(
                    neural_xai_cfg.get("enable_integrated_gradients_image", True)
                ),
                enable_gradcam=bool(neural_xai_cfg.get("enable_gradcam", True)),
                enable_modality_ablation=bool(neural_xai_cfg.get("enable_modality_ablation", True)),
                enable_modality_shapley=bool(neural_xai_cfg.get("enable_modality_shapley", True)),
                enable_token_occlusion=bool(neural_xai_cfg.get("enable_token_occlusion", True))
                and not args.no_occlusion,
                enable_image_occlusion=bool(neural_xai_cfg.get("enable_image_occlusion", True))
                and not args.no_occlusion,
                image_occlusion_grid=image_occlusion_grid,
                occlusion_batch_size=occlusion_batch_size,
                enable_experimental_set_explanation=enable_experimental_set_explanation,
            )
        else:
            print(f"Neural checkpoint not found, skipping: {neural_checkpoint}")

    pprint(results.keys())


if __name__ == "__main__":
    main()
