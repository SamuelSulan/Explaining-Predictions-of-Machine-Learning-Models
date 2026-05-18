"""Final saved-model evaluation and XAI analysis utilities."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse

from mmimdb.constants import GENRE_LABELS
from mmimdb.data import DatasetPaths, load_labels
from mmimdb.evaluation import multilabel_metrics, threshold_predictions, threshold_to_serializable
from mmimdb.image_utils import image_descriptor
from mmimdb.models.classic import load_reconstructed_texts
from mmimdb.models.neural import MMIMDBTorchDataset, make_loader, predict
from mmimdb.splits import load_split_indices
from mmimdb.utils import ensure_dir, load_config, resolve_path, save_json
from mmimdb.xai import (
    directory_output_stats,
    explain_classic_samples,
    explain_neural_samples,
    load_neural_model,
    patch_classic_classifier_compat,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "sample_f1",
        "micro_f1",
        "macro_f1",
        "weighted_f1",
        "micro_precision",
        "micro_recall",
        "hamming_loss",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def _classic_feature_matrix(artifact: dict, hdf5_path: str | Path, metadata_path: str | Path, indices: np.ndarray):
    cfg = artifact.get("config", {})
    hist_bins = int(cfg.get("image_hist_bins", 16))
    thumbnail_size = tuple(cfg.get("image_thumbnail_size", (32, 20)))
    texts = load_reconstructed_texts(hdf5_path, metadata_path, indices)
    x_text = artifact["vectorizer"].transform(texts)
    descriptors = []
    with h5py.File(resolve_path(hdf5_path), "r") as f:
        images = f["images"]
        for idx in indices:
            descriptors.append(
                image_descriptor(
                    images[int(idx)],
                    hist_bins=hist_bins,
                    thumbnail_size=thumbnail_size,
                )
            )
    x_img = artifact["image_scaler"].transform(np.vstack(descriptors).astype(np.float32))
    return sparse.hstack([x_text, sparse.csr_matrix(x_img)], format="csr")


def evaluate_classic_model(
    model_path: str | Path,
    hdf5_path: str | Path,
    metadata_path: str | Path,
    test_indices: np.ndarray,
    output_dir: str | Path,
) -> dict[str, Any]:
    artifact = joblib.load(resolve_path(model_path))
    patch_classic_classifier_compat(artifact["classifier"])
    y = load_labels(hdf5_path)[test_indices]
    x_test = _classic_feature_matrix(artifact, hdf5_path, metadata_path, test_indices)
    y_prob = np.asarray(artifact["classifier"].predict_proba(x_test), dtype=np.float32)
    threshold = artifact.get("threshold", 0.5)
    metrics = multilabel_metrics(y, y_prob, threshold=threshold)
    y_pred = threshold_predictions(y_prob, threshold)
    out = ensure_dir(output_dir)
    np.savez_compressed(out / "classic_test_predictions.npz", indices=test_indices, y_true=y, y_prob=y_prob, y_pred=y_pred)
    result = {
        "model_type": "classic",
        "model_path": str(resolve_path(model_path)),
        "threshold": threshold_to_serializable(threshold),
        "n_test": int(len(test_indices)),
        "metrics": metrics,
        "prediction_path": str(out / "classic_test_predictions.npz"),
    }
    save_json(_json_safe(result), out / "classic_test_metrics.json")
    return result


def evaluate_neural_model(
    checkpoint_path: str | Path,
    hdf5_path: str | Path,
    metadata_path: str | Path,
    test_indices: np.ndarray,
    output_dir: str | Path,
    batch_size: int | None = None,
) -> dict[str, Any]:
    model, cfg, metadata, checkpoint, device = load_neural_model(checkpoint_path, metadata_path)
    dataset = MMIMDBTorchDataset(
        hdf5_path,
        test_indices,
        vocab_size=int(metadata["vocab_size"]),
        max_length=cfg.max_length,
        input_size=cfg.input_size,
    )
    loader = make_loader(dataset, batch_size or cfg.batch_size, shuffle=False, num_workers=0)
    y, y_prob = predict(model, loader, device)
    threshold = checkpoint.get("threshold", 0.5)
    metrics = multilabel_metrics(y, y_prob, threshold=threshold)
    y_pred = threshold_predictions(y_prob, threshold)
    out = ensure_dir(output_dir)
    np.savez_compressed(out / "neural_test_predictions.npz", indices=test_indices, y_true=y, y_prob=y_prob, y_pred=y_pred)
    result = {
        "model_type": "neural",
        "model_path": str(resolve_path(checkpoint_path)),
        "device": device,
        "threshold": threshold_to_serializable(threshold),
        "n_test": int(len(test_indices)),
        "metrics": metrics,
        "prediction_path": str(out / "neural_test_predictions.npz"),
    }
    save_json(_json_safe(result), out / "neural_test_metrics.json")
    return result


def saved_metric_result(
    model_type: str,
    model_path: str | Path,
    metrics_path: str | Path,
    output_dir: str | Path,
    reason: str,
) -> dict[str, Any]:
    """Load saved test metrics when fresh inference dependencies are unavailable."""
    import json

    metrics_path = resolve_path(metrics_path)
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload.get("test") or payload.get("test_metrics")
    if not metrics:
        raise KeyError(f"No test metrics found in {metrics_path}.")
    threshold = payload.get("threshold", payload.get("test", {}).get("threshold", 0.5))
    result = {
        "model_type": model_type,
        "model_path": str(resolve_path(model_path)),
        "threshold": threshold_to_serializable(threshold),
        "n_test": int(payload.get("n_test", 0)),
        "metrics": metrics,
        "prediction_path": None,
        "fresh_inference": False,
        "fallback_reason": reason,
        "source_metrics_path": str(metrics_path),
    }
    out = ensure_dir(output_dir)
    save_json(_json_safe(result), out / f"{model_type}_test_metrics.json")
    return result


def select_best_model(results: dict[str, dict[str, Any]], metric: str = "macro_f1") -> dict[str, Any]:
    ranked = sorted(
        results.values(),
        key=lambda item: float(item["metrics"][metric]),
        reverse=True,
    )
    best = ranked[0]
    return {
        "metric": metric,
        "best_model_type": best["model_type"],
        "best_model_path": best["model_path"],
        "best_score": float(best["metrics"][metric]),
        "ranking": [
            {
                "model_type": item["model_type"],
                "model_path": item["model_path"],
                "score": float(item["metrics"][metric]),
                "metrics": _compact_metrics(item["metrics"]),
            }
            for item in ranked
        ],
    }


def _metrics_frame(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_type, result in results.items():
        for metric, value in _compact_metrics(result["metrics"]).items():
            rows.append({"model_type": model_type, "metric": metric, "value": float(value)})
    return pd.DataFrame(rows)


def _per_label_frame(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_type, result in results.items():
        per_label = result["metrics"].get("per_label", {})
        for genre, values in per_label.items():
            rows.append({"model_type": model_type, "genre": genre, **values})
    return pd.DataFrame(rows)


def _prediction_frequency_frame(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_type, result in results.items():
        if not result.get("prediction_path"):
            continue
        data = np.load(result["prediction_path"])
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        for i, genre in enumerate(GENRE_LABELS):
            rows.append(
                {
                    "model_type": model_type,
                    "genre": genre,
                    "true_frequency": float(y_true[:, i].mean()),
                    "predicted_frequency": float(y_pred[:, i].mean()),
                }
            )
    return pd.DataFrame(rows)


def _save_metric_plots(results: dict[str, dict[str, Any]], output_dir: str | Path) -> dict[str, str]:
    out = ensure_dir(output_dir)
    figures: dict[str, str] = {}
    metric_df = _metrics_frame(results)
    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=metric_df, x="metric", y="value", hue="model_type")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, max(1.0, float(metric_df["value"].max()) * 1.15))
    plt.title("Final test metrics for saved best models")
    plt.tight_layout()
    path = out / "final_test_metric_comparison.png"
    plt.savefig(path, dpi=170)
    plt.close()
    figures["final_test_metric_comparison"] = str(path)

    per_label_df = _per_label_frame(results)
    plt.figure(figsize=(11, 7))
    sns.barplot(data=per_label_df, y="genre", x="f1", hue="model_type")
    plt.xlim(0, 1)
    plt.title("Per-label F1 on the held-out test set")
    plt.tight_layout()
    path = out / "per_label_f1_comparison.png"
    plt.savefig(path, dpi=170)
    plt.close()
    figures["per_label_f1_comparison"] = str(path)

    freq_df = _prediction_frequency_frame(results)
    if not freq_df.empty:
        freq_long = freq_df.melt(
            id_vars=["model_type", "genre"],
            value_vars=["true_frequency", "predicted_frequency"],
            var_name="frequency_type",
            value_name="frequency",
        )
        for model_type in sorted(freq_df["model_type"].unique()):
            plt.figure(figsize=(11, 7))
            sns.barplot(
                data=freq_long[freq_long["model_type"] == model_type],
                y="genre",
                x="frequency",
                hue="frequency_type",
            )
            plt.xlim(0, max(0.65, float(freq_long["frequency"].max()) * 1.1))
            plt.title(f"True vs predicted genre frequency: {model_type}")
            plt.tight_layout()
            path = out / f"{model_type}_predicted_vs_true_frequency.png"
            plt.savefig(path, dpi=170)
            plt.close()
            figures[f"{model_type}_predicted_vs_true_frequency"] = str(path)

    for model_type, result in results.items():
        if not result.get("prediction_path"):
            continue
        data = np.load(result["prediction_path"])
        confidence = data["y_prob"].max(axis=1)
        plt.figure(figsize=(7, 4))
        sns.histplot(confidence, bins=30, color="#146c94")
        plt.xlabel("max label probability")
        plt.title(f"Prediction confidence distribution: {model_type}")
        plt.tight_layout()
        path = out / f"{model_type}_confidence_distribution.png"
        plt.savefig(path, dpi=170)
        plt.close()
        figures[f"{model_type}_confidence_distribution"] = str(path)
    return figures


def _flatten_xai_samples(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return list(summary.get("samples", []))


def summarize_xai_outputs(xai_results: dict[str, dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    rows = []
    token_counter: Counter[str] = Counter()
    feature_counter: Counter[str] = Counter()
    for model_type, summary in xai_results.items():
        for sample in _flatten_xai_samples(summary):
            shapley = sample.get("modality_shapley", {}).get("logit", {}).get("shapley", {})
            if shapley:
                rows.append(
                    {
                        "model_type": model_type,
                        "index": sample.get("index"),
                        "target_genre": sample.get("target_genre"),
                        "text_utilization": float(shapley.get("text_utilization", 0.0)),
                        "image_utilization": float(shapley.get("image_utilization", 0.0)),
                        "text_logit": float(shapley.get("text", 0.0)),
                        "image_logit": float(shapley.get("image", 0.0)),
                        "interaction_logit": float(shapley.get("interaction", 0.0)),
                    }
                )
            for item in sample.get("top_tokens", []):
                token_counter[str(item.get("token", ""))] += 1
            for item in sample.get("top_text_features", []):
                feature_counter[str(item.get("feature", ""))] += 1

    out = ensure_dir(output_dir)
    shapley_df = pd.DataFrame(rows)
    figures = {}
    if not shapley_df.empty:
        shapley_df.to_csv(out / "xai_modality_utilization.csv", index=False)
        util_long = shapley_df.melt(
            id_vars=["model_type", "index", "target_genre"],
            value_vars=["text_utilization", "image_utilization"],
            var_name="modality",
            value_name="utilization",
        )
        plt.figure(figsize=(8, 4.5))
        sns.barplot(data=util_long, x="model_type", y="utilization", hue="modality")
        plt.ylim(0, 1)
        plt.title("Mean local modality utilization from Shapley explanations")
        plt.tight_layout()
        path = out / "xai_modality_utilization.png"
        plt.savefig(path, dpi=170)
        plt.close()
        figures["xai_modality_utilization"] = str(path)

    summary = {
        "local_explanation_count": int(sum(len(_flatten_xai_samples(s)) for s in xai_results.values())),
        "by_model": {
            model_type: {
                "sample_count": int(len(_flatten_xai_samples(model_summary))),
                "modality_shapley_summary": model_summary.get("modality_shapley_summary", {}),
                "performance": model_summary.get("performance", {}).get("summary", {}),
                "output_stats": model_summary.get("output_stats", {}),
            }
            for model_type, model_summary in xai_results.items()
        },
        "top_neural_tokens": [{"token": token, "count": count} for token, count in token_counter.most_common(20)],
        "top_classic_text_features": [
            {"feature": feature, "count": count} for feature, count in feature_counter.most_common(20)
        ],
        "figures": figures,
    }
    save_json(_json_safe(summary), out / "xai_global_summary.json")
    return summary


def build_html_report(analysis: dict[str, Any], output_path: str | Path) -> Path:
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def rel(path: str) -> str:
        return Path(path).resolve().relative_to(output_path.parent.resolve()).as_posix()

    figures = analysis.get("figures", {})
    xai_figures = analysis.get("xai_summary", {}).get("figures", {})
    ranking_rows = "".join(
        f"<tr><td>{row['model_type']}</td><td>{row['score']:.4f}</td>"
        f"<td>{row['metrics'].get('micro_f1', 0):.4f}</td><td>{row['metrics'].get('sample_f1', 0):.4f}</td></tr>"
        for row in analysis["best_model"]["ranking"]
    )
    figure_html = "\n".join(
        f"<figure><img src='{rel(path)}' alt='{name}'><figcaption>{name.replace('_', ' ')}</figcaption></figure>"
        for name, path in {**figures, **xai_figures}.items()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Final MM-IMDb XAI Analysis</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, system-ui, sans-serif; background: #f7f8fa; color: #18212b; }}
    header {{ background: #18242e; color: white; padding: 22px 28px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 22px; }}
    section {{ margin: 18px 0 28px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9e0e7; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #d9e0e7; text-align: left; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; background: white; border: 1px solid #d9e0e7; border-radius: 8px; overflow: hidden; }}
    img {{ display: block; width: 100%; }}
    figcaption {{ padding: 9px 12px; color: #627180; border-top: 1px solid #d9e0e7; }}
    .pill {{ display: inline-block; padding: 4px 9px; border-radius: 999px; background: #e8f3f5; }}
  </style>
</head>
<body>
  <header>
    <h1>Final MM-IMDb XAI Analysis</h1>
    <div>Global and local analysis on saved trained models using the held-out test split.</div>
  </header>
  <main>
    <section>
      <h2>Best Model</h2>
      <p><span class="pill">{analysis['best_model']['best_model_type']}</span>
      selected by {analysis['best_model']['metric']} = {analysis['best_model']['best_score']:.4f}.</p>
      <table><thead><tr><th>Model</th><th>Macro F1</th><th>Micro F1</th><th>Sample F1</th></tr></thead><tbody>{ranking_rows}</tbody></table>
    </section>
    <section>
      <h2>Visual Analysis</h2>
      <div class="grid">{figure_html}</div>
    </section>
    <section>
      <h2>Artifacts</h2>
      <p>Machine-readable metrics and summaries are saved next to this report. Local qualitative explanations are under <code>xai/classic</code> and <code>xai/neural</code>.</p>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def run_final_xai_analysis(
    config_path: str | Path = "configs/default.yaml",
    output_dir: str | Path = "outputs/final_xai_analysis",
    xai_limit: int | None = None,
    xai_model_type: str = "both",
    skip_local_xai: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    paths = DatasetPaths.from_config(config)
    _, _, test_idx = load_split_indices(resolve_path(config["splits"]["output_dir"]))
    out = ensure_dir(output_dir)
    eval_out = ensure_dir(out / "metrics")
    figure_out = ensure_dir(out / "figures")

    xai_cfg = config.get("xai", {})
    classic_model = resolve_path(xai_cfg.get("classic", {}).get("model_path", "outputs/models/best/classic_multimodal_best.joblib"))
    neural_checkpoint = resolve_path(xai_cfg.get("neural", {}).get("checkpoint_path", "outputs/models/best/neural_multimodal_best.pt"))
    results = {}
    if classic_model.exists():
        results["classic"] = evaluate_classic_model(classic_model, paths.hdf5, paths.metadata, test_idx, eval_out)
    if neural_checkpoint.exists():
        try:
            results["neural"] = evaluate_neural_model(neural_checkpoint, paths.hdf5, paths.metadata, test_idx, eval_out)
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            results["neural"] = saved_metric_result(
                "neural",
                neural_checkpoint,
                "outputs/models/best/neural_multimodal_best_metrics.json",
                eval_out,
                "PyTorch is not installed in the active local Python environment; loaded saved test metrics instead.",
            )
    if not results:
        raise FileNotFoundError("No saved best models were found for final analysis.")

    metric = str(config.get("model_registry", {}).get("metric", "macro_f1"))
    best_model = select_best_model(results, metric=metric)
    figures = _save_metric_plots(results, figure_out)

    xai_results = {}
    if not skip_local_xai:
        local_limit = int(xai_limit if xai_limit is not None else xai_cfg.get("limit", 10))
        local_indices = test_idx[:local_limit]
        neural_xai_cfg = xai_cfg.get("neural", {})
        classic_xai_cfg = xai_cfg.get("classic", {})
        target_policy = xai_cfg.get("target_policy", "predicted")
        target_top_k = int(xai_cfg.get("target_top_k", 3))
        max_targets_per_sample = xai_cfg.get("max_targets_per_sample", 3)
        if xai_model_type in {"classic", "both"} and "classic" in results:
            xai_results["classic"] = explain_classic_samples(
                classic_model,
                paths.hdf5,
                paths.metadata,
                local_indices,
                out / "xai" / "classic",
                target_genre=xai_cfg.get("target_genre"),
                target_genres=xai_cfg.get("target_genres", []),
                target_policy=target_policy,
                target_top_k=target_top_k,
                max_targets_per_sample=max_targets_per_sample,
                ensure_at_least_one_target=bool(xai_cfg.get("ensure_at_least_one_target", True)),
                top_k=int(xai_cfg.get("top_k", 12)),
                measure_performance=bool(xai_cfg.get("measure_performance", True)),
                performance_sample_interval_seconds=float(xai_cfg.get("performance_sample_interval_seconds", 0.02)),
                enable_modality_shapley=bool(classic_xai_cfg.get("enable_modality_shapley", True)),
                enable_experimental_set_explanation=bool(xai_cfg.get("enable_experimental_set_explanation", True)),
            )
        if xai_model_type in {"neural", "both"} and "neural" in results:
            try:
                xai_results["neural"] = explain_neural_samples(
                    neural_checkpoint,
                    paths.hdf5,
                    paths.metadata,
                    local_indices,
                    out / "xai" / "neural",
                    target_genre=xai_cfg.get("target_genre"),
                    target_genres=xai_cfg.get("target_genres", []),
                    target_policy=target_policy,
                    target_top_k=target_top_k,
                    max_targets_per_sample=max_targets_per_sample,
                    ensure_at_least_one_target=bool(xai_cfg.get("ensure_at_least_one_target", True)),
                    top_k=int(xai_cfg.get("top_k", 12)),
                    n_steps=int(xai_cfg.get("n_steps", 8)),
                    measure_performance=bool(xai_cfg.get("measure_performance", True)),
                    performance_sample_interval_seconds=float(xai_cfg.get("performance_sample_interval_seconds", 0.02)),
                    enable_layer_integrated_gradients_text=bool(
                        neural_xai_cfg.get("enable_layer_integrated_gradients_text", True)
                    ),
                    enable_integrated_gradients_image=bool(
                        neural_xai_cfg.get("enable_integrated_gradients_image", True)
                    ),
                    enable_gradcam=bool(neural_xai_cfg.get("enable_gradcam", True)),
                    enable_modality_ablation=bool(neural_xai_cfg.get("enable_modality_ablation", True)),
                    enable_modality_shapley=bool(neural_xai_cfg.get("enable_modality_shapley", True)),
                    enable_token_occlusion=bool(neural_xai_cfg.get("enable_token_occlusion", True)),
                    enable_image_occlusion=bool(neural_xai_cfg.get("enable_image_occlusion", True)),
                    image_occlusion_grid=int(neural_xai_cfg.get("image_occlusion_grid", 4)),
                    occlusion_batch_size=int(neural_xai_cfg.get("occlusion_batch_size", 32)),
                    enable_experimental_set_explanation=bool(xai_cfg.get("enable_experimental_set_explanation", True)),
                )
            except ModuleNotFoundError as exc:
                if exc.name != "torch":
                    raise
                import json

                existing_summary = resolve_path("outputs/xai/neural/neural_xai_summary.json")
                if existing_summary.exists():
                    with existing_summary.open("r", encoding="utf-8") as f:
                        xai_results["neural"] = json.load(f)
                    xai_results["neural"]["fresh_inference"] = False
                    xai_results["neural"]["fallback_reason"] = (
                        "PyTorch is not installed in the active local Python environment; "
                        "loaded the existing neural XAI summary instead."
                    )
    xai_summary = summarize_xai_outputs(xai_results, figure_out) if xai_results else {}

    analysis = {
        "config_path": str(resolve_path(config_path)),
        "test_count": int(len(test_idx)),
        "results": results,
        "best_model": best_model,
        "figures": figures,
        "xai_summary": xai_summary,
        "output_stats": directory_output_stats(out),
    }
    save_json(_json_safe(analysis), out / "final_xai_analysis_summary.json")
    report_path = build_html_report(analysis, out / "final_xai_analysis.html")
    analysis["html_report_path"] = str(report_path)
    save_json(_json_safe(analysis), out / "final_xai_analysis_summary.json")
    return analysis
