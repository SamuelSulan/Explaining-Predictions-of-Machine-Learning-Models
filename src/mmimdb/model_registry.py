"""Best-model registry utilities."""

from __future__ import annotations

import shutil
from pathlib import Path

from mmimdb.utils import ensure_dir, resolve_path, save_json


def _metric_value(metrics: dict, metric: str) -> float:
    if metric in metrics:
        return float(metrics[metric])
    if "validation" in metrics and metric in metrics["validation"]:
        return float(metrics["validation"][metric])
    if "test" in metrics and metric in metrics["test"]:
        return float(metrics["test"][metric])
    raise KeyError(f"Metric '{metric}' not found in metrics result.")


def update_best_model(
    source_model_path: str | Path,
    result: dict,
    registry_cfg: dict,
    model_kind: str,
    is_full_run: bool,
) -> dict:
    """Copy a run model to the canonical best path when it improves validation metric."""
    if registry_cfg.get("update_best_only_for_full_runs", True) and not is_full_run:
        return {
            "updated": False,
            "reason": "limited run; best registry updates are disabled for smoke tests",
        }

    metric = str(registry_cfg.get("metric", "macro_f1"))
    best_dir = ensure_dir(registry_cfg.get("best_dir", "outputs/models/best"))
    if model_kind == "classic":
        best_name = registry_cfg.get("classic_best_name", "classic_multimodal_best.joblib")
    elif model_kind == "neural":
        best_name = registry_cfg.get("neural_best_name", "neural_multimodal_best.pt")
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")

    best_model_path = best_dir / best_name
    best_metrics_path = best_dir / f"{Path(best_name).stem}_metrics.json"
    source_model_path = resolve_path(source_model_path)
    candidate_score = _metric_value(result, metric)

    previous_score = None
    if best_metrics_path.exists():
        import json

        with best_metrics_path.open("r", encoding="utf-8") as f:
            previous = json.load(f)
        previous_score = previous.get("registry", {}).get("score")

    should_update = previous_score is None or candidate_score >= float(previous_score)
    registry = {
        "updated": bool(should_update),
        "metric": metric,
        "score": float(candidate_score),
        "previous_score": previous_score,
        "best_model_path": str(best_model_path),
        "best_metrics_path": str(best_metrics_path),
    }

    if should_update:
        shutil.copy2(source_model_path, best_model_path)
        payload = dict(result)
        payload["registry"] = registry
        save_json(payload, best_metrics_path)

    return registry
