"""Cross-validation model selection for training-only workflows."""

from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from mmimdb.data import DatasetPaths, load_labels
from mmimdb.model_registry import update_best_model
from mmimdb.models.classic import ClassicConfig, train_classic_multimodal
from mmimdb.models.neural import NeuralConfig, train_neural_multimodal
from mmimdb.splits import load_split_indices, multilabel_kfold_split, save_splits
from mmimdb.utils import ensure_dir, resolve_path, save_json


@dataclass
class Candidate:
    name: str
    params: dict[str, Any]


def _slug(value: Any) -> str:
    import re

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return slug[:120] or "candidate"


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _param_suffix(params: dict[str, Any]) -> str:
    if not params:
        return "base"
    return "__".join(_slug(f"{key}-{value}") for key, value in sorted(params.items()))


def _candidate_from_raw(raw: dict[str, Any], default_name: str) -> Candidate:
    if not isinstance(raw, dict):
        raise ValueError("Candidate entries must be dictionaries.")
    params = dict(raw.get("params", {}))
    for key, value in raw.items():
        if key not in {"name", "params"}:
            params[key] = value
    name = str(raw.get("name") or f"{default_name}_{_param_suffix(params)}")
    return Candidate(name=_slug(name), params=params)


def configured_candidates(training_cfg: dict[str, Any], kind: str) -> list[Candidate]:
    kind_cfg = training_cfg.get(kind, {})
    explicit = kind_cfg.get("candidates", training_cfg.get(f"{kind}_candidates", []))
    grid = kind_cfg.get("grid", training_cfg.get(f"{kind}_grid", {}))
    base_name = f"{kind}_base"

    candidates: list[Candidate] = []
    for raw in explicit or []:
        candidates.append(_candidate_from_raw(raw, base_name))

    if grid:
        keys = list(grid)
        values = [_as_list(grid[key]) for key in keys]
        for combo in product(*values):
            params = dict(zip(keys, combo))
            candidates.append(Candidate(name=_slug(f"{kind}_{_param_suffix(params)}"), params=params))

    if not candidates:
        candidates.append(Candidate(name=base_name, params={}))

    seen: dict[str, int] = {}
    unique: list[Candidate] = []
    for candidate in candidates:
        count = seen.get(candidate.name, 0)
        seen[candidate.name] = count + 1
        name = candidate.name if count == 0 else f"{candidate.name}_{count + 1}"
        unique.append(Candidate(name=name, params=dict(candidate.params)))
    return unique


def _none_if_requested(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    return value


def _coerce_like(current: Any, value: Any) -> Any:
    value = _none_if_requested(value)
    if value is None:
        return None
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)
    if isinstance(current, tuple):
        return tuple(value) if isinstance(value, (list, tuple)) else (value,)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _apply_params(config_obj: Any, params: dict[str, Any]) -> Any:
    valid_fields = {field.name for field in fields(config_obj)}
    unknown = sorted(set(params) - valid_fields)
    if unknown:
        raise ValueError(f"Unsupported parameters for {type(config_obj).__name__}: {unknown}")
    for key, value in params.items():
        setattr(config_obj, key, _coerce_like(getattr(config_obj, key), value))
    return config_obj


def classic_config_for_candidate(config: dict[str, Any], candidate: Candidate) -> ClassicConfig:
    return _apply_params(ClassicConfig.from_config(config), candidate.params)


def neural_config_for_candidate(config: dict[str, Any], candidate: Candidate) -> NeuralConfig:
    cfg = _apply_params(NeuralConfig.from_config(config), candidate.params)
    if "model_name" not in candidate.params:
        cfg.model_name = candidate.name
    return cfg


def _metric_score(result: dict[str, Any], metric: str) -> float:
    if metric in result:
        return float(result[metric])
    if "validation" in result and metric in result["validation"]:
        return float(result["validation"][metric])
    if "test" in result and metric in result["test"]:
        return float(result["test"][metric])
    raise KeyError(f"Metric '{metric}' not found in result.")


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


def _ensure_base_splits(config: dict[str, Any], paths: DatasetPaths) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return load_split_indices(split_dir)


def _select_indices_for_cv(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    training_cfg: dict[str, Any],
    limit: int | None,
) -> np.ndarray:
    combine = bool(training_cfg.get("combine_train_val_for_cv", True))
    indices = np.concatenate([train_idx, val_idx]) if combine else np.asarray(train_idx)
    indices = np.sort(np.unique(indices.astype(np.int64)))
    if limit is not None:
        indices = indices[: max(2, min(len(indices), int(limit)))]
    return indices


def _fold_summary(scores: list[float]) -> dict[str, float | int]:
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "folds_completed": int(arr.size),
        "mean_score": float(arr.mean()),
        "std_score": float(arr.std(ddof=0)),
        "min_score": float(arr.min()),
        "max_score": float(arr.max()),
    }


def _train_fold(
    kind: str,
    config: dict[str, Any],
    candidate: Candidate,
    paths: DatasetPaths,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    if kind == "classic":
        return train_classic_multimodal(
            paths.hdf5,
            paths.metadata,
            train_idx,
            val_idx,
            None,
            output_dir=output_dir,
            cfg=classic_config_for_candidate(config, candidate),
        )
    if kind == "neural":
        return train_neural_multimodal(
            paths.hdf5,
            paths.metadata,
            train_idx,
            val_idx,
            None,
            output_dir=output_dir,
            cfg=neural_config_for_candidate(config, candidate),
            seed=seed,
        )
    raise ValueError(f"Unsupported model kind: {kind}")


def _train_final(
    kind: str,
    config: dict[str, Any],
    candidate: Candidate,
    paths: DatasetPaths,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path,
    seed: int,
    limit: int | None,
) -> dict[str, Any]:
    if kind == "classic":
        return train_classic_multimodal(
            paths.hdf5,
            paths.metadata,
            train_idx,
            val_idx,
            test_idx,
            output_dir=output_dir,
            cfg=classic_config_for_candidate(config, candidate),
            limit=limit,
        )
    if kind == "neural":
        return train_neural_multimodal(
            paths.hdf5,
            paths.metadata,
            train_idx,
            val_idx,
            test_idx,
            output_dir=output_dir,
            cfg=neural_config_for_candidate(config, candidate),
            seed=seed,
            limit=limit,
        )
    raise ValueError(f"Unsupported model kind: {kind}")


def _run_kind_selection(
    kind: str,
    config: dict[str, Any],
    paths: DatasetPaths,
    folds: list[tuple[np.ndarray, np.ndarray]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path,
    metric: str,
    seed: int,
    limit: int | None,
) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    kind_cfg = training_cfg.get(kind, {})
    candidates = configured_candidates(training_cfg, kind)
    kind_out = ensure_dir(output_dir / kind)
    candidate_summaries: list[dict[str, Any]] = []

    if bool(kind_cfg.get("final_only", False)):
        candidate = candidates[0]
        print(f"[{kind}] final-only candidate: {candidate.name}")
        final_out = kind_out / candidate.name / "final"
        final_result = _train_final(
            kind,
            config,
            candidate,
            paths,
            train_idx,
            val_idx,
            test_idx,
            final_out,
            seed,
            limit,
        )
        final_result["selection"] = {
            "metric": metric,
            "candidate": candidate.name,
            "candidate_params": candidate.params,
            "cv_mean_score": None,
            "cv_std_score": None,
            "cv_folds": 0,
            "final_training_policy": (
                "final_only=true; selected the first configured candidate and trained with the original "
                "train/validation split for threshold tuning"
            ),
        }
        final_result["best_registry"] = update_best_model(
            final_result["model_path"],
            final_result,
            config.get("model_registry", {}),
            model_kind=kind,
            is_full_run=limit is None,
        )
        save_json(_json_safe(final_result), final_out / "final_result.json")
        return {
            "status": "completed",
            "metric": metric,
            "selection_mode": "final_only",
            "candidates": [
                {
                    "name": candidate.name,
                    "params": candidate.params,
                    "status": "selected_without_cv",
                }
            ],
            "final": final_result,
        }

    for candidate_no, candidate in enumerate(candidates, start=1):
        print(f"[{kind}] candidate {candidate_no}/{len(candidates)}: {candidate.name}")
        fold_results: list[dict[str, Any]] = []
        candidate_failed = False
        for fold_no, (fold_train_idx, fold_val_idx) in enumerate(folds, start=1):
            print(f"[{kind}]   fold {fold_no}/{len(folds)}")
            fold_out = kind_out / candidate.name / f"fold_{fold_no:02d}"
            try:
                result = _train_fold(
                    kind,
                    config,
                    candidate,
                    paths,
                    fold_train_idx,
                    fold_val_idx,
                    fold_out,
                    seed + fold_no,
                )
                score = _metric_score(result, metric)
                fold_results.append(
                    {
                        "fold": fold_no,
                        "score": score,
                        "n_train": int(len(fold_train_idx)),
                        "n_val": int(len(fold_val_idx)),
                        "model_path": result.get("model_path"),
                        "validation": result.get("validation", {}),
                    }
                )
            except Exception as exc:
                candidate_failed = True
                fold_results.append(
                    {
                        "fold": fold_no,
                        "error": f"{type(exc).__name__}: {exc}",
                        "n_train": int(len(fold_train_idx)),
                        "n_val": int(len(fold_val_idx)),
                    }
                )
                print(f"[{kind}]   fold {fold_no} failed: {type(exc).__name__}: {exc}")
                break

        scores = [float(result["score"]) for result in fold_results if "score" in result]
        summary = {
            "name": candidate.name,
            "params": candidate.params,
            "status": "failed" if candidate_failed or len(scores) != len(folds) else "completed",
            "folds": fold_results,
        }
        if scores:
            summary.update(_fold_summary(scores))
        candidate_summaries.append(summary)
        save_json(_json_safe({"candidates": candidate_summaries}), kind_out / "cv_summary.json")

    completed = [candidate for candidate in candidate_summaries if candidate.get("status") == "completed"]
    if not completed:
        return {
            "status": "failed",
            "metric": metric,
            "candidates": candidate_summaries,
            "error": f"No {kind} candidates completed all folds.",
        }

    completed.sort(key=lambda item: (float(item["mean_score"]), -float(item["std_score"])), reverse=True)
    best_summary = completed[0]
    best_candidate = next(candidate for candidate in candidates if candidate.name == best_summary["name"])

    print(f"[{kind}] selected {best_candidate.name} ({metric}={best_summary['mean_score']:.4f})")
    final_out = kind_out / best_candidate.name / "final"
    final_result = _train_final(
        kind,
        config,
        best_candidate,
        paths,
        train_idx,
        val_idx,
        test_idx,
        final_out,
        seed,
        limit,
    )
    final_result["selection"] = {
        "metric": metric,
        "candidate": best_candidate.name,
        "candidate_params": best_candidate.params,
        "cv_mean_score": best_summary["mean_score"],
        "cv_std_score": best_summary["std_score"],
        "cv_folds": len(folds),
        "final_training_policy": "selected by CV on train+val; final fit uses original train/validation split for threshold tuning and early stopping",
    }
    final_result["best_registry"] = update_best_model(
        final_result["model_path"],
        final_result,
        config.get("model_registry", {}),
        model_kind=kind,
        is_full_run=limit is None,
    )
    save_json(_json_safe(final_result), final_out / "final_result.json")

    return {
        "status": "completed",
        "metric": metric,
        "candidates": candidate_summaries,
        "best_candidate": best_summary,
        "final": final_result,
    }


def run_training_process(
    config: dict[str, Any],
    model_type: str = "both",
    limit: int | None = None,
    folds_override: int | None = None,
) -> dict[str, Any]:
    """Run CV model selection and final training for classic and/or neural models."""
    training_cfg = config.get("training", {})
    paths = DatasetPaths.from_config(config)
    train_idx, val_idx, test_idx = _ensure_base_splits(config, paths)
    seed = int(config.get("project", {}).get("seed", 42))
    metric = str(training_cfg.get("metric", config.get("model_registry", {}).get("metric", "macro_f1")))
    output_dir = ensure_dir(training_cfg.get("output_dir", Path(config["project"]["output_dir"]) / "model_selection"))

    cv_indices = _select_indices_for_cv(train_idx, val_idx, training_cfg, limit)
    y = load_labels(paths.hdf5)
    requested_folds = int(folds_override or training_cfg.get("n_folds", 3))
    folds = multilabel_kfold_split(cv_indices, y, n_splits=requested_folds, random_state=seed)

    summary: dict[str, Any] = {
        "status": "completed",
        "metric": metric,
        "limit": limit,
        "n_folds": len(folds),
        "cv_indices": {
            "source": "train+val" if training_cfg.get("combine_train_val_for_cv", True) else "train",
            "count": int(len(cv_indices)),
        },
        "final_fit": {
            "train_count": int(len(train_idx if limit is None else train_idx[:limit])),
            "val_count": int(len(val_idx if limit is None else val_idx[: max(1, min(len(val_idx), limit // 5))])),
            "test_count": int(len(test_idx if limit is None else test_idx[: max(1, min(len(test_idx), limit // 5))])),
        },
        "kinds": {},
    }

    requested_kinds = ["classic", "neural"] if model_type == "both" else [model_type]
    for kind in requested_kinds:
        kind_cfg = training_cfg.get(kind, {})
        if not bool(kind_cfg.get("enabled", True)):
            summary["kinds"][kind] = {"status": "skipped", "reason": "disabled in config"}
            continue
        summary["kinds"][kind] = _run_kind_selection(
            kind,
            config,
            paths,
            folds,
            train_idx,
            val_idx,
            test_idx,
            output_dir,
            metric,
            seed,
            limit,
        )

    completed_finals = {
        kind: result["final"]
        for kind, result in summary["kinds"].items()
        if result.get("status") == "completed" and "final" in result
    }
    if completed_finals:
        summary["best_by_kind"] = {
            kind: {
                "model_path": final["model_path"],
                "score": _metric_score(final, metric),
                "metric": metric,
                "registry": final.get("best_registry", {}),
            }
            for kind, final in completed_finals.items()
        }
        ranked = sorted(
            completed_finals.items(),
            key=lambda item: _metric_score(item[1], metric),
            reverse=True,
        )
        summary["overall_best"] = {
            "kind": ranked[0][0],
            "model_path": ranked[0][1]["model_path"],
            "score": _metric_score(ranked[0][1], metric),
            "metric": metric,
        }

    save_json(_json_safe(summary), output_dir / "training_summary.json")
    return summary
