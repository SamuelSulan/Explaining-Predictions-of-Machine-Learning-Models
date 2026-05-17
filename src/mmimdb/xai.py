"""XAI utilities for classic and neural multimodal MM-IMDb models."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import sparse

from mmimdb.constants import GENRE_LABELS
from mmimdb.data import load_metadata
from mmimdb.evaluation import predicted_label_indices, threshold_to_serializable
from mmimdb.image_utils import image_descriptor, restore_poster_rgb
from mmimdb.models.classic import load_reconstructed_texts
from mmimdb.models.neural import MMIMDBTorchDataset, MultimodalGenreModel, NeuralConfig
from mmimdb.perf import PerfConfig, measured, summarize_measurements
from mmimdb.text_utils import build_embedding_matrix, pad_token_id, sequence_to_tokens
from mmimdb.utils import ensure_dir, resolve_path, save_json


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def probability_to_logit(probability: float) -> float:
    p = float(np.clip(probability, 1e-7, 1.0 - 1e-7))
    return float(np.log(p / (1.0 - p)))


def parse_genre_list(target_genre: str | None = None, target_genres: str | list[str] | None = None) -> list[str]:
    genres: list[str] = []
    if target_genre:
        genres.append(str(target_genre))
    if isinstance(target_genres, str):
        genres.extend([item.strip() for item in target_genres.split(",") if item.strip()])
    elif target_genres:
        genres.extend([str(item) for item in target_genres if str(item)])
    unknown = [genre for genre in genres if genre not in GENRE_LABELS]
    if unknown:
        raise ValueError(f"Unknown genre(s): {unknown}. Valid genres: {GENRE_LABELS}")
    return list(dict.fromkeys(genres))


def top_probability_dict(probabilities: np.ndarray, n: int = 5) -> dict[str, float]:
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    return {GENRE_LABELS[i]: float(probs[i]) for i in np.argsort(probs)[::-1][:n]}


def selected_probability_and_logit(
    probabilities: np.ndarray,
    logits: np.ndarray | None,
    target_indices: list[int] | np.ndarray,
) -> tuple[float, float]:
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    indices = [int(i) for i in target_indices]
    probability = float(np.mean(probs[indices]))
    if logits is not None:
        logits_arr = np.asarray(logits, dtype=np.float32).reshape(-1)
        logit = float(np.sum(logits_arr[indices]))
    else:
        logit = float(np.sum([probability_to_logit(float(probs[i])) for i in indices]))
    return probability, logit


def select_xai_target_indices(
    probabilities: np.ndarray,
    threshold: float | list[float] | np.ndarray,
    target_genre: str | None = None,
    target_genres: str | list[str] | None = None,
    target_policy: str = "top",
    target_top_k: int = 1,
    max_targets_per_sample: int | None = None,
    ensure_at_least_one_target: bool = True,
) -> dict:
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    explicit_genres = parse_genre_list(target_genre, target_genres)
    if explicit_genres:
        policy = "explicit"
        indices = [GENRE_LABELS.index(genre) for genre in explicit_genres]
    else:
        policy = str(target_policy or "top").lower().replace("-", "_")
        if policy in {"top", "top_1", "argmax"}:
            indices = [int(np.argmax(probs))]
        elif policy in {"top_k", "topk"}:
            k = max(1, min(int(target_top_k), len(GENRE_LABELS)))
            indices = [int(i) for i in np.argsort(probs)[::-1][:k]]
        elif policy in {"predicted", "threshold", "thresholded"}:
            indices = predicted_label_indices(probs, threshold, ensure_at_least_one=ensure_at_least_one_target)
        elif policy == "all":
            indices = list(range(len(GENRE_LABELS)))
        else:
            raise ValueError(f"Unsupported XAI target policy: {target_policy}")

    indices = sorted(list(dict.fromkeys(int(i) for i in indices)), key=lambda i: float(probs[i]), reverse=True)
    predicted_indices = predicted_label_indices(probs, threshold, ensure_at_least_one=ensure_at_least_one_target)
    all_selected_indices = list(indices)
    if max_targets_per_sample is not None and int(max_targets_per_sample) > 0:
        indices = indices[: int(max_targets_per_sample)]
    return {
        "policy": policy,
        "target_indices": indices,
        "target_genres": [GENRE_LABELS[i] for i in indices],
        "all_selected_indices": all_selected_indices,
        "all_selected_genres": [GENRE_LABELS[i] for i in all_selected_indices],
        "predicted_indices": predicted_indices,
        "predicted_genres": [GENRE_LABELS[i] for i in predicted_indices],
        "threshold": threshold_to_serializable(threshold),
        "max_targets_per_sample": max_targets_per_sample,
    }


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - np.nanmin(arr)
    denom = np.nanmax(arr)
    if denom > 0:
        arr = arr / denom
    return np.nan_to_num(arr)


def directory_output_stats(path: str | Path) -> dict:
    p = resolve_path(path)
    files = [f for f in p.rglob("*") if f.is_file()] if p.exists() else []
    total_bytes = sum(f.stat().st_size for f in files)
    by_suffix: dict[str, int] = {}
    for f in files:
        suffix = f.suffix.lower() or "<no_suffix>"
        by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
    return {
        "file_count": len(files),
        "total_bytes": int(total_bytes),
        "total_mb": float(total_bytes / (1024 * 1024)),
        "file_count_by_suffix": by_suffix,
    }


def save_heatmap_overlay(rgb: np.ndarray, heatmap: np.ndarray, output_path: str | Path, alpha: float = 0.45) -> None:
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(rgb, dtype=np.uint8)
    heatmap = normalize_map(heatmap)
    hm_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        (rgb.shape[1], rgb.shape[0]),
        Image.Resampling.BILINEAR,
    )
    hm = np.asarray(hm_img, dtype=np.float32) / 255.0
    cmap = plt.get_cmap("jet")
    colored = (cmap(hm)[:, :, :3] * 255).astype(np.uint8)
    overlay = np.clip((1.0 - alpha) * rgb + alpha * colored, 0, 255).astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(output_path)


def save_token_bar(tokens: list[str], scores: list[float], output_path: str | Path, title: str) -> None:
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not tokens:
        return
    order = np.argsort(np.abs(scores))
    tokens_sorted = [tokens[i] for i in order]
    scores_sorted = [scores[i] for i in order]
    plt.figure(figsize=(8, max(3, len(tokens) * 0.35)))
    colors = ["#D95F02" if s < 0 else "#1B9E77" for s in scores_sorted]
    plt.barh(tokens_sorted, scores_sorted, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def two_modality_shapley(values: dict[str, float]) -> dict[str, float]:
    """Compute exact two-player Shapley values from modality coalitions."""
    blank = float(values["blank"])
    text_only = float(values["text_only"])
    image_only = float(values["image_only"])
    both = float(values["both"])
    text_value = 0.5 * ((text_only - blank) + (both - image_only))
    image_value = 0.5 * ((image_only - blank) + (both - text_only))
    interaction = both - text_only - image_only + blank
    abs_total = abs(text_value) + abs(image_value)
    if abs_total > 0:
        text_utilization = abs(text_value) / abs_total
        image_utilization = abs(image_value) / abs_total
    else:
        text_utilization = 0.0
        image_utilization = 0.0
    return {
        "text": float(text_value),
        "image": float(image_value),
        "interaction": float(interaction),
        "blank_to_both_delta": float(both - blank),
        "abs_total": float(abs_total),
        "text_utilization": float(text_utilization),
        "image_utilization": float(image_utilization),
    }


def build_modality_shapley(probability_values: dict[str, float], logit_values: dict[str, float]) -> dict:
    return {
        "method": "two_modality_shapley",
        "probability": {
            "values": {k: float(v) for k, v in probability_values.items()},
            "shapley": two_modality_shapley(probability_values),
        },
        "logit": {
            "values": {k: float(v) for k, v in logit_values.items()},
            "shapley": two_modality_shapley(logit_values),
        },
    }


def summarize_modality_shapley(explanations: list[dict]) -> dict:
    rows = []
    for explanation in explanations:
        shapley = explanation.get("modality_shapley", {})
        logit_shapley = shapley.get("logit", {}).get("shapley")
        probability_shapley = shapley.get("probability", {}).get("shapley")
        if logit_shapley and probability_shapley:
            rows.append((logit_shapley, probability_shapley))

    if not rows:
        return {"count": 0}

    def mean_field(index: int, field: str) -> float:
        return float(np.mean([row[index][field] for row in rows]))

    return {
        "count": int(len(rows)),
        "mean_logit_text": mean_field(0, "text"),
        "mean_logit_image": mean_field(0, "image"),
        "mean_logit_interaction": mean_field(0, "interaction"),
        "mean_logit_text_utilization": mean_field(0, "text_utilization"),
        "mean_logit_image_utilization": mean_field(0, "image_utilization"),
        "mean_probability_text": mean_field(1, "text"),
        "mean_probability_image": mean_field(1, "image"),
        "mean_probability_interaction": mean_field(1, "interaction"),
        "mean_probability_text_utilization": mean_field(1, "text_utilization"),
        "mean_probability_image_utilization": mean_field(1, "image_utilization"),
    }


def save_modality_shapley_bar(modality_shapley: dict, output_path: str | Path, title: str) -> None:
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shapley = modality_shapley["logit"]["shapley"]
    labels = ["text", "image", "interaction"]
    values = [shapley["text"], shapley["image"], shapley["interaction"]]
    colors = ["#1B9E77" if v >= 0 else "#D95F02" for v in values]
    plt.figure(figsize=(6, 3.2))
    plt.bar(labels, values, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("logit contribution")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def image_descriptor_names(hist_bins: int = 16, thumbnail_size: tuple[int, int] = (32, 20)) -> list[str]:
    regions = ["global", "top_left", "top_right", "bottom_left", "bottom_right"]
    channels = ["R", "G", "B"]
    names = []
    for region in regions:
        for channel in channels:
            for bin_idx in range(hist_bins):
                names.append(f"{region}_{channel}_hist_bin_{bin_idx}")
    width, height = thumbnail_size
    for y in range(height):
        for x in range(width):
            names.append(f"thumbnail_y{y}_x{x}")
    return names


def classic_target_probability_and_logit(classifier, x, target_idx: int) -> tuple[float, float]:
    probability = float(np.asarray(classifier.predict_proba(x))[0, target_idx])
    try:
        decision = np.asarray(classifier.decision_function(x))
        if decision.ndim == 1:
            logit = float(decision[target_idx])
        else:
            logit = float(decision[0, target_idx])
    except Exception:
        logit = probability_to_logit(probability)
    return probability, logit


def classic_all_probabilities_and_logits(classifier, x) -> tuple[np.ndarray, np.ndarray | None]:
    probabilities = np.asarray(classifier.predict_proba(x), dtype=np.float32)[0]
    try:
        decision = np.asarray(classifier.decision_function(x), dtype=np.float32)
        logits = decision if decision.ndim == 1 else decision[0]
    except Exception:
        logits = None
    return probabilities, logits


def classic_selected_probability_and_logit(classifier, x, target_indices: list[int]) -> tuple[float, float]:
    probabilities, logits = classic_all_probabilities_and_logits(classifier, x)
    return selected_probability_and_logit(probabilities, logits, target_indices)


def classic_classifier_estimator_and_features(classifier, x, target_idx: int):
    if not hasattr(classifier, "order_"):
        return classifier.estimators_[target_idx], x, []

    order = [int(i) for i in classifier.order_]
    chain_pos = order.index(int(target_idx))
    x_aug = x
    dependency_features = []
    chain_method = getattr(classifier, "chain_method_", "predict")
    for pos in range(chain_pos):
        estimator = classifier.estimators_[pos]
        if chain_method == "predict_proba" and hasattr(estimator, "predict_proba"):
            value = float(np.asarray(estimator.predict_proba(x_aug))[0, -1])
        elif chain_method == "decision_function" and hasattr(estimator, "decision_function"):
            value = float(np.asarray(estimator.decision_function(x_aug)).reshape(-1)[0])
        else:
            value = float(np.asarray(estimator.predict(x_aug)).reshape(-1)[0])
        dependency_features.append(
            {
                "genre": GENRE_LABELS[order[pos]],
                "value": value,
            }
        )
        x_aug = sparse.hstack([x_aug, sparse.csr_matrix([[value]], dtype=x_aug.dtype)], format="csr")
    return classifier.estimators_[chain_pos], x_aug, dependency_features


def classic_modality_shapley(classifier, x_text, desc_scaled: np.ndarray, target_idx: int) -> dict:
    zero_text = sparse.csr_matrix(x_text.shape, dtype=x_text.dtype)
    zero_image = sparse.csr_matrix(desc_scaled.shape, dtype=desc_scaled.dtype)
    image_block = sparse.csr_matrix(desc_scaled)
    variants = {
        "blank": sparse.hstack([zero_text, zero_image], format="csr"),
        "text_only": sparse.hstack([x_text, zero_image], format="csr"),
        "image_only": sparse.hstack([zero_text, image_block], format="csr"),
        "both": sparse.hstack([x_text, image_block], format="csr"),
    }
    probability_values = {}
    logit_values = {}
    for name, variant in variants.items():
        probability, logit = classic_target_probability_and_logit(classifier, variant, target_idx)
        probability_values[name] = probability
        logit_values[name] = logit
    return build_modality_shapley(probability_values, logit_values)


def classic_set_modality_shapley(classifier, x_text, desc_scaled: np.ndarray, target_indices: list[int]) -> dict:
    zero_text = sparse.csr_matrix(x_text.shape, dtype=x_text.dtype)
    zero_image = sparse.csr_matrix(desc_scaled.shape, dtype=desc_scaled.dtype)
    image_block = sparse.csr_matrix(desc_scaled)
    variants = {
        "blank": sparse.hstack([zero_text, zero_image], format="csr"),
        "text_only": sparse.hstack([x_text, zero_image], format="csr"),
        "image_only": sparse.hstack([zero_text, image_block], format="csr"),
        "both": sparse.hstack([x_text, image_block], format="csr"),
    }
    probability_values = {}
    logit_values = {}
    for name, variant in variants.items():
        probability, logit = classic_selected_probability_and_logit(classifier, variant, target_indices)
        probability_values[name] = probability
        logit_values[name] = logit
    return build_modality_shapley(probability_values, logit_values)


def patch_classic_classifier_compat(classifier) -> None:
    """Repair known sklearn pickle drift for older LogisticRegression loaders."""
    for estimator in getattr(classifier, "estimators_", []):
        if estimator.__class__.__name__ == "LogisticRegression" and not hasattr(estimator, "multi_class"):
            estimator.multi_class = "ovr"


def explain_classic_samples(
    model_path: str | Path,
    hdf5_path: str | Path,
    metadata_path: str | Path,
    indices: list[int] | np.ndarray,
    output_dir: str | Path,
    target_genre: str | None = None,
    target_genres: str | list[str] | None = None,
    target_policy: str = "top",
    target_top_k: int = 1,
    max_targets_per_sample: int | None = None,
    ensure_at_least_one_target: bool = True,
    top_k: int = 12,
    measure_performance: bool = True,
    performance_sample_interval_seconds: float = 0.02,
    enable_modality_shapley: bool = True,
    enable_experimental_set_explanation: bool = False,
) -> dict:
    """Explain a saved classic multimodal model with linear contributions."""
    artifact = joblib.load(resolve_path(model_path))
    classifier = artifact["classifier"]
    patch_classic_classifier_compat(classifier)
    vectorizer = artifact["vectorizer"]
    image_scaler = artifact["image_scaler"]
    cfg = artifact.get("config", {})
    hist_bins = int(cfg.get("image_hist_bins", 16))
    thumbnail_size = tuple(cfg.get("image_thumbnail_size", (32, 20)))
    feature_names = vectorizer.get_feature_names_out()
    img_feature_names = image_descriptor_names(hist_bins, thumbnail_size)
    out = ensure_dir(output_dir)

    metadata = load_metadata(metadata_path)
    threshold = artifact.get("threshold", 0.5)
    explanations = []
    all_measurements = []
    perf_cfg = PerfConfig(
        enabled=measure_performance,
        sample_interval_seconds=performance_sample_interval_seconds,
    )

    with h5py.File(resolve_path(hdf5_path), "r") as f:
        y = f["genres"][:].astype(np.int8)
        for raw_idx in indices:
            idx = int(raw_idx)
            text = load_reconstructed_texts(hdf5_path, metadata_path, np.asarray([idx]))[0]
            x_text = vectorizer.transform([text])
            rgb = restore_poster_rgb(f["images"][idx])
            desc = image_descriptor(
                f["images"][idx],
                hist_bins=hist_bins,
                thumbnail_size=thumbnail_size,
            ).reshape(1, -1)
            desc_scaled = image_scaler.transform(desc)
            x = sparse.hstack([x_text, sparse.csr_matrix(desc_scaled)], format="csr")
            probs = np.asarray(classifier.predict_proba(x))[0]
            selection = select_xai_target_indices(
                probs,
                threshold,
                target_genre=target_genre,
                target_genres=target_genres,
                target_policy=target_policy,
                target_top_k=target_top_k,
                max_targets_per_sample=max_targets_per_sample,
                ensure_at_least_one_target=ensure_at_least_one_target,
            )
            img_values = desc_scaled.reshape(-1)
            original_feature_count = int(len(feature_names) + img_values.size)
            true_genres = [GENRE_LABELS[i] for i in np.flatnonzero(y[idx])]

            for target_idx in selection["target_indices"]:
                target_label = GENRE_LABELS[target_idx]
                estimator, _x_estimator, dependency_features = classic_classifier_estimator_and_features(
                    classifier,
                    x,
                    target_idx,
                )
                sample_dir = ensure_dir(out / f"classic_idx_{idx}_{target_label}")
                Image.fromarray(rgb, mode="RGB").save(sample_dir / "poster.png")

                explanation: dict[str, Any] = {
                    "index": idx,
                    "target_genre": target_label,
                    "target_genres": [target_label],
                    "true_genres": true_genres,
                    "predicted_genres": selection["predicted_genres"],
                    "probability": float(probs[target_idx]),
                    "top_probabilities": top_probability_dict(probs),
                    "target_selection": selection,
                    "method": "linear_feature_contribution",
                    "methods": [],
                }

                method_measurements = []
                if enable_modality_shapley:
                    with measured(f"classic_modality_shapley_idx_{idx}_{target_label}", perf_cfg) as m:
                        modality_shapley = classic_modality_shapley(classifier, x_text, desc_scaled, target_idx)
                    method_measurements.append(m.metrics)
                    all_measurements.append(m.metrics)

                    with measured(f"classic_modality_shapley_visualization_idx_{idx}_{target_label}", perf_cfg) as m:
                        save_modality_shapley_bar(
                            modality_shapley,
                            sample_dir / "modality_shapley.png",
                            f"Classic modality Shapley: {target_label}",
                        )
                    method_measurements.append(m.metrics)
                    all_measurements.append(m.metrics)
                    explanation["methods"].append("modality_shapley")
                    explanation["modality_shapley"] = modality_shapley

                if hasattr(estimator, "coef_"):
                    with measured(f"classic_linear_contribution_idx_{idx}_{target_label}", perf_cfg) as m:
                        coef = estimator.coef_.reshape(-1)
                        intercept = float(getattr(estimator, "intercept_", np.asarray([0.0]))[0])
                        text_coef = coef[: len(feature_names)]
                        image_coef = coef[len(feature_names) : original_feature_count]
                        dependency_coef = coef[original_feature_count:]

                        text_row = x_text.tocoo()
                        text_contribs = text_row.data * text_coef[text_row.col]
                        text_order = np.argsort(np.abs(text_contribs))[::-1][:top_k]
                        top_text = [
                            {
                                "feature": str(feature_names[text_row.col[i]]),
                                "tfidf": float(text_row.data[i]),
                                "contribution": float(text_contribs[i]),
                            }
                            for i in text_order
                        ]

                        img_contribs = img_values * image_coef
                        img_order = np.argsort(np.abs(img_contribs))[::-1][:top_k]
                        top_image = [
                            {
                                "feature": str(img_feature_names[i]) if i < len(img_feature_names) else f"image_{i}",
                                "value": float(img_values[i]),
                                "contribution": float(img_contribs[i]),
                            }
                            for i in img_order
                        ]
                        label_dependency_contributions = [
                            {
                                "genre": item["genre"],
                                "value": float(item["value"]),
                                "coefficient": float(dependency_coef[i]),
                                "contribution": float(item["value"] * dependency_coef[i]),
                            }
                            for i, item in enumerate(dependency_features[: len(dependency_coef)])
                        ]
                    method_measurements.append(m.metrics)
                    all_measurements.append(m.metrics)

                    with measured(f"classic_visualization_idx_{idx}_{target_label}", perf_cfg) as m:
                        hist_len = 5 * 3 * hist_bins
                        thumb_contrib = img_contribs[hist_len:]
                        if thumb_contrib.size == thumbnail_size[0] * thumbnail_size[1]:
                            thumb_heatmap = np.abs(thumb_contrib).reshape(thumbnail_size[1], thumbnail_size[0])
                            save_heatmap_overlay(
                                rgb,
                                thumb_heatmap,
                                sample_dir / "thumbnail_descriptor_heatmap.png",
                            )
                        save_token_bar(
                            [item["feature"] for item in top_text],
                            [item["contribution"] for item in top_text],
                            sample_dir / "top_text_features.png",
                            f"Classic text contributions: {target_label}",
                        )
                    method_measurements.append(m.metrics)
                    all_measurements.append(m.metrics)

                    explanation.update(
                        {
                            "intercept": intercept,
                            "text_logit_contribution": float(text_contribs.sum()),
                            "image_logit_contribution": float(img_contribs.sum()),
                            "label_dependency_logit_contribution": float(
                                sum(item["contribution"] for item in label_dependency_contributions)
                            ),
                            "top_text_features": top_text,
                            "top_image_features": top_image,
                            "label_dependency_contributions": label_dependency_contributions,
                        }
                    )
                    explanation["methods"].insert(0, "linear_feature_contribution")
                    if label_dependency_contributions:
                        explanation["methods"].append("classifier_chain_label_dependency")
                else:
                    explanation["warning"] = "Estimator has no coefficients, likely because the label was constant in a tiny smoke-test split."

                explanation["performance"] = {
                    "measurements": method_measurements,
                    "summary": summarize_measurements(method_measurements),
                }
                save_json(explanation, sample_dir / "explanation.json")
                explanations.append(explanation)

            if enable_experimental_set_explanation and len(selection["target_indices"]) > 1:
                target_indices = [int(i) for i in selection["target_indices"]]
                target_labels = [GENRE_LABELS[i] for i in target_indices]
                sample_dir = ensure_dir(out / f"classic_idx_{idx}_label_set")
                Image.fromarray(rgb, mode="RGB").save(sample_dir / "poster.png")
                set_probability, set_logit = classic_selected_probability_and_logit(classifier, x, target_indices)
                method_measurements = []
                methods = ["experimental_set_linear_feature_contribution"]

                with measured(f"classic_set_linear_contribution_idx_{idx}", perf_cfg) as m:
                    coef_sum = np.zeros(original_feature_count, dtype=np.float32)
                    intercept_sum = 0.0
                    label_dependency_contributions = []
                    for target_idx in target_indices:
                        estimator, _x_estimator, dependency_features = classic_classifier_estimator_and_features(
                            classifier,
                            x,
                            target_idx,
                        )
                        if not hasattr(estimator, "coef_"):
                            continue
                        coef = estimator.coef_.reshape(-1)
                        coef_sum += coef[:original_feature_count]
                        intercept_sum += float(getattr(estimator, "intercept_", np.asarray([0.0]))[0])
                        dependency_coef = coef[original_feature_count:]
                        for dep_i, item in enumerate(dependency_features[: len(dependency_coef)]):
                            label_dependency_contributions.append(
                                {
                                    "target_genre": GENRE_LABELS[target_idx],
                                    "genre": item["genre"],
                                    "value": float(item["value"]),
                                    "coefficient": float(dependency_coef[dep_i]),
                                    "contribution": float(item["value"] * dependency_coef[dep_i]),
                                }
                            )

                    text_coef = coef_sum[: len(feature_names)]
                    image_coef = coef_sum[len(feature_names) : original_feature_count]
                    text_row = x_text.tocoo()
                    text_contribs = text_row.data * text_coef[text_row.col]
                    text_order = np.argsort(np.abs(text_contribs))[::-1][:top_k]
                    top_text = [
                        {
                            "feature": str(feature_names[text_row.col[i]]),
                            "tfidf": float(text_row.data[i]),
                            "contribution": float(text_contribs[i]),
                        }
                        for i in text_order
                    ]
                    img_contribs = img_values * image_coef
                    img_order = np.argsort(np.abs(img_contribs))[::-1][:top_k]
                    top_image = [
                        {
                            "feature": str(img_feature_names[i]) if i < len(img_feature_names) else f"image_{i}",
                            "value": float(img_values[i]),
                            "contribution": float(img_contribs[i]),
                        }
                        for i in img_order
                    ]
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)

                if enable_modality_shapley:
                    with measured(f"classic_set_modality_shapley_idx_{idx}", perf_cfg) as m:
                        modality_shapley = classic_set_modality_shapley(classifier, x_text, desc_scaled, target_indices)
                    method_measurements.append(m.metrics)
                    all_measurements.append(m.metrics)
                    with measured(f"classic_set_modality_shapley_visualization_idx_{idx}", perf_cfg) as m:
                        save_modality_shapley_bar(
                            modality_shapley,
                            sample_dir / "modality_shapley.png",
                            "Classic modality Shapley: label set",
                        )
                    method_measurements.append(m.metrics)
                    all_measurements.append(m.metrics)
                    methods.append("modality_shapley")
                else:
                    modality_shapley = None

                with measured(f"classic_set_visualization_idx_{idx}", perf_cfg) as m:
                    hist_len = 5 * 3 * hist_bins
                    thumb_contrib = img_contribs[hist_len:]
                    if thumb_contrib.size == thumbnail_size[0] * thumbnail_size[1]:
                        thumb_heatmap = np.abs(thumb_contrib).reshape(thumbnail_size[1], thumbnail_size[0])
                        save_heatmap_overlay(rgb, thumb_heatmap, sample_dir / "thumbnail_descriptor_heatmap.png")
                    save_token_bar(
                        [item["feature"] for item in top_text],
                        [item["contribution"] for item in top_text],
                        sample_dir / "top_text_features.png",
                        "Classic text contributions: label set",
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)

                explanation = {
                    "index": idx,
                    "target_genre": "label_set",
                    "target_genres": target_labels,
                    "true_genres": true_genres,
                    "predicted_genres": selection["predicted_genres"],
                    "probability": set_probability,
                    "set_logit": set_logit,
                    "top_probabilities": top_probability_dict(probs),
                    "target_selection": selection,
                    "methods": methods,
                    "intercept": float(intercept_sum),
                    "text_logit_contribution": float(text_contribs.sum()),
                    "image_logit_contribution": float(img_contribs.sum()),
                    "label_dependency_logit_contribution": float(
                        sum(item["contribution"] for item in label_dependency_contributions)
                    ),
                    "top_text_features": top_text,
                    "top_image_features": top_image,
                    "label_dependency_contributions": label_dependency_contributions,
                    "performance": {
                        "measurements": method_measurements,
                        "summary": summarize_measurements(method_measurements),
                    },
                }
                if modality_shapley is not None:
                    explanation["modality_shapley"] = modality_shapley
                save_json(explanation, sample_dir / "explanation.json")
                explanations.append(explanation)

    summary = {
        "model_path": str(resolve_path(model_path)),
        "target_policy": target_policy,
        "target_top_k": int(target_top_k),
        "max_targets_per_sample": max_targets_per_sample,
        "threshold": threshold_to_serializable(threshold),
        "samples": explanations,
        "modality_shapley_summary": summarize_modality_shapley(explanations),
        "performance": {
            "measurements": all_measurements,
            "summary": summarize_measurements(all_measurements),
        },
        "output_stats": directory_output_stats(out),
    }
    save_json(summary, out / "classic_xai_summary.json")
    return summary


def load_neural_model(checkpoint_path: str | Path, metadata_path: str | Path, device: str | None = None):
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_path(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    cfg_dict = dict(checkpoint["config"])
    cfg_dict["text_cnn_kernels"] = tuple(cfg_dict.get("text_cnn_kernels", (3, 4, 5)))
    cfg_dict["pretrained_image"] = False
    cfg = NeuralConfig(**cfg_dict)
    metadata = load_metadata(metadata_path)
    embedding_matrix = build_embedding_matrix(metadata)
    pad_id = int(checkpoint.get("pad_id", pad_token_id(metadata)))
    model = MultimodalGenreModel(
        embedding_matrix=embedding_matrix,
        pad_id=pad_id,
        num_labels=len(GENRE_LABELS),
        cfg=cfg,
    ).module
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, metadata, checkpoint, device


def denormalize_model_image(image_tensor) -> np.ndarray:
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    img = image_tensor.detach() * std + mean
    img = img.clamp(0, 1).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return (img * 255).astype(np.uint8)


def compute_gradcam(model, tokens, mask, image, target_idx: int | list[int]) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if not hasattr(model.image_encoder, "layer4"):
        raise ValueError("Grad-CAM currently supports ResNet-style image encoders with layer4.")

    activations = {}
    gradients = {}
    layer = model.image_encoder.layer4[-1]

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_fwd = layer.register_forward_hook(forward_hook)
    handle_bwd = layer.register_full_backward_hook(backward_hook)
    try:
        image_for_grad = image.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(tokens, mask, image_for_grad)
        if isinstance(target_idx, list):
            score = logits[:, target_idx].sum()
        else:
            score = logits[:, int(target_idx)].sum()
        score.backward()

        acts = activations["value"]
        grads = gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return normalize_map(cam)
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def neural_logits_from_features(model, text_h, image_h):
    import torch

    if model.fusion == "gmu":
        text_z = torch.tanh(model.text_gate_proj(text_h))
        image_z = torch.tanh(model.image_gate_proj(image_h))
        gate_value = torch.sigmoid(model.gate(torch.cat([text_h, image_h], dim=1)))
        fused = gate_value * text_z + (1.0 - gate_value) * image_z
    else:
        fused = torch.cat([text_h, image_h], dim=1)
    logits = model.classifier(fused)
    if getattr(model, "enable_label_correlation", False):
        logits = logits + model.label_correlation(logits)
    return logits


def torch_selected_probability_and_logit(logits, target_idx: int | list[int]) -> tuple[float, float]:
    import torch

    if isinstance(target_idx, list):
        indices = [int(i) for i in target_idx]
        selected_logits = logits[0, indices]
        selected_probs = torch.sigmoid(logits)[0, indices]
        return float(selected_probs.mean().detach().cpu().item()), float(selected_logits.sum().detach().cpu().item())
    return (
        float(torch.sigmoid(logits)[0, int(target_idx)].detach().cpu().item()),
        float(logits[0, int(target_idx)].detach().cpu().item()),
    )


def neural_modality_shapley(model, tokens, mask, image, target_idx: int | list[int], pad_id: int) -> dict:
    import torch

    pad_tokens = torch.full_like(tokens, fill_value=int(pad_id))
    zero_image = torch.zeros_like(image)
    variants = {
        "blank": (pad_tokens, zero_image),
        "text_only": (tokens, zero_image),
        "image_only": (pad_tokens, image),
        "both": (tokens, image),
    }
    probability_values = {}
    logit_values = {}
    with torch.no_grad():
        for name, (tokens_use, image_use) in variants.items():
            logits = model(tokens_use, mask, image_use)
            probability, logit = torch_selected_probability_and_logit(logits, target_idx)
            probability_values[name] = probability
            logit_values[name] = logit
    return build_modality_shapley(probability_values, logit_values)


def neural_token_occlusion(
    model,
    tokens,
    mask,
    image,
    target_idx: int,
    pad_id: int,
    raw_tokens: list[str],
    batch_size: int = 64,
) -> dict:
    import torch

    real_len = min(int(mask.sum().detach().cpu().item()), int(tokens.shape[1]), len(raw_tokens))
    if real_len <= 0:
        return {
            "baseline_probability": 0.0,
            "baseline_logit": 0.0,
            "top_tokens": [],
            "all_scores": [],
        }

    batch_size = max(1, int(batch_size))
    with torch.no_grad():
        text_h = model.text_encoder(tokens, mask)
        image_h = model.image_proj(model.image_encoder(image))
        baseline_logits = neural_logits_from_features(model, text_h, image_h)
        baseline_logit = float(baseline_logits[0, target_idx].detach().cpu().item())
        baseline_probability = float(torch.sigmoid(baseline_logits)[0, target_idx].detach().cpu().item())

        rows = []
        positions = list(range(real_len))
        for start in range(0, real_len, batch_size):
            batch_positions = positions[start : start + batch_size]
            token_batch = tokens.repeat(len(batch_positions), 1)
            mask_batch = mask.repeat(len(batch_positions), 1)
            for row_i, pos in enumerate(batch_positions):
                token_batch[row_i, pos] = int(pad_id)
            text_h_batch = model.text_encoder(token_batch, mask_batch)
            image_h_batch = image_h.expand(len(batch_positions), -1)
            logits = neural_logits_from_features(model, text_h_batch, image_h_batch)
            probs = torch.sigmoid(logits)[:, target_idx].detach().cpu().numpy()
            logits_np = logits[:, target_idx].detach().cpu().numpy()
            for pos, probability, logit in zip(batch_positions, probs, logits_np):
                rows.append(
                    {
                        "position": int(pos),
                        "token": str(raw_tokens[pos]),
                        "occluded_probability": float(probability),
                        "probability_delta": float(baseline_probability - probability),
                        "occluded_logit": float(logit),
                        "logit_delta": float(baseline_logit - logit),
                    }
                )

    rows_sorted = sorted(rows, key=lambda item: abs(item["probability_delta"]), reverse=True)
    return {
        "baseline_probability": baseline_probability,
        "baseline_logit": baseline_logit,
        "top_tokens": rows_sorted,
        "all_scores": rows,
    }


def neural_image_occlusion(
    model,
    tokens,
    mask,
    image,
    target_idx: int,
    grid_size: int = 4,
    batch_size: int = 16,
) -> dict:
    import torch

    grid_size = max(1, int(grid_size))
    batch_size = max(1, int(batch_size))
    _, _, height, width = image.shape

    with torch.no_grad():
        text_h = model.text_encoder(tokens, mask)
        image_h = model.image_proj(model.image_encoder(image))
        baseline_logits = neural_logits_from_features(model, text_h, image_h)
        baseline_logit = float(baseline_logits[0, target_idx].detach().cpu().item())
        baseline_probability = float(torch.sigmoid(baseline_logits)[0, target_idx].detach().cpu().item())

        cells = []
        for row in range(grid_size):
            y0 = int(round(row * height / grid_size))
            y1 = int(round((row + 1) * height / grid_size))
            for col in range(grid_size):
                x0 = int(round(col * width / grid_size))
                x1 = int(round((col + 1) * width / grid_size))
                cells.append((row, col, y0, y1, x0, x1))

        rows = []
        heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
        for start in range(0, len(cells), batch_size):
            batch_cells = cells[start : start + batch_size]
            image_batch = image.repeat(len(batch_cells), 1, 1, 1)
            for row_i, (_row, _col, y0, y1, x0, x1) in enumerate(batch_cells):
                image_batch[row_i, :, y0:y1, x0:x1] = 0.0
            image_h_batch = model.image_proj(model.image_encoder(image_batch))
            text_h_batch = text_h.expand(len(batch_cells), -1)
            logits = neural_logits_from_features(model, text_h_batch, image_h_batch)
            probs = torch.sigmoid(logits)[:, target_idx].detach().cpu().numpy()
            logits_np = logits[:, target_idx].detach().cpu().numpy()
            for cell, probability, logit in zip(batch_cells, probs, logits_np):
                row, col, y0, y1, x0, x1 = cell
                probability_delta = float(baseline_probability - probability)
                logit_delta = float(baseline_logit - logit)
                heatmap[row, col] = probability_delta
                rows.append(
                    {
                        "row": int(row),
                        "col": int(col),
                        "y0": int(y0),
                        "y1": int(y1),
                        "x0": int(x0),
                        "x1": int(x1),
                        "occluded_probability": float(probability),
                        "probability_delta": probability_delta,
                        "occluded_logit": float(logit),
                        "logit_delta": logit_delta,
                    }
                )

    rows_sorted = sorted(rows, key=lambda item: abs(item["probability_delta"]), reverse=True)
    return {
        "baseline_probability": baseline_probability,
        "baseline_logit": baseline_logit,
        "grid_size": int(grid_size),
        "top_patches": rows_sorted,
        "heatmap": heatmap.tolist(),
    }


def neural_modality_ablation(model, tokens, mask, image, target_idx: int | list[int], pad_id: int) -> dict[str, float]:
    import torch

    pad_tokens = torch.full_like(tokens, fill_value=int(pad_id))
    zero_image = torch.zeros_like(image)
    with torch.no_grad():
        both, both_logit = torch_selected_probability_and_logit(model(tokens, mask, image), target_idx)
        no_text, no_text_logit = torch_selected_probability_and_logit(model(pad_tokens, mask, image), target_idx)
        no_image, no_image_logit = torch_selected_probability_and_logit(model(tokens, mask, zero_image), target_idx)
        blank, blank_logit = torch_selected_probability_and_logit(model(pad_tokens, mask, zero_image), target_idx)
    return {
        "both": float(both),
        "no_text": float(no_text),
        "no_image": float(no_image),
        "blank": float(blank),
        "text_delta": float(both - no_text),
        "image_delta": float(both - no_image),
        "both_logit": float(both_logit),
        "no_text_logit": float(no_text_logit),
        "no_image_logit": float(no_image_logit),
        "blank_logit": float(blank_logit),
    }


def explain_neural_samples(
    checkpoint_path: str | Path,
    hdf5_path: str | Path,
    metadata_path: str | Path,
    indices: list[int] | np.ndarray,
    output_dir: str | Path,
    target_genre: str | None = None,
    target_genres: str | list[str] | None = None,
    target_policy: str = "top",
    target_top_k: int = 1,
    max_targets_per_sample: int | None = None,
    ensure_at_least_one_target: bool = True,
    top_k: int = 12,
    n_steps: int = 8,
    measure_performance: bool = True,
    performance_sample_interval_seconds: float = 0.02,
    enable_layer_integrated_gradients_text: bool = True,
    enable_integrated_gradients_image: bool = True,
    enable_gradcam: bool = True,
    enable_modality_ablation: bool = True,
    enable_modality_shapley: bool = True,
    enable_token_occlusion: bool = True,
    enable_image_occlusion: bool = True,
    image_occlusion_grid: int = 4,
    occlusion_batch_size: int = 32,
    enable_experimental_set_explanation: bool = False,
) -> dict:
    """Explain a saved neural multimodal model using IG, Grad-CAM, and ablation."""
    import torch
    from captum.attr import IntegratedGradients, LayerIntegratedGradients

    model, cfg, metadata, checkpoint, device = load_neural_model(checkpoint_path, metadata_path)
    out = ensure_dir(output_dir)
    dataset = MMIMDBTorchDataset(
        hdf5_path,
        np.asarray(indices, dtype=np.int64),
        vocab_size=int(metadata["vocab_size"]),
        max_length=cfg.max_length,
        input_size=cfg.input_size,
    )
    threshold = checkpoint.get("threshold", 0.5)
    perf_cfg = PerfConfig(
        enabled=measure_performance,
        sample_interval_seconds=performance_sample_interval_seconds,
    )
    all_measurements = []

    def forward_for_text(tokens, mask, image):
        batch_size = tokens.shape[0]
        if mask.shape[0] != batch_size:
            mask = mask.expand(batch_size, -1)
        if image.shape[0] != batch_size:
            image = image.expand(batch_size, -1, -1, -1)
        return model(tokens, mask, image)

    lig = LayerIntegratedGradients(forward_for_text, model.text_encoder.embedding)
    explanations = []
    y = None
    with h5py.File(resolve_path(hdf5_path), "r") as f:
        y = f["genres"][:].astype(np.int8)

    for local_i, raw_idx in enumerate(indices):
        idx = int(raw_idx)
        item = dataset[local_i]
        tokens = item["tokens"].unsqueeze(0).to(device)
        mask = item["mask"].unsqueeze(0).to(device)
        image = item["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits, gate = model(tokens, mask, image, return_gate=True)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
            logits_np = logits.detach().cpu().numpy()[0]
        selection = select_xai_target_indices(
            probs,
            threshold,
            target_genre=target_genre,
            target_genres=target_genres,
            target_policy=target_policy,
            target_top_k=target_top_k,
            max_targets_per_sample=max_targets_per_sample,
            ensure_at_least_one_target=ensure_at_least_one_target,
        )
        model_rgb = denormalize_model_image(image[0])

        pad_id_value = int(checkpoint.get("pad_id", pad_token_id(metadata)))
        baseline_tokens = torch.full_like(tokens, fill_value=pad_id_value)
        real_len = int(mask.sum().detach().cpu().item())
        with h5py.File(resolve_path(hdf5_path), "r") as f:
            raw_tokens = sequence_to_tokens(f["sequences"][idx][:real_len], metadata["ix_to_word"])
        true_genres = [GENRE_LABELS[i] for i in np.flatnonzero(y[idx])]

        for target_idx in selection["target_indices"]:
            target_label = GENRE_LABELS[target_idx]
            sample_dir = ensure_dir(out / f"neural_idx_{idx}_{target_label}")
            Image.fromarray(model_rgb, mode="RGB").save(sample_dir / "model_input_poster.png")

            method_measurements = []
            methods = []
            top_tokens = []
            if enable_layer_integrated_gradients_text:
                with measured(f"neural_text_layer_integrated_gradients_idx_{idx}_{target_label}", perf_cfg) as m:
                    text_attr = lig.attribute(
                        inputs=tokens,
                        baselines=baseline_tokens,
                        additional_forward_args=(mask, image),
                        target=target_idx,
                        n_steps=n_steps,
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                token_scores = text_attr.sum(dim=-1).squeeze(0).detach().cpu().numpy()
                token_scores_real = token_scores[:real_len]
                token_order = np.argsort(np.abs(token_scores_real))[::-1][:top_k]
                top_tokens = [
                    {
                        "position": int(pos),
                        "token": str(raw_tokens[pos]),
                        "attribution": float(token_scores_real[pos]),
                    }
                    for pos in token_order
                ]
                with measured(f"neural_text_visualization_idx_{idx}_{target_label}", perf_cfg) as m:
                    save_token_bar(
                        [item["token"] for item in top_tokens],
                        [item["attribution"] for item in top_tokens],
                        sample_dir / "top_token_attributions.png",
                        f"Layer IG token attributions: {target_label}",
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                methods.append("LayerIntegratedGradients_text_embedding")

            token_occlusion = None
            if enable_token_occlusion:
                with measured(f"neural_token_occlusion_idx_{idx}_{target_label}", perf_cfg) as m:
                    token_occlusion = neural_token_occlusion(
                        model,
                        tokens,
                        mask,
                        image,
                        target_idx,
                        pad_id=pad_id_value,
                        raw_tokens=raw_tokens,
                        batch_size=occlusion_batch_size,
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                token_occlusion["top_tokens"] = token_occlusion["top_tokens"][:top_k]
                with measured(f"neural_token_occlusion_visualization_idx_{idx}_{target_label}", perf_cfg) as m:
                    save_token_bar(
                        [item["token"] for item in token_occlusion["top_tokens"]],
                        [item["probability_delta"] for item in token_occlusion["top_tokens"]],
                        sample_dir / "token_occlusion_attributions.png",
                        f"Token occlusion: {target_label}",
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                methods.append("token_occlusion")

            def image_forward(image_input):
                batch_size = image_input.shape[0]
                tokens_use = tokens.expand(batch_size, -1)
                mask_use = mask.expand(batch_size, -1)
                return model(tokens_use, mask_use, image_input)

            image_ig_heatmap = None
            if enable_integrated_gradients_image:
                ig = IntegratedGradients(image_forward)
                with measured(f"neural_image_integrated_gradients_idx_{idx}_{target_label}", perf_cfg) as m:
                    image_attr = ig.attribute(
                        inputs=image,
                        baselines=torch.zeros_like(image),
                        target=target_idx,
                        n_steps=n_steps,
                    )
                    image_ig_heatmap = image_attr.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                with measured(f"neural_image_ig_visualization_idx_{idx}_{target_label}", perf_cfg) as m:
                    save_heatmap_overlay(
                        model_rgb,
                        image_ig_heatmap,
                        sample_dir / "integrated_gradients_image_overlay.png",
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                methods.append("IntegratedGradients_image_pixels")

            gradcam_heatmap = None
            if enable_gradcam:
                with measured(f"neural_image_gradcam_idx_{idx}_{target_label}", perf_cfg) as m:
                    gradcam_heatmap = compute_gradcam(model, tokens, mask, image, target_idx)
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                with measured(f"neural_image_gradcam_visualization_idx_{idx}_{target_label}", perf_cfg) as m:
                    save_heatmap_overlay(model_rgb, gradcam_heatmap, sample_dir / "gradcam_overlay.png")
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                methods.append("GradCAM_image_branch")

            image_occlusion = None
            if enable_image_occlusion:
                with measured(f"neural_image_occlusion_idx_{idx}_{target_label}", perf_cfg) as m:
                    image_occlusion = neural_image_occlusion(
                        model,
                        tokens,
                        mask,
                        image,
                        target_idx,
                        grid_size=image_occlusion_grid,
                        batch_size=occlusion_batch_size,
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                image_occlusion["top_patches"] = image_occlusion["top_patches"][:top_k]
                with measured(f"neural_image_occlusion_visualization_idx_{idx}_{target_label}", perf_cfg) as m:
                    occlusion_heatmap = np.abs(np.asarray(image_occlusion["heatmap"], dtype=np.float32))
                    save_heatmap_overlay(
                        model_rgb,
                        occlusion_heatmap,
                        sample_dir / "image_occlusion_sensitivity_overlay.png",
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                methods.append("image_occlusion_sensitivity")

            modality_shapley = None
            if enable_modality_shapley:
                with measured(f"neural_modality_shapley_idx_{idx}_{target_label}", perf_cfg) as m:
                    modality_shapley = neural_modality_shapley(
                        model,
                        tokens,
                        mask,
                        image,
                        target_idx,
                        pad_id=pad_id_value,
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                with measured(f"neural_modality_shapley_visualization_idx_{idx}_{target_label}", perf_cfg) as m:
                    save_modality_shapley_bar(
                        modality_shapley,
                        sample_dir / "modality_shapley.png",
                        f"Neural modality Shapley: {target_label}",
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                methods.append("modality_shapley")

            ablation = None
            if enable_modality_ablation:
                with measured(f"neural_modality_ablation_idx_{idx}_{target_label}", perf_cfg) as m:
                    ablation = neural_modality_ablation(
                        model,
                        tokens,
                        mask,
                        image,
                        target_idx,
                        pad_id=pad_id_value,
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                methods.append("modality_ablation")
            if gate is not None:
                methods.append("GMU_gate_summary")
            if getattr(model, "enable_label_correlation", False):
                methods.append("experimental_label_correlation_head")

            explanation = {
                "index": idx,
                "target_genre": target_label,
                "target_genres": [target_label],
                "true_genres": true_genres,
                "predicted_genres": selection["predicted_genres"],
                "probability": float(probs[target_idx]),
                "top_probabilities": top_probability_dict(probs),
                "threshold": threshold_to_serializable(threshold),
                "target_selection": selection,
                "methods": methods,
                "top_tokens": top_tokens,
                "gate_mean": float(gate.mean().detach().cpu().item()) if gate is not None else None,
                "gate_std": float(gate.std().detach().cpu().item()) if gate is not None else None,
                "config": asdict(cfg),
                "performance": {
                    "measurements": method_measurements,
                    "summary": summarize_measurements(method_measurements),
                },
            }
            if token_occlusion is not None:
                explanation["token_occlusion"] = token_occlusion
            if image_occlusion is not None:
                explanation["image_occlusion"] = image_occlusion
            if modality_shapley is not None:
                explanation["modality_shapley"] = modality_shapley
            if ablation is not None:
                explanation["modality_ablation"] = ablation
            save_json(explanation, sample_dir / "explanation.json")
            explanations.append(explanation)

        if enable_experimental_set_explanation and len(selection["target_indices"]) > 1:
            target_indices = [int(i) for i in selection["target_indices"]]
            target_labels = [GENRE_LABELS[i] for i in target_indices]
            sample_dir = ensure_dir(out / f"neural_idx_{idx}_label_set")
            Image.fromarray(model_rgb, mode="RGB").save(sample_dir / "model_input_poster.png")
            set_probability, set_logit = selected_probability_and_logit(probs, logits_np, target_indices)
            method_measurements = []
            methods = ["experimental_set_level_attribution"]

            def forward_for_text_set(tokens_input, mask_input, image_input):
                batch_size = tokens_input.shape[0]
                if mask_input.shape[0] != batch_size:
                    mask_input = mask_input.expand(batch_size, -1)
                if image_input.shape[0] != batch_size:
                    image_input = image_input.expand(batch_size, -1, -1, -1)
                return model(tokens_input, mask_input, image_input)[:, target_indices].sum(dim=1)

            top_tokens = []
            if enable_layer_integrated_gradients_text:
                lig_set = LayerIntegratedGradients(forward_for_text_set, model.text_encoder.embedding)
                with measured(f"neural_set_text_layer_integrated_gradients_idx_{idx}", perf_cfg) as m:
                    text_attr = lig_set.attribute(
                        inputs=tokens,
                        baselines=baseline_tokens,
                        additional_forward_args=(mask, image),
                        n_steps=n_steps,
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                token_scores = text_attr.sum(dim=-1).squeeze(0).detach().cpu().numpy()
                token_scores_real = token_scores[:real_len]
                token_order = np.argsort(np.abs(token_scores_real))[::-1][:top_k]
                top_tokens = [
                    {
                        "position": int(pos),
                        "token": str(raw_tokens[pos]),
                        "attribution": float(token_scores_real[pos]),
                    }
                    for pos in token_order
                ]
                with measured(f"neural_set_text_visualization_idx_{idx}", perf_cfg) as m:
                    save_token_bar(
                        [item["token"] for item in top_tokens],
                        [item["attribution"] for item in top_tokens],
                        sample_dir / "top_token_attributions.png",
                        "Layer IG token attributions: label set",
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                methods.append("LayerIntegratedGradients_text_embedding")

            def image_forward_set(image_input):
                batch_size = image_input.shape[0]
                tokens_use = tokens.expand(batch_size, -1)
                mask_use = mask.expand(batch_size, -1)
                return model(tokens_use, mask_use, image_input)[:, target_indices].sum(dim=1)

            if enable_integrated_gradients_image:
                ig_set = IntegratedGradients(image_forward_set)
                with measured(f"neural_set_image_integrated_gradients_idx_{idx}", perf_cfg) as m:
                    image_attr = ig_set.attribute(
                        inputs=image,
                        baselines=torch.zeros_like(image),
                        n_steps=n_steps,
                    )
                    image_ig_heatmap = image_attr.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                with measured(f"neural_set_image_ig_visualization_idx_{idx}", perf_cfg) as m:
                    save_heatmap_overlay(
                        model_rgb,
                        image_ig_heatmap,
                        sample_dir / "integrated_gradients_image_overlay.png",
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                methods.append("IntegratedGradients_image_pixels")

            if enable_gradcam:
                with measured(f"neural_set_image_gradcam_idx_{idx}", perf_cfg) as m:
                    gradcam_heatmap = compute_gradcam(model, tokens, mask, image, target_indices)
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                with measured(f"neural_set_image_gradcam_visualization_idx_{idx}", perf_cfg) as m:
                    save_heatmap_overlay(model_rgb, gradcam_heatmap, sample_dir / "gradcam_overlay.png")
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                methods.append("GradCAM_image_branch")

            modality_shapley = None
            if enable_modality_shapley:
                with measured(f"neural_set_modality_shapley_idx_{idx}", perf_cfg) as m:
                    modality_shapley = neural_modality_shapley(
                        model,
                        tokens,
                        mask,
                        image,
                        target_indices,
                        pad_id=pad_id_value,
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                with measured(f"neural_set_modality_shapley_visualization_idx_{idx}", perf_cfg) as m:
                    save_modality_shapley_bar(
                        modality_shapley,
                        sample_dir / "modality_shapley.png",
                        "Neural modality Shapley: label set",
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                methods.append("modality_shapley")

            ablation = None
            if enable_modality_ablation:
                with measured(f"neural_set_modality_ablation_idx_{idx}", perf_cfg) as m:
                    ablation = neural_modality_ablation(
                        model,
                        tokens,
                        mask,
                        image,
                        target_indices,
                        pad_id=pad_id_value,
                    )
                method_measurements.append(m.metrics)
                all_measurements.append(m.metrics)
                methods.append("modality_ablation")
            if gate is not None:
                methods.append("GMU_gate_summary")
            if getattr(model, "enable_label_correlation", False):
                methods.append("experimental_label_correlation_head")

            explanation = {
                "index": idx,
                "target_genre": "label_set",
                "target_genres": target_labels,
                "true_genres": true_genres,
                "predicted_genres": selection["predicted_genres"],
                "probability": set_probability,
                "set_logit": set_logit,
                "top_probabilities": top_probability_dict(probs),
                "threshold": threshold_to_serializable(threshold),
                "target_selection": selection,
                "methods": methods,
                "top_tokens": top_tokens,
                "gate_mean": float(gate.mean().detach().cpu().item()) if gate is not None else None,
                "gate_std": float(gate.std().detach().cpu().item()) if gate is not None else None,
                "config": asdict(cfg),
                "performance": {
                    "measurements": method_measurements,
                    "summary": summarize_measurements(method_measurements),
                },
            }
            if modality_shapley is not None:
                explanation["modality_shapley"] = modality_shapley
            if ablation is not None:
                explanation["modality_ablation"] = ablation
            save_json(explanation, sample_dir / "explanation.json")
            explanations.append(explanation)

    summary = {
        "checkpoint_path": str(resolve_path(checkpoint_path)),
        "target_policy": target_policy,
        "target_top_k": int(target_top_k),
        "max_targets_per_sample": max_targets_per_sample,
        "threshold": threshold_to_serializable(threshold),
        "samples": explanations,
        "modality_shapley_summary": summarize_modality_shapley(explanations),
        "performance": {
            "measurements": all_measurements,
            "summary": summarize_measurements(all_measurements),
        },
        "output_stats": directory_output_stats(out),
    }
    save_json(summary, out / "neural_xai_summary.json")
    return summary
