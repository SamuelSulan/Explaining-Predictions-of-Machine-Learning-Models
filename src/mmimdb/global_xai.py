"""Global XAI utilities for the best neural MM-IMDb model."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mmimdb.constants import GENRE_LABELS
from mmimdb.data import DatasetPaths, load_labels, load_metadata
from mmimdb.evaluation import multilabel_metrics, threshold_predictions, threshold_to_serializable
from mmimdb.models.neural import MMIMDBTorchDataset
from mmimdb.splits import load_split_indices
from mmimdb.text_utils import pad_token_id, sequence_to_tokens
from mmimdb.utils import ensure_dir, load_config, resolve_path, save_json, set_seed
from mmimdb.xai import build_modality_shapley, load_neural_model


METHOD_DESCRIPTIONS = [
    {
        "method": "global_modality_ablation",
        "thesis_description": "Compares normal predictions with text-removed, image-removed, and blank-input variants to estimate dataset-level reliance on each modality.",
    },
    {
        "method": "permutation_importance_by_modality",
        "thesis_description": "Shuffles text or image inputs across samples and measures metric degradation, testing whether a modality carries predictive signal beyond local examples.",
    },
    {
        "method": "global_token_occlusion",
        "thesis_description": "Masks candidate words across many examples and aggregates probability/logit drops per genre to identify globally influential words.",
    },
    {
        "method": "aggregated_layer_integrated_gradients_tokens",
        "thesis_description": "Aggregates saved local Layer Integrated Gradients token attributions into per-label positive and negative global token summaries.",
    },
    {
        "method": "per_label_image_occlusion_heatmaps",
        "thesis_description": "Averages patch-occlusion sensitivity maps per genre, showing which poster regions tend to affect a label globally.",
    },
    {
        "method": "global_modality_shapley",
        "thesis_description": "Computes two-modality Shapley values from global text/image/blank coalitions for interaction-aware modality contribution summaries.",
    },
    {
        "method": "prediction_context_summaries",
        "thesis_description": "Reports support, prediction frequency, confidence, and per-label metrics so XAI patterns are read together with label imbalance and performance.",
    },
]


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


def _batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), int(batch_size)):
        yield items[start : start + int(batch_size)]


def _stack_batch(batch: list[dict[str, Any]], device: str):
    import torch

    tokens = torch.stack([item["tokens"] for item in batch]).to(device)
    mask = torch.stack([item["mask"] for item in batch]).to(device)
    image = torch.stack([item["image"] for item in batch]).to(device)
    labels = torch.stack([item["labels"] for item in batch]).cpu().numpy().astype(np.int8)
    indices = np.asarray([int(item["index"]) for item in batch], dtype=np.int64)
    return indices, tokens, mask, image, labels


def _predict_items(
    model,
    items: list[dict[str, Any]],
    device: str,
    batch_size: int,
    pad_id_value: int,
    variant: str = "both",
    seed: int = 42,
):
    import torch

    rng = np.random.default_rng(seed)
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for raw_batch in _batched(items, batch_size):
            idx, tokens, mask, image, y = _stack_batch(raw_batch, device)
            if variant in {"text_only", "no_image"}:
                image = torch.zeros_like(image)
            elif variant in {"image_only", "no_text"}:
                tokens = torch.full_like(tokens, fill_value=int(pad_id_value))
            elif variant == "blank":
                tokens = torch.full_like(tokens, fill_value=int(pad_id_value))
                image = torch.zeros_like(image)
            elif variant == "permute_text" and tokens.shape[0] > 1:
                order = torch.tensor(rng.permutation(tokens.shape[0]), device=tokens.device)
                tokens = tokens[order]
                mask = mask[order]
            elif variant == "permute_image" and image.shape[0] > 1:
                order = torch.tensor(rng.permutation(image.shape[0]), device=image.device)
                image = image[order]
            logits = model(tokens, mask, image)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(y)
            indices.append(idx)
    return np.concatenate(indices), np.vstack(labels), np.vstack(probs)


def _choose_per_label_indices(
    candidate_indices: np.ndarray,
    labels: np.ndarray,
    samples_per_label: int,
    seed: int,
) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    per_label: dict[str, list[int]] = {}
    for label_i, genre in enumerate(GENRE_LABELS):
        positives = np.asarray([int(idx) for idx in candidate_indices if labels[int(idx), label_i] == 1], dtype=np.int64)
        if positives.size > 0:
            rng.shuffle(positives)
        per_label[genre] = positives[: int(samples_per_label)].astype(int).tolist()
    return per_label


def _load_dataset_items(hdf5_path: str | Path, indices: list[int], metadata: dict, cfg) -> list[dict[str, Any]]:
    dataset = MMIMDBTorchDataset(
        hdf5_path,
        np.asarray(indices, dtype=np.int64),
        vocab_size=int(metadata["vocab_size"]),
        max_length=cfg.max_length,
        input_size=cfg.input_size,
    )
    return [dataset[i] for i in range(len(dataset))]


def _metrics_summary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float | list[float] | np.ndarray) -> dict[str, Any]:
    metrics = multilabel_metrics(y_true, y_prob, threshold=threshold)
    y_pred = threshold_predictions(y_prob, threshold)
    rows = []
    for label_i, genre in enumerate(GENRE_LABELS):
        prob = y_prob[:, label_i]
        rows.append(
            {
                "genre": genre,
                "support": int(y_true[:, label_i].sum()),
                "prediction_frequency": float(y_pred[:, label_i].mean()),
                "mean_probability": float(prob.mean()),
                "mean_positive_probability": float(prob[y_true[:, label_i] == 1].mean())
                if np.any(y_true[:, label_i] == 1)
                else 0.0,
                **metrics["per_label"][genre],
            }
        )
    return {"metrics": metrics, "per_label_rows": rows}


def _save_modality_plots(modality_df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    figures: dict[str, str] = {}
    if modality_df.empty:
        return figures

    overall = modality_df[modality_df["genre"] == "GLOBAL"]
    if not overall.empty:
        long = overall.melt(
            id_vars=["genre"],
            value_vars=["text_utilization", "image_utilization"],
            var_name="modality",
            value_name="utilization",
        )
        plt.figure(figsize=(6, 4))
        sns.barplot(data=long, x="modality", y="utilization", hue="modality", legend=False)
        plt.ylim(0, 1)
        plt.title("Global aggregate modality utilization")
        plt.tight_layout()
        path = output_dir / "global_modality_utilization.png"
        plt.savefig(path, dpi=170)
        plt.close()
        figures["global_modality_utilization"] = str(path)

    per_label = modality_df[modality_df["genre"] != "GLOBAL"]
    if not per_label.empty:
        long = per_label.melt(
            id_vars=["genre"],
            value_vars=["text_utilization", "image_utilization"],
            var_name="modality",
            value_name="utilization",
        )
        plt.figure(figsize=(11, 7))
        sns.barplot(data=long, y="genre", x="utilization", hue="modality")
        plt.xlim(0, 1)
        plt.title("Global aggregate modality utilization by label")
        plt.tight_layout()
        path = output_dir / "per_label_modality_utilization.png"
        plt.savefig(path, dpi=170)
        plt.close()
        figures["per_label_modality_utilization"] = str(path)
    return figures


def _save_context_plot(context_df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    figures = {}
    if context_df.empty:
        return figures
    plt.figure(figsize=(11, 7))
    plot_df = context_df.melt(
        id_vars=["genre"],
        value_vars=["support", "prediction_frequency"],
        var_name="measure",
        value_name="value",
    )
    support_max = max(float(context_df["support"].max()), 1.0)
    plot_df.loc[plot_df["measure"] == "support", "value"] /= support_max
    sns.barplot(data=plot_df, y="genre", x="value", hue="measure")
    plt.xlim(0, 1)
    plt.title("Prediction context: normalized support and prediction frequency")
    plt.tight_layout()
    path = output_dir / "prediction_context.png"
    plt.savefig(path, dpi=170)
    plt.close()
    figures["prediction_context"] = str(path)
    return figures


def _candidate_tokens(
    hdf5_path: str | Path,
    metadata: dict,
    label_indices: list[int],
    label_i: int,
    top_n: int,
) -> list[dict[str, Any]]:
    ix_to_word = metadata["ix_to_word"]
    counter: Counter[str] = Counter()
    with h5py.File(resolve_path(hdf5_path), "r") as f:
        sequences = f["sequences"]
        for idx in label_indices:
            tokens = sequence_to_tokens(sequences[int(idx)], ix_to_word)
            counter.update(token for token in tokens if len(str(token)) > 1 and str(token) != "N")
    return [
        {"genre": GENRE_LABELS[label_i], "token": token, "positive_sample_count": int(count)}
        for token, count in counter.most_common(int(top_n))
    ]


def _token_occlusion_for_label(
    model,
    items: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    label_i: int,
    pad_id_value: int,
    metadata: dict,
    hdf5_path: str | Path,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch

    if not items or not candidate_rows:
        return []
    idx_to_tokens: dict[int, list[str]] = {}
    ix_to_word = metadata["ix_to_word"]
    with h5py.File(resolve_path(hdf5_path), "r") as f:
        for item in items:
            idx = int(item["index"])
            idx_to_tokens[idx] = sequence_to_tokens(f["sequences"][idx], ix_to_word)

    rows = []
    model.eval()
    with torch.no_grad():
        for candidate in candidate_rows:
            token = str(candidate["token"])
            deltas = []
            logit_deltas = []
            affected = 0
            for raw_batch in _batched(items, batch_size):
                _, tokens, mask, image, _ = _stack_batch(raw_batch, device)
                occluded = tokens.clone()
                batch_has_token = []
                for row_i, item in enumerate(raw_batch):
                    words = idx_to_tokens[int(item["index"])]
                    positions = [pos for pos, word in enumerate(words[: occluded.shape[1]]) if word == token]
                    batch_has_token.append(bool(positions))
                    for pos in positions:
                        occluded[row_i, pos] = int(pad_id_value)
                if not any(batch_has_token):
                    continue
                base_logits = model(tokens, mask, image)
                occ_logits = model(occluded, mask, image)
                base_prob = torch.sigmoid(base_logits)[:, label_i]
                occ_prob = torch.sigmoid(occ_logits)[:, label_i]
                keep = torch.tensor(batch_has_token, dtype=torch.bool, device=device)
                deltas.extend((base_prob[keep] - occ_prob[keep]).detach().cpu().numpy().tolist())
                logit_deltas.extend((base_logits[:, label_i][keep] - occ_logits[:, label_i][keep]).detach().cpu().numpy().tolist())
                affected += int(keep.sum().detach().cpu().item())
            if affected:
                rows.append(
                    {
                        "genre": GENRE_LABELS[label_i],
                        "token": token,
                        "positive_sample_count": int(candidate["positive_sample_count"]),
                        "affected_samples": int(affected),
                        "mean_probability_delta": float(np.mean(deltas)),
                        "mean_abs_probability_delta": float(np.mean(np.abs(deltas))),
                        "mean_logit_delta": float(np.mean(logit_deltas)),
                    }
                )
    return sorted(rows, key=lambda row: abs(row["mean_probability_delta"]), reverse=True)


def _image_occlusion_for_label(
    model,
    items: list[dict[str, Any]],
    label_i: int,
    grid_size: int,
    device: str,
    batch_size: int,
) -> np.ndarray | None:
    import torch

    if not items:
        return None
    heatmaps = []
    model.eval()
    with torch.no_grad():
        for raw_batch in _batched(items, batch_size):
            _, tokens, mask, image, _ = _stack_batch(raw_batch, device)
            base_prob = torch.sigmoid(model(tokens, mask, image))[:, label_i]
            _, _, h, w = image.shape
            batch_heatmap = np.zeros((image.shape[0], grid_size, grid_size), dtype=np.float32)
            for row in range(grid_size):
                y0 = int(round(row * h / grid_size))
                y1 = int(round((row + 1) * h / grid_size))
                for col in range(grid_size):
                    x0 = int(round(col * w / grid_size))
                    x1 = int(round((col + 1) * w / grid_size))
                    occluded = image.clone()
                    occluded[:, :, y0:y1, x0:x1] = 0.0
                    occ_prob = torch.sigmoid(model(tokens, mask, occluded))[:, label_i]
                    batch_heatmap[:, row, col] = (base_prob - occ_prob).detach().cpu().numpy()
            heatmaps.append(batch_heatmap)
    if not heatmaps:
        return None
    return np.vstack(heatmaps).mean(axis=0)


def _save_heatmap(heatmap: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(4.6, 4))
    sns.heatmap(heatmap, cmap="coolwarm", center=0, square=True, cbar_kws={"label": "probability drop"})
    plt.title(title)
    plt.xlabel("poster patch column")
    plt.ylabel("poster patch row")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def aggregate_local_lig_tokens(local_xai_dir: str | Path | None, output_dir: str | Path, top_k: int = 20) -> dict[str, Any]:
    if local_xai_dir is None:
        return {"rows": [], "warning": "No local XAI directory was provided; aggregated Layer IG tokens were skipped."}
    root = resolve_path(local_xai_dir)
    if not root.exists():
        return {"rows": [], "warning": f"Local XAI directory does not exist: {root}"}

    accum: dict[tuple[str, str], list[float]] = defaultdict(list)
    explanation_paths = list(root.rglob("explanation.json"))
    if not explanation_paths:
        return {"rows": [], "warning": f"No local explanation.json files found under: {root}"}

    for explanation_path in explanation_paths:
        try:
            with explanation_path.open("r", encoding="utf-8") as f:
                explanation = json.load(f)
        except Exception:
            continue
        genres = explanation.get("target_genres") or [explanation.get("target_genre")]
        if not genres:
            continue
        for item in explanation.get("top_tokens", []):
            token = str(item.get("token", ""))
            if not token:
                continue
            value = float(item.get("attribution", 0.0))
            for genre in genres:
                if genre in GENRE_LABELS:
                    accum[(genre, token)].append(value)

    rows = []
    for (genre, token), values in accum.items():
        rows.append(
            {
                "genre": genre,
                "token": token,
                "count": int(len(values)),
                "mean_attribution": float(np.mean(values)),
                "mean_abs_attribution": float(np.mean(np.abs(values))),
            }
        )
    rows = sorted(rows, key=lambda row: (row["genre"], -row["mean_abs_attribution"]))
    out = ensure_dir(output_dir)
    pd.DataFrame(rows).to_csv(out / "aggregated_lig_tokens.csv", index=False)

    top_rows = []
    for genre in GENRE_LABELS:
        label_rows = [row for row in rows if row["genre"] == genre]
        top_rows.extend(sorted(label_rows, key=lambda row: row["mean_attribution"], reverse=True)[:top_k])
        top_rows.extend(sorted(label_rows, key=lambda row: row["mean_attribution"])[:top_k])
    return {"rows": top_rows, "source_dir": str(root)}


def run_global_neural_xai(
    config_path: str | Path = "configs/default.yaml",
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path = "outputs/global_xai/best_neural",
    split: str = "test",
    samples_per_label: int = 25,
    seed: int = 42,
    batch_size: int | None = None,
    token_candidates_per_label: int = 20,
    token_occlusion_top_k: int = 10,
    image_occlusion_grid: int = 4,
    enable_token_occlusion: bool = True,
    enable_image_occlusion: bool = True,
    local_xai_dir: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    start = time.perf_counter()
    config = load_config(config_path)
    paths = DatasetPaths.from_config(config)
    train_idx, val_idx, test_idx = load_split_indices(resolve_path(config["splits"]["output_dir"]))
    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    labels_all = load_labels(paths.hdf5)
    per_label_indices = _choose_per_label_indices(split_indices, labels_all, samples_per_label, seed)
    selected_indices = sorted({idx for indices in per_label_indices.values() for idx in indices})
    if not selected_indices:
        raise ValueError(f"No samples selected for split={split!r}.")

    out = ensure_dir(output_dir)
    model_path = resolve_path(
        checkpoint_path
        or config.get("xai", {}).get("neural", {}).get("checkpoint_path", "outputs/models/best/neural_multimodal_best.pt")
    )
    model, cfg, metadata, checkpoint, device = load_neural_model(model_path, paths.metadata)
    actual_batch_size = int(batch_size or min(cfg.batch_size, 32))
    pad_id_value = int(checkpoint.get("pad_id", pad_token_id(metadata)))

    all_items = _load_dataset_items(paths.hdf5, selected_indices, metadata, cfg)
    _, y_true, prob_both = _predict_items(model, all_items, device, actual_batch_size, pad_id_value, "both", seed)
    _, _, prob_text = _predict_items(model, all_items, device, actual_batch_size, pad_id_value, "text_only", seed)
    _, _, prob_image = _predict_items(model, all_items, device, actual_batch_size, pad_id_value, "image_only", seed)
    _, _, prob_blank = _predict_items(model, all_items, device, actual_batch_size, pad_id_value, "blank", seed)
    _, _, prob_perm_text = _predict_items(
        model, all_items, device, actual_batch_size, pad_id_value, "permute_text", seed + 11
    )
    _, _, prob_perm_image = _predict_items(
        model, all_items, device, actual_batch_size, pad_id_value, "permute_image", seed + 23
    )

    threshold = checkpoint.get("threshold", 0.5)
    context = _metrics_summary(y_true, prob_both, threshold)
    context_df = pd.DataFrame(context["per_label_rows"])
    context_df.to_csv(out / "prediction_context.csv", index=False)

    variant_rows = []
    for name, probs in [
        ("both", prob_both),
        ("text_only", prob_text),
        ("image_only", prob_image),
        ("blank", prob_blank),
        ("permute_text", prob_perm_text),
        ("permute_image", prob_perm_image),
    ]:
        metrics = multilabel_metrics(y_true, probs, threshold=threshold)
        variant_rows.append(
            {
                "variant": name,
                "sample_f1": metrics["sample_f1"],
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "hamming_loss": metrics["hamming_loss"],
            }
        )
    variant_df = pd.DataFrame(variant_rows)
    variant_df.to_csv(out / "modality_ablation_and_permutation.csv", index=False)

    modality_rows = []
    coalitions = {
        "blank": prob_blank.mean(axis=0),
        "text_only": prob_text.mean(axis=0),
        "image_only": prob_image.mean(axis=0),
        "both": prob_both.mean(axis=0),
    }
    global_shapley = build_modality_shapley(
        {name: float(values.mean()) for name, values in coalitions.items()},
        {name: float(np.mean(np.log(np.clip(values, 1e-7, 1 - 1e-7) / np.clip(1 - values, 1e-7, 1)))) for name, values in coalitions.items()},
    )
    global_s = global_shapley["probability"]["shapley"]
    modality_rows.append(
        {
            "genre": "GLOBAL",
            "text_utilization": global_s["text_utilization"],
            "image_utilization": global_s["image_utilization"],
            "text_contribution": global_s["text"],
            "image_contribution": global_s["image"],
            "interaction": global_s["interaction"],
        }
    )
    per_label_shapley = {}
    for label_i, genre in enumerate(GENRE_LABELS):
        values = {name: float(arr[label_i]) for name, arr in coalitions.items()}
        label_shapley = build_modality_shapley(values, values)
        per_label_shapley[genre] = label_shapley
        s = label_shapley["probability"]["shapley"]
        modality_rows.append(
            {
                "genre": genre,
                "text_utilization": s["text_utilization"],
                "image_utilization": s["image_utilization"],
                "text_contribution": s["text"],
                "image_contribution": s["image"],
                "interaction": s["interaction"],
            }
        )
    modality_df = pd.DataFrame(modality_rows)
    modality_df.to_csv(out / "global_modality_shapley.csv", index=False)

    token_candidate_rows = []
    token_occlusion_rows = []
    image_heatmap_rows = []
    heatmap_figures = {}
    if enable_token_occlusion or enable_image_occlusion:
        for label_i, genre in enumerate(GENRE_LABELS):
            label_indices = per_label_indices[genre]
            label_items = _load_dataset_items(paths.hdf5, label_indices, metadata, cfg)
            if enable_token_occlusion:
                candidates = _candidate_tokens(paths.hdf5, metadata, label_indices, label_i, token_candidates_per_label)
                token_candidate_rows.extend(candidates)
                token_occlusion_rows.extend(
                    _token_occlusion_for_label(
                        model,
                        label_items,
                        candidates[:token_occlusion_top_k],
                        label_i,
                        pad_id_value,
                        metadata,
                        paths.hdf5,
                        device,
                        actual_batch_size,
                    )
                )
            if enable_image_occlusion:
                heatmap = _image_occlusion_for_label(
                    model,
                    label_items,
                    label_i,
                    image_occlusion_grid,
                    device,
                    actual_batch_size,
                )
                if heatmap is not None:
                    heatmap_path = out / f"image_occlusion_heatmap_{genre.replace('/', '_')}.png"
                    _save_heatmap(heatmap, heatmap_path, f"Global aggregate image occlusion: {genre}")
                    heatmap_figures[f"image_occlusion_heatmap_{genre}"] = str(heatmap_path)
                    for row in range(heatmap.shape[0]):
                        for col in range(heatmap.shape[1]):
                            image_heatmap_rows.append(
                                {
                                    "genre": genre,
                                    "row": int(row),
                                    "col": int(col),
                                    "mean_probability_delta": float(heatmap[row, col]),
                                }
                            )

    pd.DataFrame(token_candidate_rows).to_csv(out / "token_candidates.csv", index=False)
    token_occlusion_df = pd.DataFrame(token_occlusion_rows)
    token_occlusion_df.to_csv(out / "global_token_occlusion.csv", index=False)
    pd.DataFrame(image_heatmap_rows).to_csv(out / "per_label_image_occlusion_heatmaps.csv", index=False)

    lig_summary = aggregate_local_lig_tokens(local_xai_dir, out, top_k=20)
    methods_df = pd.DataFrame(METHOD_DESCRIPTIONS)
    methods_df.to_csv(out / "global_xai_methods_for_thesis.csv", index=False)
    figures = {}
    figures.update(_save_modality_plots(modality_df, out))
    figures.update(_save_context_plot(context_df, out))
    figures.update(heatmap_figures)

    if not token_occlusion_df.empty:
        plot_rows = []
        for genre in GENRE_LABELS:
            label_rows = token_occlusion_df[token_occlusion_df["genre"] == genre].copy()
            if label_rows.empty:
                continue
            label_rows = label_rows.reindex(label_rows["mean_abs_probability_delta"].abs().sort_values(ascending=False).index)
            plot_rows.extend(label_rows.head(5).to_dict("records"))
        if plot_rows:
            plt.figure(figsize=(11, max(7, len(plot_rows) * 0.18)))
            plot_df = pd.DataFrame(plot_rows)
            plot_df["token_label"] = plot_df["genre"] + ": " + plot_df["token"]
            sns.barplot(data=plot_df, y="token_label", x="mean_probability_delta", hue="genre", dodge=False, legend=False)
            plt.axvline(0, color="black", linewidth=0.8)
            plt.title("Global aggregate token occlusion impact")
            plt.tight_layout()
            path = out / "global_token_occlusion_top.png"
            plt.savefig(path, dpi=170)
            plt.close()
            figures["global_token_occlusion_top"] = str(path)

    label_warnings = [
        {
            "genre": genre,
            "selected_samples": int(len(indices)),
            "warning": "Few selected positive examples; interpret this label's global XAI cautiously.",
        }
        for genre, indices in per_label_indices.items()
        if len(indices) < max(3, min(10, samples_per_label))
    ]
    summary = {
        "model_type": "neural",
        "model_role": "best_neural",
        "checkpoint_path": str(model_path),
        "split": split,
        "num_labels": len(GENRE_LABELS),
        "label_names": GENRE_LABELS,
        "selected_sample_count": int(len(selected_indices)),
        "samples_per_label_requested": int(samples_per_label),
        "per_label_selected_indices": per_label_indices,
        "threshold": threshold_to_serializable(threshold),
        "device": device,
        "runtime_seconds": float(time.perf_counter() - start),
        "methods": METHOD_DESCRIPTIONS,
        "warnings": label_warnings + ([{"warning": lig_summary["warning"]}] if lig_summary.get("warning") else []),
        "prediction_context": context["per_label_rows"],
        "metrics": context["metrics"],
        "modality_ablation_and_permutation": variant_rows,
        "global_modality_shapley": {
            "overall": global_shapley,
            "per_label": per_label_shapley,
        },
        "top_token_occlusion": token_occlusion_df.head(200).to_dict("records") if not token_occlusion_df.empty else [],
        "aggregated_lig_tokens": lig_summary.get("rows", []),
        "figures": figures,
        "artifacts": {
            "prediction_context_csv": str(out / "prediction_context.csv"),
            "modality_ablation_csv": str(out / "modality_ablation_and_permutation.csv"),
            "modality_shapley_csv": str(out / "global_modality_shapley.csv"),
            "token_occlusion_csv": str(out / "global_token_occlusion.csv"),
            "image_occlusion_heatmaps_csv": str(out / "per_label_image_occlusion_heatmaps.csv"),
            "methods_csv": str(out / "global_xai_methods_for_thesis.csv"),
        },
    }
    save_json(_json_safe(summary), out / "global_xai_summary.json")
    return _json_safe(summary)
