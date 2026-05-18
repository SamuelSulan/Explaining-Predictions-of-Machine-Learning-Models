"""Classic multimodal ML baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import joblib
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import StandardScaler

from mmimdb.constants import GENRE_LABELS
from mmimdb.data import load_metadata, read_h5_rows
from mmimdb.evaluation import multilabel_metrics, threshold_to_serializable, tune_thresholds
from mmimdb.image_utils import image_descriptor
from mmimdb.text_utils import sequence_to_text
from mmimdb.utils import ensure_dir, resolve_path, save_json


class SafeBinaryClassifier(ClassifierMixin, BaseEstimator):
    """Binary classifier wrapper that tolerates constant labels in smoke splits."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, x, y):
        y = np.asarray(y)
        self.classes_ = np.asarray([0, 1])
        unique = np.unique(y)
        if unique.size < 2:
            self.constant_ = int(unique[0]) if unique.size else 0
            self.estimator_ = None
            return self
        self.constant_ = None
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(x, y)
        return self

    def predict(self, x):
        if self.estimator_ is None:
            return np.full(x.shape[0], self.constant_, dtype=np.int8)
        return self.estimator_.predict(x)

    def predict_proba(self, x):
        if self.estimator_ is None:
            probs = np.zeros((x.shape[0], 2), dtype=np.float32)
            probs[:, int(self.constant_)] = 1.0
            return probs
        return self.estimator_.predict_proba(x)

    def decision_function(self, x):
        if self.estimator_ is None:
            value = 20.0 if int(self.constant_) == 1 else -20.0
            return np.full(x.shape[0], value, dtype=np.float32)
        return self.estimator_.decision_function(x)

    @property
    def coef_(self):
        if self.estimator_ is None:
            raise AttributeError("Constant predictor has no coefficients.")
        return self.estimator_.coef_

    @property
    def intercept_(self):
        if self.estimator_ is None:
            raise AttributeError("Constant predictor has no intercept.")
        return self.estimator_.intercept_


@dataclass
class ClassicConfig:
    text_max_features: int = 30000
    text_ngram_min: int = 1
    text_ngram_max: int = 2
    text_min_df: int = 2
    image_hist_bins: int = 16
    image_thumbnail_size: tuple[int, int] = (32, 20)
    logistic_max_iter: int = 1000
    logistic_c: float = 1.0
    logistic_penalty: str = "l2"
    logistic_solver: str = "saga"
    logistic_l1_ratio: float | None = None
    logistic_class_weight: str | None = "balanced"
    estimator: str = "logistic"
    sgd_loss: str = "log_loss"
    sgd_penalty: str = "l2"
    sgd_alpha: float = 1e-5
    sgd_max_iter: int = 1000
    sgd_tol: float = 1e-3
    sgd_class_weight: str | None = "balanced"
    threshold_metric: str = "macro_f1"
    threshold_strategy: str = "per_label"
    classifier: str = "classifier_chain"
    chain_order: str | list[int] | None = None
    random_state: int = 42

    @classmethod
    def from_config(cls, config: dict) -> "ClassicConfig":
        raw = config.get("classic", {})
        thumb = raw.get("image_thumbnail_size", [32, 20])
        chain_order = raw.get("chain_order", None)
        if isinstance(chain_order, str) and chain_order.lower() in {"", "none", "default"}:
            chain_order = None
        return cls(
            text_max_features=int(raw.get("text_max_features", 30000)),
            text_ngram_min=int(raw.get("text_ngram_min", 1)),
            text_ngram_max=int(raw.get("text_ngram_max", 2)),
            text_min_df=int(raw.get("text_min_df", 2)),
            image_hist_bins=int(raw.get("image_hist_bins", 16)),
            image_thumbnail_size=(int(thumb[0]), int(thumb[1])),
            logistic_max_iter=int(raw.get("logistic_max_iter", 1000)),
            logistic_c=float(raw.get("logistic_c", 1.0)),
            logistic_penalty=str(raw.get("logistic_penalty", "l2")),
            logistic_solver=str(raw.get("logistic_solver", "saga")),
            logistic_l1_ratio=(
                None if raw.get("logistic_l1_ratio", None) is None else float(raw.get("logistic_l1_ratio"))
            ),
            logistic_class_weight=raw.get("logistic_class_weight", "balanced"),
            estimator=str(raw.get("estimator", "logistic")),
            sgd_loss=str(raw.get("sgd_loss", "log_loss")),
            sgd_penalty=str(raw.get("sgd_penalty", "l2")),
            sgd_alpha=float(raw.get("sgd_alpha", 1e-5)),
            sgd_max_iter=int(raw.get("sgd_max_iter", 1000)),
            sgd_tol=float(raw.get("sgd_tol", 1e-3)),
            sgd_class_weight=raw.get("sgd_class_weight", "balanced"),
            threshold_metric=str(raw.get("threshold_metric", "macro_f1")),
            threshold_strategy=str(raw.get("threshold_strategy", "per_label")),
            classifier=str(raw.get("classifier", "classifier_chain")),
            chain_order=chain_order,
            random_state=int(raw.get("random_state", config.get("project", {}).get("seed", 42))),
        )


def load_reconstructed_texts(
    hdf5_path: str | Path,
    metadata_path: str | Path,
    indices: np.ndarray,
) -> list[str]:
    metadata = load_metadata(metadata_path)
    ix_to_word = metadata["ix_to_word"]
    texts: list[str] = []
    with h5py.File(resolve_path(hdf5_path), "r") as f:
        seq_ds = f["sequences"]
        for idx in indices:
            texts.append(sequence_to_text(seq_ds[int(idx)], ix_to_word))
    return texts


def extract_image_descriptors(
    hdf5_path: str | Path,
    indices: np.ndarray,
    hist_bins: int = 16,
    thumbnail_size: tuple[int, int] = (32, 20),
    batch_size: int = 256,
) -> np.ndarray:
    descriptors = []
    with h5py.File(resolve_path(hdf5_path), "r") as f:
        images = f["images"]
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = read_h5_rows(images, batch_idx)
            for image in batch:
                descriptors.append(
                    image_descriptor(
                        image,
                        hist_bins=hist_bins,
                        thumbnail_size=thumbnail_size,
                    )
                )
    return np.vstack(descriptors).astype(np.float32)


def make_feature_blocks(
    hdf5_path: str | Path,
    metadata_path: str | Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray | None,
    cfg: ClassicConfig,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix | None, dict]:
    train_texts = load_reconstructed_texts(hdf5_path, metadata_path, train_idx)
    val_texts = load_reconstructed_texts(hdf5_path, metadata_path, val_idx)
    test_texts = load_reconstructed_texts(hdf5_path, metadata_path, test_idx) if test_idx is not None else None

    vectorizer = TfidfVectorizer(
        max_features=cfg.text_max_features,
        ngram_range=(cfg.text_ngram_min, cfg.text_ngram_max),
        min_df=cfg.text_min_df,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
    )
    x_text_train = vectorizer.fit_transform(train_texts)
    x_text_val = vectorizer.transform(val_texts)
    x_text_test = vectorizer.transform(test_texts) if test_texts is not None else None

    x_img_train = extract_image_descriptors(
        hdf5_path,
        train_idx,
        hist_bins=cfg.image_hist_bins,
        thumbnail_size=cfg.image_thumbnail_size,
    )
    x_img_val = extract_image_descriptors(
        hdf5_path,
        val_idx,
        hist_bins=cfg.image_hist_bins,
        thumbnail_size=cfg.image_thumbnail_size,
    )
    x_img_test = (
        extract_image_descriptors(
            hdf5_path,
            test_idx,
            hist_bins=cfg.image_hist_bins,
            thumbnail_size=cfg.image_thumbnail_size,
        )
        if test_idx is not None
        else None
    )

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_img_train = scaler.fit_transform(x_img_train)
    x_img_val = scaler.transform(x_img_val)
    x_img_test = scaler.transform(x_img_test) if x_img_test is not None else None

    x_train = sparse.hstack([x_text_train, sparse.csr_matrix(x_img_train)], format="csr")
    x_val = sparse.hstack([x_text_val, sparse.csr_matrix(x_img_val)], format="csr")
    x_test = (
        sparse.hstack([x_text_test, sparse.csr_matrix(x_img_test)], format="csr")
        if x_text_test is not None and x_img_test is not None
        else None
    )

    artifacts = {
        "vectorizer": vectorizer,
        "image_scaler": scaler,
        "text_feature_count": int(x_text_train.shape[1]),
        "image_feature_count": int(x_img_train.shape[1]),
    }
    return x_train, x_val, x_test, artifacts


def build_base_estimator(cfg: ClassicConfig):
    estimator_name = cfg.estimator.lower()
    if estimator_name in {"logistic", "logreg", "logistic_regression"}:
        return LogisticRegression(
            penalty=cfg.logistic_penalty,
            C=cfg.logistic_c,
            class_weight=cfg.logistic_class_weight,
            max_iter=cfg.logistic_max_iter,
            l1_ratio=cfg.logistic_l1_ratio if cfg.logistic_penalty == "elasticnet" else None,
            solver=cfg.logistic_solver,
            n_jobs=1,
            random_state=cfg.random_state,
            verbose=0,
        )
    if estimator_name in {"sgd", "sgd_classifier", "sgdclassifier"}:
        if cfg.sgd_loss not in {"log_loss", "modified_huber"}:
            raise ValueError(
                "SGD classic estimator requires sgd_loss='log_loss' or "
                "sgd_loss='modified_huber' so predict_proba is available."
            )
        return SGDClassifier(
            loss=cfg.sgd_loss,
            penalty=cfg.sgd_penalty,
            alpha=cfg.sgd_alpha,
            max_iter=cfg.sgd_max_iter,
            tol=cfg.sgd_tol,
            class_weight=cfg.sgd_class_weight,
            random_state=cfg.random_state,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported classic estimator: {cfg.estimator}")


def train_classic_multimodal(
    hdf5_path: str | Path,
    metadata_path: str | Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray | None,
    output_dir: str | Path,
    cfg: ClassicConfig,
    limit: int | None = None,
) -> dict:
    if limit is not None:
        train_idx = train_idx[:limit]
        val_idx = val_idx[: max(1, min(len(val_idx), limit // 5))]
        if test_idx is not None:
            test_idx = test_idx[: max(1, min(len(test_idx), limit // 5))]

    with h5py.File(resolve_path(hdf5_path), "r") as f:
        y = f["genres"][:].astype(np.int8)
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx] if test_idx is not None else None

    x_train, x_val, x_test, artifacts = make_feature_blocks(
        hdf5_path,
        metadata_path,
        train_idx,
        val_idx,
        test_idx,
        cfg,
    )

    base = build_base_estimator(cfg)
    classifier_name = cfg.classifier.lower()
    if classifier_name in {"ovr", "one_vs_rest", "one-vs-rest"}:
        classifier = OneVsRestClassifier(base, n_jobs=-1)
    elif classifier_name in {"classifier_chain", "chain"}:
        classifier = ClassifierChain(
            SafeBinaryClassifier(base),
            order=cfg.chain_order,
            random_state=cfg.random_state,
        )
    else:
        raise ValueError(f"Unsupported classic classifier: {cfg.classifier}")
    classifier.fit(x_train, y_train)

    val_prob = classifier.predict_proba(x_val)
    threshold, val_metrics = tune_thresholds(
        y_val,
        val_prob,
        metric=cfg.threshold_metric,
        strategy=cfg.threshold_strategy,
    )
    test_metrics = None
    if x_test is not None and y_test is not None:
        test_prob = classifier.predict_proba(x_test)
        test_metrics = multilabel_metrics(y_test, test_prob, threshold=threshold)
    threshold_saved = threshold_to_serializable(threshold)

    out = ensure_dir(output_dir)
    suffix = f"_limit{limit}" if limit is not None else ""
    model_path = out / f"classic_multimodal{suffix}.joblib"
    joblib.dump(
        {
            "classifier": classifier,
            "vectorizer": artifacts["vectorizer"],
            "image_scaler": artifacts["image_scaler"],
            "threshold": threshold_saved,
            "genre_labels": GENRE_LABELS,
            "config": cfg.__dict__,
            "estimator": cfg.estimator,
            "feature_info": {
                "text_feature_count": artifacts["text_feature_count"],
                "image_feature_count": artifacts["image_feature_count"],
            },
        },
        model_path,
    )

    result = {
        "model_path": str(model_path),
        "threshold": threshold_saved,
        "threshold_strategy": cfg.threshold_strategy,
        "classifier": cfg.classifier,
        "estimator": cfg.estimator,
        "feature_info": {
            "text_feature_count": artifacts["text_feature_count"],
            "image_feature_count": artifacts["image_feature_count"],
            "total_feature_count": int(x_train.shape[1]),
        },
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)) if test_idx is not None else 0,
        "validation": val_metrics,
    }
    if test_metrics is not None:
        result["test"] = test_metrics
    save_json(result, out / f"classic_multimodal{suffix}_metrics.json")
    return result
