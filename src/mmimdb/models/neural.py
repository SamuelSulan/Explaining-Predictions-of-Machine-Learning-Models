"""PyTorch multimodal models and training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from mmimdb.constants import CAFFE_BGR_MEAN, GENRE_LABELS
from mmimdb.data import load_labels, load_metadata
from mmimdb.evaluation import multilabel_metrics, threshold_to_serializable, tune_thresholds
from mmimdb.text_utils import build_embedding_matrix, pad_token_id, prepare_token_ids
from mmimdb.utils import ensure_dir, resolve_path, save_json, set_seed


@dataclass
class NeuralConfig:
    model_name: str = "multimodal_bigru_attention_resnet18_gmu"
    modality: str = "multimodal"
    text_encoder: str = "bigru_attention"
    image_encoder: str = "resnet18"
    fusion: str = "gmu"
    hidden_dim: int = 256
    text_cnn_channels: int = 128
    text_cnn_kernels: tuple[int, ...] = (3, 4, 5)
    text_rnn_hidden_dim: int = 128
    text_rnn_layers: int = 1
    text_rnn_dropout: float = 0.2
    text_attention_dim: int = 128
    text_transformer_layers: int = 2
    text_transformer_heads: int = 6
    text_transformer_ff_dim: int = 512
    text_transformer_dropout: float = 0.1
    pretrained_image: bool = True
    freeze_image_backbone: bool = True
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_workers: int = 0
    patience: int = 3
    threshold_metric: str = "macro_f1"
    threshold_strategy: str = "per_label"
    threshold_min: float = 0.01
    threshold_max: float = 0.99
    threshold_steps: int = 99
    loss: str = "bce"
    focal_gamma: float = 2.0
    pos_weight_clip: float | None = None
    scheduler: str = "none"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 1
    min_learning_rate: float = 1e-6
    enable_label_correlation: bool = False
    max_length: int = 512
    input_size: int = 224
    freeze_embeddings: bool = False

    @classmethod
    def from_config(cls, config: dict) -> "NeuralConfig":
        raw = config.get("neural", {})
        text = config.get("text", {})
        image = config.get("image", {})
        return cls(
            model_name=str(raw.get("model_name", "multimodal_bigru_attention_resnet18_gmu")),
            modality=str(raw.get("modality", "multimodal")),
            text_encoder=str(raw.get("text_encoder", "bigru_attention")),
            image_encoder=str(raw.get("image_encoder", "resnet18")),
            fusion=str(raw.get("fusion", "gmu")),
            hidden_dim=int(raw.get("hidden_dim", 256)),
            text_cnn_channels=int(raw.get("text_cnn_channels", 128)),
            text_cnn_kernels=tuple(int(k) for k in raw.get("text_cnn_kernels", [3, 4, 5])),
            text_rnn_hidden_dim=int(raw.get("text_rnn_hidden_dim", 128)),
            text_rnn_layers=int(raw.get("text_rnn_layers", 1)),
            text_rnn_dropout=float(raw.get("text_rnn_dropout", 0.2)),
            text_attention_dim=int(raw.get("text_attention_dim", 128)),
            text_transformer_layers=int(raw.get("text_transformer_layers", 2)),
            text_transformer_heads=int(raw.get("text_transformer_heads", 6)),
            text_transformer_ff_dim=int(raw.get("text_transformer_ff_dim", 512)),
            text_transformer_dropout=float(raw.get("text_transformer_dropout", 0.1)),
            pretrained_image=bool(raw.get("pretrained_image", True)),
            freeze_image_backbone=bool(raw.get("freeze_image_backbone", True)),
            batch_size=int(raw.get("batch_size", 16)),
            epochs=int(raw.get("epochs", 10)),
            learning_rate=float(raw.get("learning_rate", 1e-4)),
            weight_decay=float(raw.get("weight_decay", 0.01)),
            num_workers=int(raw.get("num_workers", 0)),
            patience=int(raw.get("patience", 3)),
            threshold_metric=str(raw.get("threshold_metric", "macro_f1")),
            threshold_strategy=str(raw.get("threshold_strategy", "per_label")),
            threshold_min=float(raw.get("threshold_min", 0.01)),
            threshold_max=float(raw.get("threshold_max", 0.99)),
            threshold_steps=int(raw.get("threshold_steps", 99)),
            loss=str(raw.get("loss", "bce")),
            focal_gamma=float(raw.get("focal_gamma", 2.0)),
            pos_weight_clip=(
                None if raw.get("pos_weight_clip", None) is None else float(raw.get("pos_weight_clip"))
            ),
            scheduler=str(raw.get("scheduler", "none")),
            scheduler_factor=float(raw.get("scheduler_factor", 0.5)),
            scheduler_patience=int(raw.get("scheduler_patience", 1)),
            min_learning_rate=float(raw.get("min_learning_rate", 1e-6)),
            enable_label_correlation=bool(raw.get("enable_label_correlation", False)),
            max_length=int(text.get("max_length", 512)),
            input_size=int(image.get("input_size", 224)),
            freeze_embeddings=bool(text.get("freeze_embeddings", False)),
        )


class MMIMDBTorchDataset:
    """Lazy HDF5-backed dataset for PyTorch DataLoader."""

    def __init__(
        self,
        hdf5_path: str | Path,
        indices: np.ndarray,
        vocab_size: int,
        max_length: int,
        input_size: int,
    ) -> None:
        self.hdf5_path = resolve_path(hdf5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.vocab_size = int(vocab_size)
        self.pad_id = int(vocab_size)
        self.max_length = int(max_length)
        self.input_size = int(input_size)
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return int(len(self.indices))

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __getitem__(self, item: int) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        f = self._file()
        idx = int(self.indices[item])
        token_ids, mask = prepare_token_ids(
            f["sequences"][idx],
            vocab_size=self.vocab_size,
            max_length=self.max_length,
            pad_id=self.pad_id,
        )
        image = f["images"][idx].astype(np.float32)
        image = _stored_image_to_rgb_tensor(image)
        if image.shape[-1] != self.input_size or image.shape[-2] != self.input_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        image = _imagenet_normalize(image)
        labels = f["genres"][idx].astype(np.float32)
        return {
            "index": idx,
            "tokens": torch.from_numpy(token_ids),
            "mask": torch.from_numpy(mask),
            "image": image,
            "labels": torch.from_numpy(labels),
        }


def _stored_image_to_rgb_tensor(image_chw: np.ndarray):
    import torch

    image = image_chw.astype(np.float32).copy()
    image += np.asarray(CAFFE_BGR_MEAN, dtype=np.float32).reshape(3, 1, 1)
    image = image[[2, 1, 0], :, :]
    image = np.clip(image, 0, 255) / 255.0
    return torch.from_numpy(image.astype(np.float32))


def _imagenet_normalize(image):
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
    return (image - mean) / std


class TextCNNEncoder:
    def __init__(self, *args, **kwargs):
        import torch.nn as nn

        class _TextCNN(nn.Module):
            def __init__(
                self,
                embedding_matrix: np.ndarray,
                channels: int,
                kernels: tuple[int, ...],
                output_dim: int,
                freeze_embeddings: bool,
                pad_id: int,
            ) -> None:
                super().__init__()
                weights = torch_tensor(embedding_matrix)
                self.embedding = nn.Embedding.from_pretrained(
                    weights,
                    freeze=freeze_embeddings,
                    padding_idx=pad_id,
                )
                embedding_dim = int(embedding_matrix.shape[1])
                self.convs = nn.ModuleList(
                    [
                        nn.Conv1d(embedding_dim, channels, kernel_size=k, padding=k // 2)
                        for k in kernels
                    ]
                )
                self.dropout = nn.Dropout(0.2)
                self.proj = nn.Linear(channels * len(kernels), output_dim)

            def forward(self, tokens, mask):
                import torch
                import torch.nn.functional as F

                emb = self.embedding(tokens)
                x = emb.transpose(1, 2)
                pooled = []
                for conv in self.convs:
                    h = F.relu(conv(x))
                    if h.shape[-1] > mask.shape[-1]:
                        h = h[:, :, : mask.shape[-1]]
                    elif h.shape[-1] < mask.shape[-1]:
                        mask_use = mask[:, : h.shape[-1]]
                    else:
                        mask_use = mask
                    if h.shape[-1] == mask.shape[-1]:
                        mask_use = mask
                    h = h.masked_fill(mask_use.unsqueeze(1) == 0, -1e4)
                    pooled.append(torch.max(h, dim=-1).values)
                h = torch.cat(pooled, dim=1)
                return self.proj(self.dropout(h))

        self.cls = _TextCNN

    def __call__(self, *args, **kwargs):
        return self.cls(*args, **kwargs)


class BiGRUAttentionEncoder:
    def __init__(self, *args, **kwargs):
        import torch.nn as nn

        class _BiGRUAttention(nn.Module):
            def __init__(
                self,
                embedding_matrix: np.ndarray,
                rnn_hidden_dim: int,
                rnn_layers: int,
                rnn_dropout: float,
                attention_dim: int,
                output_dim: int,
                freeze_embeddings: bool,
                pad_id: int,
            ) -> None:
                super().__init__()
                weights = torch_tensor(embedding_matrix)
                self.embedding = nn.Embedding.from_pretrained(
                    weights,
                    freeze=freeze_embeddings,
                    padding_idx=pad_id,
                )
                embedding_dim = int(embedding_matrix.shape[1])
                self.dropout = nn.Dropout(float(rnn_dropout))
                self.gru = nn.GRU(
                    input_size=embedding_dim,
                    hidden_size=int(rnn_hidden_dim),
                    num_layers=int(rnn_layers),
                    batch_first=True,
                    bidirectional=True,
                    dropout=float(rnn_dropout) if int(rnn_layers) > 1 else 0.0,
                )
                rnn_out_dim = int(rnn_hidden_dim) * 2
                self.attention = nn.Sequential(
                    nn.Linear(rnn_out_dim, int(attention_dim)),
                    nn.Tanh(),
                    nn.Linear(int(attention_dim), 1),
                )
                self.proj = nn.Sequential(
                    nn.Dropout(float(rnn_dropout)),
                    nn.Linear(rnn_out_dim, output_dim),
                )

            def forward(self, tokens, mask):
                import torch

                emb = self.dropout(self.embedding(tokens))
                lengths = mask.sum(dim=1).clamp_min(1).to(dtype=torch.long).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    emb,
                    lengths,
                    batch_first=True,
                    enforce_sorted=False,
                )
                packed_out, _ = self.gru(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_out,
                    batch_first=True,
                    total_length=tokens.shape[1],
                )
                scores = self.attention(out).squeeze(-1)
                scores = scores.masked_fill(mask == 0, -1e4)
                weights = torch.softmax(scores, dim=1)
                pooled = torch.sum(out * weights.unsqueeze(-1), dim=1)
                return self.proj(pooled)

        self.cls = _BiGRUAttention

    def __call__(self, *args, **kwargs):
        return self.cls(*args, **kwargs)


class TransformerTextEncoder:
    def __init__(self, *args, **kwargs):
        import torch.nn as nn

        class _TransformerText(nn.Module):
            def __init__(
                self,
                embedding_matrix: np.ndarray,
                max_length: int,
                num_layers: int,
                num_heads: int,
                ff_dim: int,
                dropout: float,
                output_dim: int,
                freeze_embeddings: bool,
                pad_id: int,
            ) -> None:
                super().__init__()
                weights = torch_tensor(embedding_matrix)
                self.embedding = nn.Embedding.from_pretrained(
                    weights,
                    freeze=freeze_embeddings,
                    padding_idx=pad_id,
                )
                embedding_dim = int(embedding_matrix.shape[1])
                if embedding_dim % int(num_heads) != 0:
                    raise ValueError(
                        f"text_transformer_heads={num_heads} must divide embedding_dim={embedding_dim}."
                    )
                self.position_embedding = nn.Embedding(int(max_length), embedding_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=int(num_heads),
                    dim_feedforward=int(ff_dim),
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=int(num_layers),
                    enable_nested_tensor=False,
                )
                self.dropout = nn.Dropout(float(dropout))
                self.proj = nn.Linear(embedding_dim, output_dim)

            def forward(self, tokens, mask):
                import torch

                positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
                x = self.embedding(tokens) + self.position_embedding(positions)
                x = self.dropout(x)
                key_padding_mask = mask == 0
                h = self.encoder(x, src_key_padding_mask=key_padding_mask)
                mask_f = mask.unsqueeze(-1)
                pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
                return self.proj(self.dropout(pooled))

        self.cls = _TransformerText

    def __call__(self, *args, **kwargs):
        return self.cls(*args, **kwargs)


def torch_tensor(arr: np.ndarray):
    import torch

    return torch.tensor(arr, dtype=torch.float32)


def build_text_encoder(embedding_matrix: np.ndarray, pad_id: int, cfg: NeuralConfig):
    name = cfg.text_encoder.lower()
    if name == "textcnn":
        return TextCNNEncoder()(
            embedding_matrix=embedding_matrix,
            channels=cfg.text_cnn_channels,
            kernels=cfg.text_cnn_kernels,
            output_dim=cfg.hidden_dim,
            freeze_embeddings=cfg.freeze_embeddings,
            pad_id=pad_id,
        )
    if name in {"bigru", "bigru_attention", "gru_attention"}:
        return BiGRUAttentionEncoder()(
            embedding_matrix=embedding_matrix,
            rnn_hidden_dim=cfg.text_rnn_hidden_dim,
            rnn_layers=cfg.text_rnn_layers,
            rnn_dropout=cfg.text_rnn_dropout,
            attention_dim=cfg.text_attention_dim,
            output_dim=cfg.hidden_dim,
            freeze_embeddings=cfg.freeze_embeddings,
            pad_id=pad_id,
        )
    if name in {"transformer", "transformer_encoder"}:
        return TransformerTextEncoder()(
            embedding_matrix=embedding_matrix,
            max_length=cfg.max_length,
            num_layers=cfg.text_transformer_layers,
            num_heads=cfg.text_transformer_heads,
            ff_dim=cfg.text_transformer_ff_dim,
            dropout=cfg.text_transformer_dropout,
            output_dim=cfg.hidden_dim,
            freeze_embeddings=cfg.freeze_embeddings,
            pad_id=pad_id,
        )
    raise ValueError(f"Unsupported text encoder: {cfg.text_encoder}")


def build_image_encoder(name: str, pretrained: bool, freeze_backbone: bool):
    import torch.nn as nn
    from torchvision import models

    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        out_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        out_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        out_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported image encoder: {name}")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    return model, int(out_dim)


class MultimodalGenreModel:
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        pad_id: int,
        num_labels: int,
        cfg: NeuralConfig,
    ) -> None:
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.modality = cfg.modality.lower()
                if self.modality not in {"multimodal", "text_only", "image_only"}:
                    raise ValueError(f"Unsupported modality: {cfg.modality}")
                self.uses_text = self.modality in {"multimodal", "text_only"}
                self.uses_image = self.modality in {"multimodal", "image_only"}
                if self.uses_text:
                    self.text_encoder = build_text_encoder(embedding_matrix, pad_id, cfg)
                else:
                    self.text_encoder = None
                if self.uses_image:
                    self.image_encoder, image_dim = build_image_encoder(
                        cfg.image_encoder,
                        pretrained=cfg.pretrained_image,
                        freeze_backbone=cfg.freeze_image_backbone,
                    )
                    self.image_proj = nn.Sequential(
                        nn.Linear(image_dim, cfg.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                    )
                else:
                    self.image_encoder = None
                    self.image_proj = None
                self.fusion = cfg.fusion.lower()
                if self.modality != "multimodal":
                    classifier_input = cfg.hidden_dim
                elif self.fusion == "gmu":
                    self.text_gate_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
                    self.image_gate_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
                    self.gate = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)
                    classifier_input = cfg.hidden_dim
                elif self.fusion == "concat":
                    classifier_input = cfg.hidden_dim * 2
                else:
                    raise ValueError(f"Unsupported fusion: {cfg.fusion}")
                self.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(classifier_input, cfg.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(cfg.hidden_dim, num_labels),
                )
                self.enable_label_correlation = bool(cfg.enable_label_correlation)
                if self.enable_label_correlation:
                    self.label_correlation = nn.Linear(num_labels, num_labels, bias=False)
                    nn.init.zeros_(self.label_correlation.weight)

            def forward(self, tokens, mask, image, return_gate: bool = False):
                import torch

                gate_value = None
                if self.modality == "text_only":
                    fused = self.text_encoder(tokens, mask)
                elif self.modality == "image_only":
                    fused = self.image_proj(self.image_encoder(image))
                elif self.fusion == "gmu":
                    text_h = self.text_encoder(tokens, mask)
                    image_h = self.image_proj(self.image_encoder(image))
                    text_z = torch.tanh(self.text_gate_proj(text_h))
                    image_z = torch.tanh(self.image_gate_proj(image_h))
                    gate_value = torch.sigmoid(self.gate(torch.cat([text_h, image_h], dim=1)))
                    fused = gate_value * text_z + (1.0 - gate_value) * image_z
                else:
                    text_h = self.text_encoder(tokens, mask)
                    image_h = self.image_proj(self.image_encoder(image))
                    fused = torch.cat([text_h, image_h], dim=1)
                logits = self.classifier(fused)
                if self.enable_label_correlation:
                    logits = logits + self.label_correlation(logits)
                if return_gate:
                    return logits, gate_value
                return logits

        self.module = _Model()

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        return self


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def compute_pos_weight(y_train: np.ndarray, clip_max: float | None = None):
    import torch

    pos = y_train.sum(axis=0).astype(np.float32)
    neg = y_train.shape[0] - pos
    weights = neg / np.maximum(pos, 1.0)
    if clip_max is not None:
        weights = np.minimum(weights, float(clip_max))
    return torch.tensor(weights, dtype=torch.float32)


class FocalBCEWithLogitsLoss:
    def __init__(self, pos_weight, gamma: float = 2.0) -> None:
        import torch.nn as nn

        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        self.gamma = float(gamma)

    def __call__(self, logits, labels):
        import torch

        bce = self.bce(logits, labels)
        prob = torch.sigmoid(logits)
        p_t = prob * labels + (1.0 - prob) * (1.0 - labels)
        focal_weight = (1.0 - p_t).clamp_min(1e-6).pow(self.gamma)
        return (focal_weight * bce).mean()


def build_criterion(cfg: NeuralConfig, y_train: np.ndarray, device: str):
    import torch.nn as nn

    pos_weight = compute_pos_weight(y_train, clip_max=cfg.pos_weight_clip).to(device)
    loss_name = cfg.loss.lower()
    if loss_name in {"bce", "weighted_bce", "bce_with_logits"}:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_name in {"focal", "focal_bce", "focal_loss"}:
        return FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=cfg.focal_gamma)
    raise ValueError(f"Unsupported neural loss: {cfg.loss}")


def build_scheduler(optimizer, cfg: NeuralConfig):
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    scheduler_name = cfg.scheduler.lower()
    if scheduler_name in {"", "none", "off"}:
        return None
    if scheduler_name in {"plateau", "reduce_on_plateau", "reduce_lr_on_plateau"}:
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.min_learning_rate,
        )
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")


def run_epoch(model, loader, criterion, optimizer, device: str, train: bool) -> float:
    import torch

    model.train(train)
    total_loss = 0.0
    total = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        image = batch["image"].to(device)
        labels = batch["labels"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(tokens, mask, image)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu()) * labels.shape[0]
        total += labels.shape[0]
    return total_loss / max(total, 1)


def predict(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["tokens"].to(device),
                batch["mask"].to(device),
                batch["image"].to(device),
            )
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch["labels"].numpy())
    return np.vstack(labels), np.vstack(probs)


def train_neural_multimodal(
    hdf5_path: str | Path,
    metadata_path: str | Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray | None,
    output_dir: str | Path,
    cfg: NeuralConfig,
    seed: int = 42,
    limit: int | None = None,
) -> dict:
    import torch
    from torch.optim import AdamW

    set_seed(seed)
    if limit is not None:
        train_idx = train_idx[:limit]
        val_idx = val_idx[: max(1, min(len(val_idx), limit // 5))]
        if test_idx is not None:
            test_idx = test_idx[: max(1, min(len(test_idx), limit // 5))]

    metadata = load_metadata(metadata_path)
    embedding_matrix = build_embedding_matrix(metadata, seed=seed)
    pad_id = pad_token_id(metadata)
    vocab_size = int(metadata["vocab_size"])
    y = load_labels(hdf5_path)
    y_train = y[train_idx]

    train_ds = MMIMDBTorchDataset(hdf5_path, train_idx, vocab_size, cfg.max_length, cfg.input_size)
    val_ds = MMIMDBTorchDataset(hdf5_path, val_idx, vocab_size, cfg.max_length, cfg.input_size)
    test_ds = (
        MMIMDBTorchDataset(hdf5_path, test_idx, vocab_size, cfg.max_length, cfg.input_size)
        if test_idx is not None
        else None
    )

    train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = make_loader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = (
        make_loader(test_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        if test_ds is not None
        else None
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = MultimodalGenreModel(
        embedding_matrix=embedding_matrix,
        pad_id=pad_id,
        num_labels=len(GENRE_LABELS),
        cfg=cfg,
    ).to(device)
    model = wrapper.module

    criterion = build_criterion(cfg, y_train, device)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg)

    out = ensure_dir(output_dir)
    suffix = f"_limit{limit}" if limit is not None else ""
    checkpoint_path = out / f"{cfg.model_name}{suffix}.pt"
    history = []
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        y_val, val_prob = predict(model, val_loader, device)
        _, val_metrics = tune_thresholds(
            y_val,
            val_prob,
            metric=cfg.threshold_metric,
            strategy=cfg.threshold_strategy,
            threshold_min=cfg.threshold_min,
            threshold_max=cfg.threshold_max,
            threshold_steps=cfg.threshold_steps,
        )
        score = float(val_metrics[cfg.threshold_metric])
        if scheduler is not None:
            scheduler.step(score)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            cfg.threshold_metric: score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "genre_labels": GENRE_LABELS,
                    "pad_id": pad_id,
                    "vocab_size": vocab_size,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                break

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_val, val_prob = predict(model, val_loader, device)
    threshold, val_metrics = tune_thresholds(
        y_val,
        val_prob,
        metric=cfg.threshold_metric,
        strategy=cfg.threshold_strategy,
        threshold_min=cfg.threshold_min,
        threshold_max=cfg.threshold_max,
        threshold_steps=cfg.threshold_steps,
    )
    test_metrics = None
    if test_loader is not None:
        y_test, test_prob = predict(model, test_loader, device)
        test_metrics = multilabel_metrics(y_test, test_prob, threshold=threshold)
    threshold_saved = threshold_to_serializable(threshold)

    checkpoint["threshold"] = threshold_saved
    checkpoint["validation_metrics"] = val_metrics
    if test_metrics is not None:
        checkpoint["test_metrics"] = test_metrics
    torch.save(checkpoint, checkpoint_path)

    result = {
        "model_path": str(checkpoint_path),
        "device": device,
        "threshold": threshold_saved,
        "threshold_strategy": cfg.threshold_strategy,
        "threshold_grid": {
            "min": float(cfg.threshold_min),
            "max": float(cfg.threshold_max),
            "steps": int(cfg.threshold_steps),
        },
        "loss": cfg.loss,
        "scheduler": cfg.scheduler,
        "modality": cfg.modality,
        "enable_label_correlation": bool(cfg.enable_label_correlation),
        "best_epoch": int(best_epoch),
        "best_validation_score": float(best_score),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)) if test_idx is not None else 0,
        "history": history,
        "validation": val_metrics,
    }
    if test_metrics is not None:
        result["test"] = test_metrics
    save_json(result, out / f"{cfg.model_name}{suffix}_metrics.json")
    return result
