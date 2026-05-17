"""Small shared utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    base = Path(root) if root is not None else project_root()
    return (base / p).resolve()


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            return yaml.safe_load(f)
        return json.load(f)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
