# Notebook Notes

The project is implemented as Python modules and scripts so notebooks can stay thin.

Colab-ready training notebooks:

- `notebooks/training_pipeline_neural_colab.ipynb`
- `notebooks/training_pipeline_classic_ml_colab.ipynb`

Colab-ready final evaluation/XAI notebook:

- `notebooks/best_model_test_inference_colab.ipynb`
- `notebooks/final_test_xai_all_models_colab.ipynb`
- `notebooks/global_xai_best_neural_colab.ipynb`

Global XAI notebook:

- `global_xai_best_neural_colab.ipynb` computes all-label global explanations for
  the canonical best neural model and saves artifacts for the dashboard's
  **Global XAI - Best Neural** tab.

Recommended workflow:

1. Activate the environment: `conda activate mmimdb-xai`
2. Start Jupyter from the project root: `jupyter notebook`
3. Import from `mmimdb` directly.

Minimal notebook cell:

```python
from mmimdb.data import DatasetPaths, load_labels
from mmimdb.utils import load_config

config = load_config("configs/default.yaml")
paths = DatasetPaths.from_config(config)
y = load_labels(paths.hdf5)
y.shape
```

For training from a notebook, prefer calling the script functions rather than duplicating logic:

```python
import numpy as np
from mmimdb.models.classic import ClassicConfig, train_classic_multimodal
from mmimdb.splits import load_split_indices

train_idx, val_idx, test_idx = load_split_indices("outputs/splits")
result = train_classic_multimodal(
    paths.hdf5,
    paths.metadata,
    train_idx,
    val_idx,
    test_idx,
    output_dir="outputs/models",
    cfg=ClassicConfig.from_config(config),
    limit=100,
)
result["validation"]["macro_f1"]
```
