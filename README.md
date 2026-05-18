# MM-IMDb Multimodal Genre Classification

This project builds an XAI-ready multimodal genre classification pipeline for:

- `dataset/data/multimodal_imdb.hdf5`
- `dataset/data/metadata.npy`
- `dataset/Article about dataset and how was it used.pdf`

For a complete fresh-clone setup guide, including environment installation,
dataset download and placement, model training, Colab/GPU notebook training,
and single-instance XAI analysis, see:

- [HOW_TO_RUN.md](HOW_TO_RUN.md)

The dataset, trained model checkpoints, and generated reports/figures are intentionally not tracked in Git. Place the MM-IMDb files in the paths above before running the pipeline; scripts will recreate `outputs/` locally.

Final comparison models are multimodal. Single-modality models are allowed only as baselines or ablations.

## Dataset

Download MM-IMDb from Kaggle:

- [johnarevalo/mmimdb](https://www.kaggle.com/datasets/johnarevalo/mmimdb/data)

After downloading, put the files in this layout:

```text
dataset/
  data/
    multimodal_imdb.hdf5
    metadata.npy
  Article about dataset and how was it used.pdf
```

The HDF5 and metadata files are required. The PDF is used only when regenerating the technical dataset report.

If you use the Kaggle CLI, first configure your Kaggle API token, then run:

```powershell
python -m pip install kaggle
kaggle datasets download -d johnarevalo/mmimdb -p dataset\_kaggle_download --unzip
New-Item -ItemType Directory -Force dataset\data
Move-Item dataset\_kaggle_download\multimodal_imdb.hdf5 dataset\data\
Move-Item dataset\_kaggle_download\metadata.npy dataset\data\
```

## Environment

Create and activate the conda environment:

```powershell
conda env create -f environment.yml
conda activate mmimdb-xai
python -m pip install -e .
```

The environment has already been created on this machine during setup.

## Configuration

Most parameters and hyperparameters live in categorized sections in:

- `configs/default.yaml`

Guide:

- `docs/config_guide.md`

## Dataset Inspection

```powershell
python scripts\inspect_dataset.py
```

## Splits

Creates a new iterative multilabel stratified split:

- 70% train
- 15% validation
- 15% test

```powershell
python scripts\create_splits.py
```

Outputs:

- `outputs/splits/train_indices.npy`
- `outputs/splits/val_indices.npy`
- `outputs/splits/test_indices.npy`
- `outputs/splits/split_metadata.json`

## Training Model Selection

Recommended one-command training workflow:

```powershell
python scripts\train_models.py
```

This command keeps training separate from XAI:

- combines the existing train and validation indices for cross-fold model selection,
- tries the neural candidate models listed under `training` in `configs/default.yaml`,
- trains the default classic model once by default because `training.classic.final_only: true` avoids repeated slow CPU-bound classic CV,
- ranks candidates by the configured metric, currently `macro_f1`,
- trains final selected classic/neural models using the original train/validation split for threshold tuning and neural early stopping,
- evaluates the final selected models on the test split,
- updates the best-model registry only for full runs.

Main outputs:

- `outputs/model_selection/training_summary.json`
- `outputs/model_selection/classic/cv_summary.json`
- `outputs/model_selection/neural/cv_summary.json`
- `outputs/models/best/classic_multimodal_best.joblib`
- `outputs/models/best/neural_multimodal_best.pt`

Smoke-test only the classic search:

```powershell
python scripts\train_models.py --model-type classic --limit 100 --folds 2
```

Smoke-test both model families:

```powershell
python scripts\train_models.py --model-type both --limit 12 --folds 2 --epochs 1 --batch-size 2 --no-pretrained-image
```

Smoke-test the current neural candidate set in the project conda environment:

```powershell
conda run -n mmimdb-xai python scripts\train_models.py --model-type neural --limit 8 --folds 2 --epochs 1 --batch-size 2 --no-pretrained-image
```

Runs with `--limit` do not overwrite the best-model registry.

## Technical Report

```powershell
python scripts\build_report.py
```

Output:

- `docs/dataset_preprocessing_report.md`
- `outputs/figures/genre_label_counts.png`
- `outputs/figures/genre_cooccurrence.png`

## Classic Multimodal Model

The one-command workflow trains the classic model in final-only mode by default, which is the fastest Colab path. The standalone script remains useful for a single classic baseline run.

Model:

- text: reconstructed plot TF-IDF
- image: reversible poster color/thumbnail descriptors, reduced by default for faster CPU training
- fusion: concatenated features
- classifier: One-vs-Rest `SGDClassifier(loss="log_loss")` by default for faster CPU training, with Logistic Regression and ClassifierChain still available by config
- thresholds: per-label validation-tuned thresholds by default

Classic scikit-learn training does not use the Colab GPU. For speed, use the default SGD setup and avoid classic CV unless you explicitly need a classical hyperparameter comparison.

Smoke test:

```powershell
python scripts\train_classic.py --limit 100
```

Full run:

```powershell
python scripts\train_classic.py
```

Full run through the combined pipeline:

```powershell
python scripts\train_models.py --model-type classic
```

Full runs update the best-model registry when their configured validation metric improves:

- `outputs/models/best/classic_multimodal_best.joblib`
- `outputs/models/best/classic_multimodal_best_metrics.json`

Runs with `--limit` are treated as smoke tests and do not overwrite best-model registry files.

## Neural Multimodal Model

The one-command model-selection workflow above is preferred for final training. This script remains useful for a single neural architecture run.

Model:

- text branch: Word2Vec-initialized lightweight Transformer for the current selected candidates, with BiGRU-attention and TextCNN still available by config
- image branch: pretrained `torchvision` ResNet18 by default, with EfficientNet-B0 available for the unfrozen image-backbone experiment
- fusion: GMU-style gated fusion for multimodal candidates
- classifier: 23 sigmoid multilabel outputs
- thresholding: per-label validation-tuned thresholds on a fine 0.01-0.99 grid
- optional scheduler: `scheduler: plateau`
- optional rare-label experiment: `loss: focal` with clipped positive weights
- optional experimental residual label-correlation head, controlled by config

The configured neural candidate list includes:

- best previous baseline: Transformer + GMU + ResNet18 at learning rate `2e-4`
- same baseline with `ReduceLROnPlateau`
- Transformer + GMU + unfrozen EfficientNet-B0
- text-only and image-only ablations, marked `selection_eligible: false`
- focal-loss variant for rare-label macro-F1 experiments

Smoke test:

```powershell
python scripts\train_neural.py --limit 8 --epochs 1 --batch-size 2
```

Full run:

```powershell
python scripts\train_neural.py
```

The neural trainer saves only the best checkpoint for a run, selected by the configured validation metric. The checkpoint also stores the tuned threshold and final validation/test metrics.

Full neural runs also update:

- `outputs/models/best/neural_multimodal_best.pt`
- `outputs/models/best/neural_multimodal_best_metrics.json`

Runs with `--limit` are treated as smoke tests and do not overwrite best-model registry files.

Note: the created environment currently reports `cuda = False`, so full neural training will run on CPU unless the environment is switched to a CUDA-enabled PyTorch build.

## XAI

XAI is intentionally a separate step. After `scripts\train_models.py` has selected and registered the best models, run XAI against those saved artifacts:

```powershell
python scripts\run_xai.py --model-type both --limit 10
```

Research summary:

- `docs/xai_research.md`
- `docs/neural_text_encoder_research.md`

Implemented explanation methods:

- Classic multimodal linear classifier:
  - linear TF-IDF word/ngram contributions
  - linear poster descriptor contributions
  - classifier-chain label dependency contributions when enabled
  - text-vs-image logit contribution totals
  - two-modality Shapley utilization for text/image reliance
- Neural multimodal model:
  - Layer Integrated Gradients for plot tokens through the active embedding-based text encoder
  - token occlusion for plot tokens
  - Integrated Gradients for image pixels
  - Grad-CAM for the ResNet image branch
  - image patch occlusion sensitivity
  - modality ablation
  - two-modality Shapley utilization for text/image reliance
  - GMU gate summary
  - experimental label-set attributions over a selected group of predicted/top-k/explicit genres

Smoke test with saved limited models:

```powershell
python scripts\run_xai.py --model-type both --limit 10 --classic-model outputs\models\classic_multimodal_limit10.joblib --neural-checkpoint outputs\models\multimodal_bigru_attention_resnet18_gmu_limit10.pt --target-genre Crime --n-steps 2
```

Multi-target examples:

```powershell
python scripts\run_xai.py --model-type both --limit 5 --target-policy predicted --max-targets-per-sample 3
python scripts\run_xai.py --model-type both --limit 5 --target-policy top_k --target-top-k 3
python scripts\run_xai.py --model-type both --limit 5 --target-genres Crime,Drama,Thriller
```

For faster debugging, add `--no-occlusion` or reduce the poster patch grid with `--image-occlusion-grid 2`.

Full model XAI after full training or model selection:

```powershell
python scripts\run_xai.py --model-type both --limit 10
```

By default, full XAI uses the best-model registry paths from `configs/default.yaml`.

Local XAI dashboard:

```powershell
python scripts\xai_dashboard.py --port 8050
```

Open `http://127.0.0.1:8050` to browse saved XAI reports, inspect poster heatmaps/patch occlusion regions, compare highest-impact text/image features, and run saved-model inference for a dataset index.

XAI metrics saved in each summary:

- wall-clock time
- process CPU time
- process CPU utilization estimate
- RSS memory start/end/delta/peak
- CUDA memory stats when CUDA is available
- output file counts and output directory size

`process_cpu_util_percent` can exceed 100% when the method uses multiple CPU threads.

Outputs:

- `outputs/xai/classic/.../explanation.json`
- `outputs/xai/classic/.../top_text_features.png`
- `outputs/xai/classic/.../thumbnail_descriptor_heatmap.png`
- `outputs/xai/classic/.../modality_shapley.png`
- `outputs/xai/neural/.../explanation.json`
- `outputs/xai/neural/.../top_token_attributions.png`
- `outputs/xai/neural/.../token_occlusion_attributions.png`
- `outputs/xai/neural/.../integrated_gradients_image_overlay.png`
- `outputs/xai/neural/.../gradcam_overlay.png`
- `outputs/xai/neural/.../image_occlusion_sensitivity_overlay.png`
- `outputs/xai/neural/.../modality_shapley.png`

## Jupyter Usage

After activating `mmimdb-xai`, start Jupyter from the project root:

```powershell
jupyter notebook
```

Then import the project code:

```python
from mmimdb.data import DatasetPaths, load_labels
from mmimdb.utils import load_config

config = load_config("configs/default.yaml")
paths = DatasetPaths.from_config(config)
y = load_labels(paths.hdf5)
y.shape
```

## Google Colab Training

Use one of the two Colab-ready notebooks:

- Neural model training: [notebooks/training_pipeline_neural_colab.ipynb](https://colab.research.google.com/github/SamuelSulan/Explaining-Predictions-of-Machine-Learning-Models/blob/main/notebooks/training_pipeline_neural_colab.ipynb)
- Classic ML training: [notebooks/training_pipeline_classic_ml_colab.ipynb](https://colab.research.google.com/github/SamuelSulan/Explaining-Predictions-of-Machine-Learning-Models/blob/main/notebooks/training_pipeline_classic_ml_colab.ipynb)

Use a GPU runtime for the neural notebook when running full training.

The setup cell can:

- mount Google Drive,
- use or clone the repository at `/content/drive/MyDrive/Explaining-Predictions-of-Machine-Learning-Models`,
- install notebook dependencies without replacing Colab's CUDA-enabled PyTorch,
- override dataset paths through `COLAB_DATA_ROOT`, `COLAB_HDF5_PATH`, and `COLAB_METADATA_PATH`,
- optionally persist outputs through `COLAB_OUTPUT_DIR`.

If your repository or dataset is stored elsewhere on Drive, edit those `COLAB_*` constants in the first setup cell, then run the notebook from top to bottom.

## XAI Readiness

The current pipeline keeps:

- token IDs and `ix_to_word` mapping for text attribution,
- restored RGB poster reconstruction for image heatmaps,
- model branches separated for text/image ablation,
- GMU-style fusion gate values available in the neural model.

Further XAI extensions can add:

- SHAP/LIME-style surrogate reports for non-linear tabular baselines,
- Grad-CAM++ or concept-level explanations for richer poster semantics,
- interaction-aware reports beyond two-modality Shapley when more modalities or aligned concepts are available.
