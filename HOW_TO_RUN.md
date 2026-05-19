# How to Run This Repository

This repository trains and explains multimodal genre classifiers on the
MM-IMDb dataset. The dataset and trained model files are not committed to Git,
so a fresh download of the repository needs three things before the full
pipeline can run:

1. A Python environment with the required packages.
2. The MM-IMDb dataset files in the expected local folder.
3. Generated train/validation/test splits and trained model artifacts.

The commands below assume you run them from the repository root.

```powershell
cd path\to\diplomka
```

On Linux/macOS, replace backslashes in paths with forward slashes.

## 1. Repository Layout

Important files and folders:

```text
configs/default.yaml        Main configuration file
dataset/                    Local dataset folder, not tracked by Git
outputs/                    Generated splits, models, reports, and XAI outputs
scripts/create_splits.py    Creates train/validation/test split files
scripts/train_models.py     Recommended full training/model-selection command
scripts/train_classic.py    Standalone classic ML training command
scripts/train_neural.py     Standalone neural training command
scripts/run_xai.py          Generates XAI explanations from saved models
scripts/xai_dashboard.py    Local dashboard for XAI reports and inference
src/mmimdb/                 Project Python package
```

The local dataset files in `dataset/` and trained model checkpoint aliases under
`outputs/models/best/` are not committed to Git. Some generated reports,
figures, metrics, and XAI examples under `outputs/` are committed as project
artifacts, and new local runs can create additional output folders. A user who
downloads the repository must still download the dataset and train models
locally, or copy already trained model files into the expected
`outputs/models/best/` paths.

## 2. Install Python Requirements

The recommended setup uses conda.

```powershell
conda env create -f environment.yml
conda activate mmimdb-xai
python -m pip install -e .
```

The editable install (`python -m pip install -e .`) makes the `mmimdb` package
available while keeping the source code editable.

If you do not use conda, create a Python 3.10+ environment and install the pip
requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Linux/macOS virtual environment activation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The provided `environment.yml` installs CPU-only PyTorch. Full neural training
is much faster with a CUDA-enabled PyTorch build, especially on Google Colab or
a local machine with an NVIDIA GPU.

## 3. Download and Place the Dataset

Download the MM-IMDb dataset from Kaggle:

[https://www.kaggle.com/datasets/johnarevalo/mmimdb/data](https://www.kaggle.com/datasets/johnarevalo/mmimdb/data)

After downloading, place the required files exactly like this:

```text
dataset/
  data/
    multimodal_imdb.hdf5
    metadata.npy
  Article about dataset and how was it used.pdf
```

Required files:

- `dataset/data/multimodal_imdb.hdf5`
- `dataset/data/metadata.npy`

Optional file:

- `dataset/Article about dataset and how was it used.pdf`

The PDF is only needed when regenerating the dataset/preprocessing report.

### Download with Kaggle CLI

First configure your Kaggle API token according to Kaggle's instructions. Then:

```powershell
python -m pip install kaggle
kaggle datasets download -d johnarevalo/mmimdb -p dataset\_kaggle_download --unzip
New-Item -ItemType Directory -Force dataset\data
Move-Item dataset\_kaggle_download\multimodal_imdb.hdf5 dataset\data\
Move-Item dataset\_kaggle_download\metadata.npy dataset\data\
```

If the downloaded archive contains the PDF, move it to:

```text
dataset/Article about dataset and how was it used.pdf
```

## 4. Check the Configuration

Default paths, model settings, split settings, training hyperparameters, and XAI
settings are in:

```text
configs/default.yaml
```

The default dataset paths are:

```yaml
paths:
  hdf5: dataset/data/multimodal_imdb.hdf5
  metadata: dataset/data/metadata.npy
  article_pdf: dataset/Article about dataset and how was it used.pdf
```

If you keep the dataset somewhere else, update these paths before running the
pipeline.

## 5. Inspect the Dataset

After placing the dataset files, run:

```powershell
python scripts\inspect_dataset.py
```

This is a quick sanity check that the HDF5 and metadata files can be loaded.

## 6. Create Train/Validation/Test Splits

Create reproducible multilabel stratified splits:

```powershell
python scripts\create_splits.py
```

This writes:

```text
outputs/splits/train_indices.npy
outputs/splits/val_indices.npy
outputs/splits/test_indices.npy
outputs/splits/split_metadata.json
```

The configured split ratio is:

- 70% train
- 15% validation
- 15% test

Training scripts can create missing splits automatically, but running this
explicitly is recommended for a fresh setup.

## 7. Get Trained Models

Because trained models are not tracked in Git, there are two ways to get them.

### Option A: Train the Models Yourself

The recommended command is:

```powershell
python scripts\train_models.py
```

This runs model selection and final training for the configured classic and
neural multimodal models. It also updates the best-model registry for full
runs.

Main outputs:

```text
outputs/model_selection/training_summary.json
outputs/model_selection/classic/cv_summary.json
outputs/model_selection/neural/cv_summary.json
outputs/models/best/classic_multimodal_best.joblib
outputs/models/best/classic_multimodal_best_metrics.json
outputs/models/best/neural_multimodal_best.pt
outputs/models/best/neural_multimodal_best_metrics.json
```

These two files are the default model artifacts used by XAI:

```text
outputs/models/best/classic_multimodal_best.joblib
outputs/models/best/neural_multimodal_best.pt
```

### Option B: Copy Already Trained Models

If someone gives you trained artifacts, copy them into:

```text
outputs/models/best/classic_multimodal_best.joblib
outputs/models/best/neural_multimodal_best.pt
```

Then XAI can run without retraining, as long as the artifacts were trained with
compatible code, configuration, dataset, and label order.

## 8. Smoke-Test Training

Before a long full run, use a small smoke test.

Classic only:

```powershell
python scripts\train_models.py --model-type classic --limit 100 --folds 2
```

Both classic and neural:

```powershell
python scripts\train_models.py --model-type both --limit 12 --folds 2 --epochs 1 --batch-size 2 --no-pretrained-image
```

Neural only:

```powershell
python scripts\train_models.py --model-type neural --limit 8 --folds 2 --epochs 1 --batch-size 2 --no-pretrained-image
```

Runs with `--limit` are smoke tests and do not overwrite the best-model
registry.

## 9. Train Only One Model Family

Classic multimodal model:

```powershell
python scripts\train_classic.py
```

Classic smoke test:

```powershell
python scripts\train_classic.py --limit 100
```

Neural multimodal model:

```powershell
python scripts\train_neural.py
```

Neural smoke test:

```powershell
python scripts\train_neural.py --limit 8 --epochs 1 --batch-size 2
```

Full standalone runs also update the best-model registry. Limited runs do not.

## 10. Run XAI on a Single Dataset Instance

XAI is a separate step from training. It expects saved model artifacts to exist,
usually in:

```text
outputs/models/best/classic_multimodal_best.joblib
outputs/models/best/neural_multimodal_best.pt
```

To explain one instance from the configured test split, run:

```powershell
python scripts\run_xai.py --model-type both --split test --limit 1
```

This explains the first item in the selected split. You can choose a different
split with `--split train`, `--split val`, or `--split test`.

For a faster single-instance XAI smoke run, reduce expensive attribution work:

```powershell
python scripts\run_xai.py --model-type both --split test --limit 1 --n-steps 2 --image-occlusion-grid 2
```

To disable occlusion methods entirely:

```powershell
python scripts\run_xai.py --model-type both --split test --limit 1 --n-steps 2 --no-occlusion
```

To explain only one model type:

```powershell
python scripts\run_xai.py --model-type classic --split test --limit 1
python scripts\run_xai.py --model-type neural --split test --limit 1
```

To explain a specific target genre:

```powershell
python scripts\run_xai.py --model-type both --split test --limit 1 --target-genre Crime
```

To explain several explicit target genres:

```powershell
python scripts\run_xai.py --model-type both --split test --limit 1 --target-genres Crime,Drama,Thriller
```

To explain the top predicted labels:

```powershell
python scripts\run_xai.py --model-type both --split test --limit 1 --target-policy predicted --max-targets-per-sample 3
```

To explain the top-k labels by probability:

```powershell
python scripts\run_xai.py --model-type both --split test --limit 1 --target-policy top_k --target-top-k 3
```

The current XAI CLI selects instances from the saved split arrays and uses
`--limit` to control how many are explained. For arbitrary saved-model inference
on a specific dataset index, use the dashboard described below.

## 11. XAI Outputs

XAI results are written under:

```text
outputs/xai/
```

Typical classic outputs:

```text
outputs/xai/classic/classic_xai_summary.json
outputs/xai/classic/classic_idx_<index>_<genre>/explanation.json
outputs/xai/classic/classic_idx_<index>_<genre>/poster.png
outputs/xai/classic/classic_idx_<index>_<genre>/top_text_features.png
outputs/xai/classic/classic_idx_<index>_<genre>/thumbnail_descriptor_heatmap.png
outputs/xai/classic/classic_idx_<index>_<genre>/modality_shapley.png
```

Typical neural outputs:

```text
outputs/xai/neural/neural_xai_summary.json
outputs/xai/neural/neural_idx_<index>_<genre>/explanation.json
outputs/xai/neural/neural_idx_<index>_<genre>/poster.png
outputs/xai/neural/neural_idx_<index>_<genre>/top_token_attributions.png
outputs/xai/neural/neural_idx_<index>_<genre>/token_occlusion_attributions.png
outputs/xai/neural/neural_idx_<index>_<genre>/integrated_gradients_image_overlay.png
outputs/xai/neural/neural_idx_<index>_<genre>/gradcam_overlay.png
outputs/xai/neural/neural_idx_<index>_<genre>/image_occlusion_sensitivity_overlay.png
outputs/xai/neural/neural_idx_<index>_<genre>/modality_shapley.png
```

## 12. Open the Local XAI Dashboard

After generating XAI outputs, start the dashboard:

```powershell
python scripts\xai_dashboard.py --port 8050
```

Open:

[http://127.0.0.1:8050](http://127.0.0.1:8050)

The dashboard lets you:

- Browse saved XAI explanation reports.
- View poster heatmaps and attribution plots.
- Compare text and image contributions.
- Run saved-model inference for a chosen dataset index.

The dashboard uses the dataset paths and default model paths from
`configs/default.yaml`.

## 13. Train with Colab or Another GPU Machine

The repository includes Colab-ready notebooks that can be used for training on
Google Colab or on another machine with a GPU:

- Neural model training:
  [notebooks/training_pipeline_neural_colab.ipynb](https://colab.research.google.com/github/SamuelSulan/Explaining-Predictions-of-Machine-Learning-Models/blob/main/notebooks/training_pipeline_neural_colab.ipynb)
- Classic ML training:
  [notebooks/training_pipeline_classic_ml_colab.ipynb](https://colab.research.google.com/github/SamuelSulan/Explaining-Predictions-of-Machine-Learning-Models/blob/main/notebooks/training_pipeline_classic_ml_colab.ipynb)

Use the neural notebook with a GPU runtime for full neural training. In Colab,
select:

```text
Runtime -> Change runtime type -> GPU
```

The classic ML notebook can also be run in Colab, but the classic scikit-learn
pipeline is CPU-based and does not benefit from GPU acceleration in the same way
as the PyTorch neural model.

The notebook setup cells are designed to:

- Mount Google Drive.
- Use or clone the repository in Google Drive.
- Install notebook dependencies without replacing Colab's CUDA-enabled PyTorch.
- Let you override dataset and output locations with `COLAB_*` path constants.
- Persist trained models and outputs in Drive.

If your repository or dataset is stored somewhere else, edit the `COLAB_*`
constants in the first setup cell before running the notebook.

After training on Colab or another GPU machine, make sure the trained artifacts
are available locally or in the expected repository output paths before running
XAI:

```text
outputs/models/best/classic_multimodal_best.joblib
outputs/models/best/neural_multimodal_best.pt
```

If you train on a remote machine, copy the relevant files back into
`outputs/models/best/` in this repository checkout, or pass explicit paths to
`scripts/run_xai.py` with `--classic-model` and `--neural-checkpoint`.

## 14. Regenerate Dataset/Preprocessing Report

If the dataset PDF is present, you can regenerate the report and figures:

```powershell
python scripts\build_report.py
```

Outputs include:

```text
docs/dataset_preprocessing_report.md
outputs/figures/genre_label_counts.png
outputs/figures/genre_cooccurrence.png
```

## 15. Recommended Fresh-Clone Workflow

For a new user starting from a clean repository download:

```powershell
conda env create -f environment.yml
conda activate mmimdb-xai
python -m pip install -e .

# Download the Kaggle dataset, then place:
# dataset/data/multimodal_imdb.hdf5
# dataset/data/metadata.npy

python scripts\inspect_dataset.py
python scripts\create_splits.py

# Optional quick check:
python scripts\train_models.py --model-type both --limit 12 --folds 2 --epochs 1 --batch-size 2 --no-pretrained-image

# Full training:
python scripts\train_models.py

# Single-instance XAI:
python scripts\run_xai.py --model-type both --split test --limit 1

# Browse results:
python scripts\xai_dashboard.py --port 8050
```

## 16. Common Problems

### Dataset file not found

Check that these files exist:

```text
dataset/data/multimodal_imdb.hdf5
dataset/data/metadata.npy
```

If you put the files somewhere else, update `configs/default.yaml`.

### XAI says model not found

Train the models first:

```powershell
python scripts\train_models.py
```

Or copy compatible trained artifacts to:

```text
outputs/models/best/classic_multimodal_best.joblib
outputs/models/best/neural_multimodal_best.pt
```

### Full neural training is slow

The default conda environment uses CPU-only PyTorch. Use a CUDA-enabled PyTorch
installation or run the neural notebook in Google Colab with a GPU runtime.

### Smoke-test model files are not used by default XAI

Limited training runs do not update:

```text
outputs/models/best/
```

If you want to run XAI on smoke-test artifacts, pass their paths explicitly:

```powershell
python scripts\run_xai.py --model-type both --limit 1 --classic-model outputs\models\classic_multimodal_limit10.joblib --neural-checkpoint outputs\models\multimodal_bigru_attention_resnet18_gmu_limit10.pt --n-steps 2
```

Adjust the filenames to match the actual files created under `outputs/models/`.
