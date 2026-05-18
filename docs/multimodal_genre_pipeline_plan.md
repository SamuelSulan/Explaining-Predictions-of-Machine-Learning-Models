# Multimodal IMDb Genre Classification Pipeline Plan

## Summary

Build a full Python-based AI pipeline for multilabel movie genre classification on `dataset/data/multimodal_imdb.hdf5`, using the closest available original modalities in the dataset:

- `sequences`: tokenized movie plot text.
- `images`: preprocessed movie poster tensors.
- `genres`: 23-label multilabel genre targets.
- `imdb_ids`: stable movie identifiers.

The final trained models used for comparison must be multimodal. Single-modality models may be trained only as baselines, ablations, or diagnostic references. The pipeline should preserve enough preprocessing metadata to support later XAI methods that highlight influential plot tokens and poster regions.

## Environment

Create a dedicated conda environment, for example `mmimdb-xai`.

Required packages:

- `python`
- `pytorch`
- `torchvision`
- `numpy`
- `pandas`
- `h5py`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `seaborn`
- `joblib`
- `pypdf`
- `iterative-stratification`

Optional later XAI packages:

- `captum`
- `shap`
- Grad-CAM utility package or a small local Grad-CAM implementation.

The project should be implemented as Python scripts/modules that can also be imported and run from Jupyter notebooks.

## Dataset Facts From Inspection

The HDF5 file contains 25,959 movies.

Main datasets:

- `images`: shape `(25959, 3, 256, 160)`, stored poster tensors.
- `sequences`: shape `(25959,)`, tokenized plot sequences.
- `features`: shape `(25959, 300)`, precomputed Word2Vec-like plot features.
- `vgg_features`: shape `(25959, 4096)`, precomputed VGG poster features.
- `word_grams`: shape `(25959, 13253)`, text n-gram features.
- `three_grams`: shape `(25959, 9946)`, 3-gram text features.
- `genres`: shape `(25959, 23)`, binary multilabel targets.
- `imdb_ids`: shape `(25959,)`, IMDb IDs.

Metadata file `metadata.npy` contains:

- `word_to_ix`
- `ix_to_word`
- `lookup`
- `vocab_size`

Important observed details:

- `genres` is multilabel, not single-label multiclass.
- Every movie has at least one genre.
- Average genres per movie: approximately `2.48`.
- `metadata.npy` can map plot token IDs back to words for later text explanations.
- Poster tensors appear to be Caffe/VGG-style BGR pixels with channel means subtracted.
- Adding `[103.939, 116.779, 123.68]` to the image channels restores valid `0..255` poster pixels.

## Paper-Derived Context

The provided article describes the MM-IMDb dataset and the Gated Multimodal Unit (GMU) approach for genre classification using plot and poster information.

Important paper details to include in the thesis document:

- MM-IMDb was built from IMDb IDs connected to MovieLens 20M.
- Movies without poster images were filtered out.
- The dataset contains plot, poster, genres, and additional IMDb metadata.
- The paper frames the task as multilabel movie genre prediction from plot and poster.
- The original paper split was 60% train, 10% development, and 30% test.
- The original paper reported sample, micro, macro, and weighted F1.
- The original paper found multimodal fusion, especially GMU, stronger than single-modality approaches.

For this project, use a new split instead of the original paper split, while documenting the original split for comparison.

## Split Strategy

Create a new reproducible multilabel split.

Default split:

- 70% train
- 15% validation
- 15% test

Use iterative multilabel stratification to preserve genre proportions across splits.

Save:

- `train_indices.npy`
- `val_indices.npy`
- `test_indices.npy`
- split metadata JSON with random seed, split ratios, dataset path, timestamp, and label order.

The validation set must come from the training side of the data, not from the final test set. The test set remains untouched until final evaluation.

## Reversible Preprocessing

The preprocessing must preserve mappings needed for later XAI.

Text:

- Use `sequences` as the main text input.
- Map token IDs to words through `metadata["ix_to_word"]`.
- Pad and truncate sequences to a configurable maximum length.
- Default maximum length: 512 tokens.
- Save attention/padding masks.
- Save vocabulary metadata and token reconstruction helper.
- Use `metadata["lookup"]` to initialize an embedding layer where possible.
- Handle token IDs outside the embedding lookup range safely, for example by mapping them to `_UNK_`.

Images:

- Use `images` as the main image input.
- Restore viewable poster pixels by adding the Caffe/VGG channel mean `[103.939, 116.779, 123.68]`.
- Convert BGR to RGB for visualization.
- For pretrained `torchvision` models, convert restored pixels into the preprocessing expected by the selected weights.
- Save image preprocessing and deprocessing functions.
- Later XAI heatmaps should be shown over the same stored poster image resolution unless an explicit resize transform is used.

Labels:

- Use the 23-column `genres` array.
- Store genre label order.
- Recover label names from the paper table order:
  `Drama`, `Comedy`, `Romance`, `Thriller`, `Crime`, `Action`, `Adventure`, `Horror`, `Documentary`, `Mystery`, `Sci-Fi`, `Fantasy`, `Family`, `Biography`, `War`, `History`, `Music`, `Animation`, `Musical`, `Western`, `Sport`, `Short`, `Film-Noir`.

## Research Section: Candidate Models

The thesis document should include a focused technical model-research section.

Text model candidates:

- Word2Vec-initialized TextCNN.
- Word2Vec-initialized BiGRU/BiLSTM with attention pooling.
- Lightweight Transformer encoder over Word2Vec-initialized token embeddings.
- Lightweight attention pooling over token embeddings.
- Optional DistilBERT/BERT route after reconstructing plot strings, but this is secondary because the dataset provides token IDs rather than original raw plot strings.

Recommended text branch for the first final multimodal neural model:

- Embedding layer initialized from `metadata["lookup"]`.
- BiGRU with attention as the default non-CNN text encoder.
- TextCNN and lightweight Transformer encoders as ablations.
- Mask-aware pooling.
- Keep token-to-word mapping intact for later Integrated Gradients or token occlusion.

Image model candidates:

- Pretrained ResNet18 or ResNet50.
- Pretrained EfficientNet-B0.
- Pretrained ConvNeXt-Tiny.
- Smaller custom CNN only as a from-scratch comparison, not as the main model.

Recommended image branch for the first final multimodal neural model:

- Pretrained ResNet18 or ResNet50 from `torchvision`.
- Replace final classifier with an embedding projection layer.
- Use deterministic preprocessing.
- Keep spatial feature maps accessible for later Grad-CAM.

Fusion candidates:

- Concatenation fusion.
- Gated multimodal fusion inspired by the MM-IMDb GMU paper.
- Late fusion by averaging calibrated probabilities.

Recommended final neural fusion:

- Train a multimodal model with text branch, image branch, and gated or concatenation fusion.
- Prefer gated fusion if implementation time allows, because it directly connects to the source paper and supports modality-contribution analysis.

Classic ML candidates:

- Text: TF-IDF word/ngram features reconstructed from `sequences`.
- Image: reversible handcrafted descriptors from restored posters, such as color histograms, regional color histograms, edge/HOG-like descriptors, or downsampled grayscale patches.
- Fusion: concatenate text and image descriptors.
- Classifier: One-vs-Rest Logistic Regression or Linear SVM.

Recommended final classic ML model:

- Multimodal TF-IDF text features plus image descriptors.
- ClassifierChain Logistic Regression to model label co-occurrence, with One-vs-Rest Logistic Regression retained as a simpler baseline.
- Single-modality text-only and image-only versions may be used as baselines/ablations.

## Final Models To Train

At minimum, train and compare two final multimodal models:

1. Classic multimodal ML baseline:
   - Text: TF-IDF from reconstructed plot tokens.
   - Image: handcrafted descriptors from restored poster images.
   - Fusion: feature concatenation.
   - Classifier: ClassifierChain Logistic Regression, with One-vs-Rest as a baseline/ablation.

2. Neural multimodal model:
   - Text branch: embedding plus BiGRU-attention by default, with TextCNN/Transformer ablations.
   - Image branch: pretrained CNN from `torchvision`.
   - Fusion: concatenation or GMU-style gated fusion.
   - Output: 23 sigmoid logits.
   - Loss: `BCEWithLogitsLoss`.
   - Handle class imbalance with positive class weights or sampling, chosen based on validation behavior.

Allowed baselines and ablations:

- Text-only classic ML.
- Image-only classic ML.
- Text-only neural.
- Image-only neural.
- Precomputed-feature model using `features + vgg_features` as a fast reference only, clearly marked as less suitable for token/region XAI.

## Training Protocol

Use reproducible training:

- Fixed random seed.
- Saved split indices.
- Saved model config.
- Saved training metrics per epoch.
- Early stopping on validation macro F1 or validation sample F1.
- Checkpoint best model by validation metric.

Recommended neural training settings:

- Optimizer: AdamW.
- Loss: `BCEWithLogitsLoss`.
- Batch size depends on available GPU/CPU memory.
- Use mixed precision only if CUDA is available.
- Tune classification thresholds on validation set rather than hardcoding `0.5`.
- Prefer per-label threshold tuning for final multilabel models because genre prevalence is highly imbalanced.

## Evaluation

Report on validation during development and on test only once for final comparison.

Metrics:

- sample F1
- micro F1
- macro F1
- weighted F1
- precision and recall
- per-genre F1
- label distribution by split
- optional multilabel confusion diagnostics

Use the same metric families as the paper for comparability.

Save results:

- metrics JSON/CSV
- per-genre table
- prediction probabilities
- threshold values
- model checkpoints
- plots for training curves and label performance

## XAI Readiness

XAI methods are not implemented in the main pipeline now, but the design must support them later.

Text explanations:

- Integrated Gradients on embeddings.
- Token occlusion.
- SHAP-style token attribution if feasible.
- Attention visualization only if the model includes attention, and only as a supporting signal.

Image explanations:

- Grad-CAM on final convolutional layers.
- Integrated Gradients on image pixels.
- Occlusion sensitivity maps.

Multimodal explanations:

- Text-only vs image-only ablation.
- Modality dropout or modality masking.
- Gated fusion weights if a GMU-style model is used.
- Per-genre explanations for every selected predicted/top-k/explicit label, plus optional experimental label-set explanations.

The pipeline must save enough metadata to map explanations back to:

- reconstructed words/tokens for plot text,
- restored RGB poster images for image heatmaps,
- genre label names.

## Thesis-Style Technical Document

Create a focused technical document, not a fluffy overview.

Sections:

- Dataset origin and task definition.
- Paper summary relevant to this project.
- HDF5 schema and metadata schema.
- Label distribution and multilabel properties.
- Split strategy and reproducibility.
- Reversible preprocessing.
- Model research and chosen model rationale.
- Training protocol.
- Evaluation protocol.
- Limitations.
- XAI readiness and planned future explainability methods.

The document should include tables and plots generated from the actual dataset where useful.

## Code Structure

Recommended project structure:

```text
configs/
  default.yaml
docs/
  multimodal_genre_pipeline_plan.md
  dataset_preprocessing_report.md
notebooks/
  01_data_inspection.ipynb
  02_training_runner.ipynb
src/
  mmimdb/
    __init__.py
    data.py
    splits.py
    preprocessing.py
    image_utils.py
    text_utils.py
    models/
      classic.py
      neural.py
      fusion.py
    training.py
    evaluation.py
    reporting.py
scripts/
  create_splits.py
  inspect_dataset.py
  train_classic.py
  train_neural.py
  evaluate.py
  build_report.py
outputs/
  splits/
  models/
  metrics/
  figures/
```

## References For Research Section

- Arevalo et al., "Gated Multimodal Units for Information Fusion": https://arxiv.org/abs/1702.01992
- TorchVision pretrained model documentation: https://docs.pytorch.org/vision/master/models.html
- Hugging Face `AutoModelForSequenceClassification` documentation: https://huggingface.co/docs/transformers/main/en/model_doc/auto
- Sanh et al., "DistilBERT, a distilled version of BERT": https://arxiv.org/abs/1910.01108
