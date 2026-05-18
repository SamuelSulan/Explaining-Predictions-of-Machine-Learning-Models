# MM-IMDb Dataset, Preprocessing, And Metrics Overview

## Purpose Of This Report

This report summarizes how the project uses the MM-IMDb dataset, how the data is preprocessed, and which metrics are calculated during dataset inspection, model training, evaluation, and explainability analysis. It is written as a readable project document, so the reader does not need to inspect the source code directly.

The project task is multilabel movie genre classification from two modalities:

- plot text
- poster image

Each movie can belong to more than one genre, so the problem is multilabel classification rather than ordinary single-label multiclass classification.

## Dataset Source

The project uses the MM-IMDb dataset from Kaggle:

- `johnarevalo/mmimdb`

The expected local dataset files are:

- `dataset/data/multimodal_imdb.hdf5`
- `dataset/data/metadata.npy`
- `dataset/Article about dataset and how was it used.pdf`

The HDF5 file and metadata file are required for the machine learning pipeline. The PDF article is used only when regenerating the technical dataset report.

## Dataset Contents

The dataset contains:

- `25,959` movie samples
- `23` genre labels
- plot text represented as token sequences and pretrained text features
- poster images represented as preprocessed image tensors
- pretrained visual features
- IMDb identifiers

The main HDF5 schema is:

| Key | Shape | Description |
|---|---:|---|
| `features` | `(25959, 300)` | Pretrained text/document-level feature vectors |
| `genres` | `(25959, 23)` | Binary multilabel genre targets |
| `images` | `(25959, 3, 256, 160)` | Poster tensors |
| `imdb_ids` | `(25959,)` | IMDb movie identifiers |
| `sequences` | `(25959,)` | Plot text as token ID sequences |
| `three_grams` | `(25959, 9946)` | Precomputed 3-gram features |
| `vgg_features` | `(25959, 4096)` | Precomputed visual CNN features |
| `word_grams` | `(25959, 13253)` | Precomputed word n-gram features |

## Metadata Contents

The metadata file contains vocabulary and embedding information:

| Key | Description |
|---|---|
| `ix_to_word` | Mapping from token IDs to words |
| `word_to_ix` | Mapping from words to token IDs |
| `lookup` | Pretrained 300-dimensional Word2Vec-style embedding matrix |
| `vocab_size` | Full vocabulary size |

Important metadata values:

- `vocab_size`: `69,980`
- `lookup` shape: `(41611, 300)`
- embedding dimension: `300`

The `lookup` matrix covers the vocabulary intersection available from the pretrained embeddings. Vocabulary items not covered by `lookup` are initialized randomly in the neural pipeline.

## Genre Labels

The project uses 23 genre labels:

| Index | Genre |
|---:|---|
| 0 | Drama |
| 1 | Comedy |
| 2 | Romance |
| 3 | Thriller |
| 4 | Crime |
| 5 | Action |
| 6 | Adventure |
| 7 | Horror |
| 8 | Documentary |
| 9 | Mystery |
| 10 | Sci-Fi |
| 11 | Fantasy |
| 12 | Family |
| 13 | Biography |
| 14 | War |
| 15 | History |
| 16 | Music |
| 17 | Animation |
| 18 | Musical |
| 19 | Western |
| 20 | Sport |
| 21 | Short |
| 22 | Film-Noir |

## Label Statistics

The dataset is imbalanced. Some genres, especially `Drama` and `Comedy`, are much more common than others.

Overall label statistics:

- number of samples: `25,959`
- number of labels: `23`
- zero-label rows: `0`
- minimum labels per movie: `1`
- mean labels per movie: `2.485`
- median labels per movie: `2`
- maximum labels per movie: `10`

Genre counts and prevalence:

| Genre | Count | Prevalence |
|---|---:|---:|
| Drama | 13967 | 53.80% |
| Comedy | 8592 | 33.10% |
| Romance | 5364 | 20.66% |
| Thriller | 5192 | 20.00% |
| Crime | 3838 | 14.78% |
| Action | 3550 | 13.68% |
| Adventure | 2710 | 10.44% |
| Horror | 2703 | 10.41% |
| Documentary | 2082 | 8.02% |
| Mystery | 2057 | 7.92% |
| Sci-Fi | 1991 | 7.67% |
| Fantasy | 1933 | 7.45% |
| Family | 1668 | 6.43% |
| Biography | 1343 | 5.17% |
| War | 1335 | 5.14% |
| History | 1143 | 4.40% |
| Music | 1045 | 4.03% |
| Animation | 997 | 3.84% |
| Musical | 841 | 3.24% |
| Western | 705 | 2.72% |
| Sport | 634 | 2.44% |
| Short | 471 | 1.81% |
| Film-Noir | 338 | 1.30% |

This imbalance is one reason the project uses macro F1 and per-label threshold tuning. Macro F1 gives rare genres more influence than metrics dominated by common genres.

## Dataset Split

The project creates its own split instead of using the original paper split as the main split.

Current split strategy:

- train: `70%`
- validation: `15%`
- test: `15%`
- random state: `42`
- strategy: iterative multilabel stratification when available

Generated split sizes:

| Split | Samples |
|---|---:|
| Train | 18171 |
| Validation | 3894 |
| Test | 3894 |

The split preserves multilabel genre distribution better than a simple random split. If the iterative stratification package is unavailable, the code falls back to a standard shuffled split.

## Text Preprocessing

Plot text is stored in the dataset as token ID sequences. The project keeps this representation reversible so explanations can be mapped back to readable tokens.

Text preprocessing steps:

1. Read token ID sequences from the HDF5 `sequences` dataset.
2. Map token IDs back to words using `metadata["ix_to_word"]`.
3. Replace invalid or unknown token IDs with `_UNK_`.
4. Truncate long sequences to the configured maximum length.
5. Pad short sequences to the configured maximum length.
6. Create a binary mask that marks real tokens as `1` and padding as `0`.
7. Initialize the neural embedding matrix from pretrained Word2Vec-style vectors.
8. Add a special all-zero padding row to the embedding matrix.

Default neural text length:

- `max_length`: `512`

Sequence length statistics:

| Statistic | Value |
|---|---:|
| Minimum | 1 |
| Mean | 124.413 |
| Median | 107 |
| 95th percentile | 301 |
| Maximum | 1887 |

The 512-token limit keeps nearly all ordinary plots intact while avoiding extreme memory usage from rare very long sequences.

## Image Preprocessing

Poster images are stored as tensors with shape:

- `(3, 256, 160)`

The stored representation is treated as Caffe/VGG-style BGR pixels with channel mean subtraction.

To restore a poster for visualization or explanation:

1. Add the Caffe BGR mean:
   - `(103.939, 116.779, 123.68)`
2. Convert BGR channel order to RGB.
3. Transpose the tensor into standard image format.
4. Clip pixel values to the range `0..255`.
5. Convert to `uint8`.

For neural image models:

1. Restore the poster to RGB.
2. Scale pixel values to `0..1`.
3. Resize to the configured input size.
4. Normalize using ImageNet mean and standard deviation.
5. Pass the image through a pretrained torchvision CNN, usually ResNet18.

Default neural image size:

- `224 x 224`

For classic machine learning models, the image is converted into handcrafted descriptors:

- global RGB color histograms
- 2x2 regional RGB color histograms
- small grayscale thumbnail descriptor

These image descriptors are standardized before being concatenated with text features.

## Classic Multimodal Pipeline

The classic model uses a scikit-learn style multimodal pipeline.

Text branch:

- reconstruct plot text from token IDs
- apply TF-IDF vectorization
- default: unigram features
- default maximum text features: `8000`
- default minimum document frequency: `3`

Image branch:

- restore poster images to RGB
- extract color histogram descriptors
- extract thumbnail descriptors
- standardize image descriptors

Fusion:

- concatenate text TF-IDF features and image descriptors

Classifier:

- default estimator: `SGDClassifier`
- default loss: `log_loss`
- default multilabel strategy: One-vs-Rest
- alternative supported strategy: Classifier Chain

The classic model is useful as a fast, interpretable baseline. Because it is linear, it can provide direct feature contribution explanations for words, n-grams, and poster descriptors.

## Neural Multimodal Pipeline

The neural model uses a lazy HDF5-backed PyTorch dataset. Each training sample contains:

- dataset index
- padded token IDs
- text padding mask
- normalized poster image
- 23-dimensional binary genre vector

Default neural architecture:

- text encoder: BiGRU with attention
- image encoder: pretrained ResNet18
- fusion: Gated Multimodal Unit style fusion
- output layer: 23 sigmoid outputs
- loss: binary cross entropy with logits

Supported text encoders:

- BiGRU with attention
- TextCNN
- lightweight Transformer encoder

Supported image encoders:

- ResNet18
- ResNet50
- EfficientNet-B0

Supported fusion methods:

- concatenation
- GMU-style gated fusion

The GMU-style fusion is especially useful for explainability because it can show how strongly the model uses text versus image information.

## Thresholding

Because the task is multilabel, the model outputs one probability per genre. These probabilities must be converted into binary predictions.

The project supports two threshold strategies:

| Strategy | Description |
|---|---|
| Global threshold | One threshold shared by all labels |
| Per-label threshold | One threshold tuned separately for each genre |

Default strategy:

- `per_label`

Default tuning metric:

- `macro_f1`

Thresholds are searched from:

- `0.05` to `0.95`
- step size: `0.05`

Per-label thresholding is useful because genre frequencies differ strongly. Rare genres may need different decision thresholds from common genres.

## Model Evaluation Metrics

The project calculates several multilabel evaluation metrics.

| Metric | Meaning |
|---|---|
| `sample_f1` | Calculates F1 for each movie, then averages across movies |
| `micro_f1` | Aggregates all label decisions globally before computing F1 |
| `macro_f1` | Computes F1 per genre, then averages all genres equally |
| `weighted_f1` | Computes F1 per genre, weighted by genre support |
| `micro_precision` | Global precision across all genre decisions |
| `micro_recall` | Global recall across all genre decisions |
| `hamming_loss` | Fraction of incorrect movie-genre decisions |
| per-label precision | Precision for each individual genre |
| per-label recall | Recall for each individual genre |
| per-label F1 | F1 score for each individual genre |
| per-label support | Number of true samples for each genre |

The most important model-selection metric in the project is:

- `macro_f1`

Macro F1 is important because it treats each genre equally. This prevents the model from looking strong only because it predicts common genres such as `Drama` and `Comedy`.

## Precision, Recall, And F1 Interpretation

Precision answers:

- When the model predicts a genre, how often is it correct?

Recall answers:

- Out of all movies that truly have a genre, how many did the model find?

F1 combines precision and recall:

- high F1 means the model balances correctness and coverage

In this project, F1 is calculated in several averaging modes because multilabel performance can look different depending on how results are aggregated.

## Hamming Loss

Hamming loss measures the fraction of wrong label decisions.

For this dataset, each movie has 23 possible genre decisions. Hamming loss checks how many of those binary decisions are incorrect.

Lower hamming loss is better.

However, hamming loss can sometimes look good even when rare labels are poorly predicted, because most movie-label pairs are negative. Therefore, it should be interpreted together with macro F1 and per-label scores.

## Per-Label Metrics

The project stores detailed metrics for every genre:

- precision
- recall
- F1
- support

Per-label metrics are important because the dataset is imbalanced. For example, a model may perform well on `Drama` but poorly on `Film-Noir`, `Short`, or `Sport`.

A good report should therefore include both:

- aggregate metrics such as `macro_f1`, `micro_f1`, and `weighted_f1`
- per-genre results

## Training And Model Selection Metrics

The training pipeline ranks candidate models using the configured metric:

- default: `macro_f1`

For model selection:

1. Train and validation indices can be combined for cross-validation.
2. Candidate models are evaluated across folds.
3. Candidates are ranked by validation metric.
4. The best candidate is trained again using the original train/validation split.
5. Thresholds are tuned on validation data.
6. Final evaluation is performed on the untouched test split.

The neural trainer also records:

- training loss
- validation loss
- validation metric per epoch
- best epoch
- validation metrics for the best checkpoint
- test metrics for the final selected checkpoint

The classic trainer records:

- selected classifier type
- selected estimator type
- number of text features
- number of image features
- tuned thresholds
- validation metrics
- test metrics

## Explainability Metrics

The project separates training from explainability. XAI runs are performed after trained model artifacts are saved.

Classic model explanations include:

- linear TF-IDF word or n-gram contributions
- poster descriptor contributions
- text-vs-image contribution totals
- modality Shapley utilization
- classifier-chain label dependency contributions when classifier chains are used

Neural model explanations include:

- Layer Integrated Gradients for text embeddings
- token occlusion
- Integrated Gradients for image pixels
- Grad-CAM for the image branch
- image patch occlusion sensitivity
- modality ablation
- modality Shapley utilization
- GMU gate summary
- optional label-set attributions

These explanation outputs are saved as JSON files and figures under:

- `outputs/xai/classic`
- `outputs/xai/neural`

## Runtime And Resource Metrics

XAI runs also measure runtime and resource usage.

Saved performance metrics include:

| Metric | Description |
|---|---|
| `wall_time_s` | Real elapsed time |
| `process_cpu_time_s` | CPU time consumed by the process |
| `process_cpu_util_percent` | CPU utilization estimate |
| `rss_start_mb` | Memory usage at start |
| `rss_end_mb` | Memory usage at end |
| `rss_delta_mb` | Memory usage difference |
| `rss_peak_mb` | Peak memory usage during measurement |
| `cuda_available` | Whether CUDA was available |
| CUDA allocated/reserved memory | GPU memory stats when CUDA is available |
| output file count | Number of generated output files |
| output size | Total size of generated outputs |

`process_cpu_util_percent` can be above 100% when libraries use multiple CPU threads.

## Current Output Files

Important generated output files include:

- `outputs/splits/split_metadata.json`
- `outputs/model_selection/training_summary.json`
- `outputs/model_selection/classic/cv_summary.json`
- `outputs/model_selection/neural/cv_summary.json`
- `outputs/xai/classic/classic_xai_summary.json`
- `outputs/xai/neural/neural_xai_summary.json`
- `outputs/figures/label_phi_correlation.png`
- `outputs/figures/label_lift_heatmap.png`
- `outputs/figures/label_conditional_probability.png`

Some currently saved training outputs are smoke-test runs with very small limits. These are useful for verifying that the pipeline works, but they should not be interpreted as final model performance.

For a deeper label relationship analysis, including scaled co-occurrence, lift, phi correlation, directional conditional probabilities, and strongest positive/negative label relationships, see:

- `docs/label_relationship_discovery_report.md`

For additional dataset discovery, including label cardinality, minority/majority label groups, exact genre combinations, co-occurrence network communities, frequent words by genre, and poster color/brightness statistics, see:

- `docs/additional_dataset_discovery_report.md`

## Summary

This project uses MM-IMDb as a multimodal multilabel classification dataset. The two main information sources are movie plot text and poster images. Text preprocessing keeps token IDs reversible so explanations can be mapped back to words. Image preprocessing reverses Caffe/VGG-style mean subtraction so posters can be visualized and used for both handcrafted descriptors and neural CNN inputs.

The most important evaluation metric is macro F1 because the dataset is strongly imbalanced across genres. The project also calculates sample F1, micro F1, weighted F1, micro precision, micro recall, hamming loss, and detailed per-label precision, recall, F1, and support.

The preprocessing and metric design are aligned with the goal of explainable multimodal genre classification: the pipeline preserves enough information to explain both textual and visual contributions to each genre prediction.
