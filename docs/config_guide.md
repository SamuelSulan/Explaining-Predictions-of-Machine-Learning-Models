# Configuration Guide

All main pipeline parameters are grouped in `configs/default.yaml`.

## Project

- `project.seed`: global reproducibility seed.
- `project.output_dir`: base output directory.

## Model Registry

- `model_registry.best_dir`: canonical folder for best model aliases.
- `model_registry.classic_best_name`: filename for the best classic multimodal model.
- `model_registry.neural_best_name`: filename for the best neural multimodal checkpoint.
- `model_registry.metric`: validation metric used to decide whether a full run replaces the current best model.
- `model_registry.update_best_only_for_full_runs`: prevents tiny `--limit` smoke tests from overwriting real best models.

## Splits

Controls the iterative multilabel stratified train/validation/test split.

## Training

Controls the one-command training/model-selection workflow:

- `training.output_dir`: where cross-validation results and final selected run artifacts are written.
- `training.metric`: metric used to rank candidates, usually `macro_f1`.
- `training.n_folds`: number of cross-validation folds.
- `training.combine_train_val_for_cv`: when `true`, the saved train and validation indices are combined for candidate selection while the saved test split remains untouched.
- `training.classic.enabled` and `training.neural.enabled`: allow either model family to be skipped.
- `training.<family>.final_only`: when `true`, skip CV/model selection for that family and train the first configured candidate directly. This is enabled for classic ML by default because the scikit-learn pipeline is CPU-bound and repeated folds are slow in Colab.
- `training.<family>.candidates`: explicit candidate list. Each candidate has a `name` and a flat `params` map that overrides fields from `ClassicConfig` or `NeuralConfig`.
- `training.<family>.candidates[].selection_eligible`: when `false`, the candidate is evaluated and reported but cannot be selected for the final model. This is used for text-only and image-only ablations.
- `training.<family>.grid`: optional Cartesian grid alternative. Each key is a config field and each value is a list of values to try.

The selected candidate is final-trained with the original train/validation split so thresholds and neural early stopping are still tuned without touching the test split. Full runs update separate best-model registry files for classic ML and neural models, comparing the new validation metric against any existing saved best model of the same family. Limited smoke runs do not update the registry.

## Text

Controls token sequence length and embedding behavior for the neural text branch. The active text encoder is selected with `neural.text_encoder`:

- `bigru_attention`: Word2Vec-initialized bidirectional GRU plus attention pooling. This is the default non-CNN text branch.
- `transformer`: lightweight in-repo Transformer encoder over the same token embeddings.
- `textcnn`: original parallel-convolution text branch retained for ablation.

## Image

Controls neural image size and the reversible Caffe/VGG poster restoration mean.

## Classic

Controls the classic multimodal baseline:

- TF-IDF vocabulary size and n-gram range.
- Poster descriptor size.
- `classic.estimator`: `sgd` for the faster default linear classifier or `logistic` for Logistic Regression.
- SGDClassifier `loss`, `penalty`, `alpha`, `max_iter`, `tol`, and class weighting. Keep `sgd_loss: log_loss` or `modified_huber` when threshold tuning needs probabilities.
- `classic.image_hist_bins` and `classic.image_thumbnail_size`: lower values reduce poster descriptor extraction time while keeping the model multimodal.
- Logistic Regression iteration count.
- Logistic Regression `C`, `penalty`, `solver`, optional `l1_ratio`, and class weighting.
- Threshold tuning metric.
- `classic.threshold_strategy`: `global` or `per_label`; per-label thresholds tune one decision threshold per genre.
- `classic.classifier`: `classifier_chain` models label dependencies; `ovr` uses independent One-vs-Rest labels.
- `classic.chain_order`: optional classifier-chain order; `null` keeps the default label order.

## Neural

Controls the neural multimodal model:

- text encoder, image encoder, fusion type, hidden sizes,
- BiGRU-attention settings: `text_rnn_hidden_dim`, `text_rnn_layers`, `text_rnn_dropout`, and `text_attention_dim`,
- Transformer settings: `text_transformer_layers`, `text_transformer_heads`, `text_transformer_ff_dim`, and `text_transformer_dropout`,
- frozen/unfrozen pretrained image branch,
- `modality`: `multimodal`, `text_only`, or `image_only` for ablation runs,
- batch size, epochs, optimizer settings,
- optional `scheduler: plateau` to reduce learning rate when validation macro F1 stops improving,
- optional `loss: focal` with `focal_gamma` and `pos_weight_clip` for rare-label experiments,
- early stopping and threshold metric.
- `neural.threshold_strategy`: `global` or `per_label`.
- `threshold_min`, `threshold_max`, and `threshold_steps`: threshold-search grid; the default uses a finer 0.01-0.99 grid.
- `neural.enable_label_correlation`: experimental residual label-correlation head over the 23 logits.

The default neural candidate set is intentionally compact: the previous best Transformer+GMU+ResNet18 setup, a plateau-scheduler variant, an unfrozen EfficientNet-B0 variant, text-only/image-only ablations, and a focal-loss variant. The ablations are useful for analysis but are not final-selection eligible.

## XAI

Controls XAI execution:

- `xai.output_dir`: explanation output directory.
- `xai.split`: split to explain.
- `xai.limit`: number of samples.
- `xai.target_genre`: optional fixed single target genre.
- `xai.target_genres`: optional explicit list of target genres.
- `xai.target_policy`: target selection when explicit genres are absent: `top`, `top_k`, `predicted`, or `all`.
- `xai.target_top_k`: number of genres used by `target_policy: top_k`.
- `xai.max_targets_per_sample`: optional cap for expensive multi-target XAI runs; set `null` to explain every selected target.
- `xai.ensure_at_least_one_target`: falls back to the top probability when no label crosses threshold.
- `xai.enable_experimental_set_explanation`: writes an additional `label_set` report that explains the selected genre group together.
- `xai.top_k`: number of top text/image features or tokens to report.
- `xai.n_steps`: Integrated Gradients steps; increase for final figures.
- `xai.measure_performance`: enables timing/resource metrics.
- `xai.performance_sample_interval_seconds`: memory sampling interval.
- `xai.classic.model_path`: default classic model used by `scripts/run_xai.py`.
- `xai.classic.enable_modality_shapley`: enables exact two-modality Shapley utilization from blank/text-only/image-only/both coalitions.
- `xai.neural.checkpoint_path`: default neural checkpoint used by `scripts/run_xai.py`.
- `xai.neural.enable_*`: toggles neural explainers such as Layer IG, image IG, Grad-CAM, ablation, modality Shapley, token occlusion, and image occlusion.
- `xai.neural.image_occlusion_grid`: poster patch grid size for occlusion sensitivity.
- `xai.neural.occlusion_batch_size`: batch size for token/image occlusion forward passes.

## XAI Metrics Currently Saved

For each explained sample and for each method block:

- wall-clock time,
- process CPU time,
- process CPU utilization estimate,
- RSS memory before/after/delta/peak,
- CUDA availability and CUDA memory stats if CUDA is available,
- output file count and output size summary.

`process_cpu_util_percent` is computed as process CPU time divided by wall-clock time. It can exceed 100% when NumPy, PyTorch, BLAS, or image libraries use multiple CPU threads.
