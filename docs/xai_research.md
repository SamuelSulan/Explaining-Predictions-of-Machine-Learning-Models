# XAI Method Research For The MM-IMDb Pipeline

## Goal

The project needs explanations for final multimodal models trained on plot tokens and poster images. Explanations should map back to human-readable plot tokens, poster regions, and modality-level behavior.

## Model Families In This Project

- Classic multimodal model: TF-IDF text features plus reversible poster descriptors, trained with ClassifierChain Logistic Regression by default and One-vs-Rest as a baseline option.
- Neural multimodal model: token embedding/TextCNN branch plus pretrained ResNet image branch, fused with GMU-style gated fusion.

## Recommended Methods

### Classic Multimodal Logistic Regression

Use direct linear feature contributions as the first explanation method.

- Text: contribution = TF-IDF value times logistic coefficient. This maps directly to words/ngrams.
- Image: contribution = scaled descriptor value times logistic coefficient. This maps to color histogram bins, coarse poster regions, and thumbnail pixels.
- Label dependencies: when `ClassifierChain` is enabled, report the previous-label prediction features and their coefficients for the current target genre.
- Multimodal: sum contributions separately for text and image descriptor blocks to estimate modality-level logit contribution.

SHAP is also relevant for linear models, but direct coefficient contributions are simpler and deterministic for feature-level reporting. The implemented two-modality Shapley diagnostic uses exact blank/text-only/image-only/both coalitions to report text-vs-image utilization without adding a heavy SHAP dependency.

### Neural Multimodal Model

Use gradient-based methods from Captum and a manual Grad-CAM implementation.

- Text: Layer Integrated Gradients on the embedding layer. This avoids taking gradients with respect to integer token IDs and returns token-level attribution scores that can be mapped through `ix_to_word`.
- Image: Integrated Gradients on normalized image pixels. This provides pixel/region attributions for the image input path.
- Image: Grad-CAM on the ResNet image branch. This produces coarse region-level heatmaps from the final convolutional block.
- Multimodal: modality ablation by comparing full prediction, no-text prediction, no-image prediction, and blank-input prediction.
- Multimodal: two-modality Shapley values over blank/text-only/image-only/both coalitions to quantify text-vs-image utilization and interaction.
- Perturbation: token occlusion and poster patch occlusion sensitivity as model-behavior checks against gradient explanations.
- Fusion: GMU gate summary to estimate whether the fused hidden representation leaned more toward text or image for a sample.
- Label dependencies: an experimental residual label-correlation head can be enabled in the neural model; gradient explanations then include its effect because it is inside the prediction graph.

### Multi-Target And Set-Level Explanations

The models produce 23 probabilities per sample, so XAI should not be limited to one genre. Supported target-selection policies are:

- `predicted`: explain every label whose probability crosses the saved threshold, optionally capped by `max_targets_per_sample`.
- `top_k`: explain the top-k most probable labels.
- `explicit`: explain only the genres passed through `target_genres` or `--target-genres`.
- `all`: explain all 23 labels when a comprehensive audit is needed.

Each selected genre gets its own normal per-label explanation. The experimental set-level report additionally explains the selected genre group as a single target: classic models sum the selected linear outputs, while neural models attribute the sum of the selected logits. Keep this enabled for thesis experiments and disable it for faster routine debugging.

## Practical Defaults

- Use validation/test samples for explanations, not training samples, unless debugging.
- Use predicted or top-k genres as the default target labels, with a cap for expensive neural explanations.
- Allow explicit `--target-genre` for thesis figures.
- Save JSON explanations plus PNG visualizations.
- Use a small `n_steps` value for smoke tests and a larger value later for final figures.

## References

- Integrated Gradients: https://arxiv.org/abs/1703.01365
- Grad-CAM: https://arxiv.org/abs/1610.02391
- Captum PyTorch tutorial: https://docs.pytorch.org/tutorials/beginner/introyt/captumyt.html
- Captum FAQ on embeddings and multiple inputs: https://captum.ai/docs/faq
- SHAP: https://arxiv.org/abs/1705.07874
