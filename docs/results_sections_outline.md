# Validation and Test Results Section Outline

## Validation Results

### Validation setup
- Use the saved split metadata: 18,171 training samples, 3,894 validation samples, and 3,894 test samples.
- Model selection was performed with 3-fold cross-validation on the combined training and validation subset.
- The selection metric was macro F1.

### Classic model validation
- Candidate models:
  - `classic_ovr_logreg_unigram_10k`
  - `classic_ovr_sgd_unigram_10k`
- The best classic validation candidate was `classic_ovr_logreg_unigram_10k`.
- Cross-validation macro F1:
  - Logistic regression OvR: 0.411 mean macro F1.
  - SGD OvR: 0.241 mean macro F1.
- Final saved classic validation metrics:
  - sample F1: 0.553
  - micro F1: 0.558
  - macro F1: 0.438
  - weighted F1: 0.562
  - hamming loss: 0.112

### Neural model validation
- Candidate models included multimodal Transformer-GMU-ResNet18, Transformer-GMU-EfficientNet-B0, text-only Transformer, and image-only EfficientNet-B0 variants.
- The best neural cross-validation candidate was `neural_transformer_gmu_resnet18_h256_lr2e4_focal`.
- Cross-validation macro F1:
  - Transformer-GMU-ResNet18 with focal loss: 0.452 mean macro F1.
  - Transformer-GMU-ResNet18: 0.437 mean macro F1.
  - Transformer-GMU-ResNet18 with plateau scheduler: 0.437 mean macro F1.
  - Transformer text-only ablation: 0.421 mean macro F1.
  - Transformer-GMU-EfficientNet-B0: 0.402 mean macro F1.
  - EfficientNet-B0 image-only ablation: 0.358 mean macro F1.
- Final saved neural validation metrics:
  - sample F1: 0.559
  - micro F1: 0.569
  - macro F1: 0.483
  - weighted F1: 0.571
  - hamming loss: 0.109

## Test Results

### Test evaluation artifacts
- The saved best models are:
  - Classic model: `outputs/models/best/classic_multimodal_best.joblib`
  - Neural model: `outputs/models/best/neural_multimodal_best.pt`
- Final comparison artifacts are stored in `outputs/final_xai_analysis`.
- The classic final test result was produced by fresh inference on 3,894 test samples.
- The neural final test result was loaded from saved metrics, not fresh local inference, and reports 3,878 test samples.

### Classic model test results
- sample F1: 0.736
- micro F1: 0.743
- macro F1: 0.718
- weighted F1: 0.747
- micro precision: 0.639
- micro recall: 0.886
- hamming loss: 0.067

### Neural model test results
- sample F1: 0.557
- micro F1: 0.563
- macro F1: 0.468
- weighted F1: 0.564
- micro precision: 0.488
- micro recall: 0.665
- hamming loss: 0.112

### Final model ranking
- Ranking metric: macro F1.
- Best final model: classic multimodal model.
- Final macro F1 ranking:
  - Classic multimodal model: 0.718.
  - Neural multimodal model: 0.468.

## Comparison of Classic and Neural Models

### Overall performance comparison
- The classic model outperformed the neural model on all reported final test F1 metrics.
- The largest global difference was in macro F1, where the classic model reached 0.718 compared with 0.468 for the neural model.
- The classic model also achieved lower hamming loss: 0.067 compared with 0.112.

### Precision and recall comparison
- The classic model had higher micro precision: 0.639 compared with 0.488.
- The classic model had higher micro recall: 0.886 compared with 0.665.
- The classic result is therefore stronger both in label retrieval and in prediction precision.

### Model-selection comparison
- During validation, the neural model had a higher saved validation macro F1 than the classic model: 0.483 compared with 0.438.
- In the final comparison artifacts, the classic model produced the stronger test result.
- This section should distinguish validation model selection from final saved-model test evaluation.

### Explainability artifacts
- Classic XAI outputs include local text-feature contributions, image descriptor heatmaps, and modality Shapley summaries.
- Neural XAI outputs include token attributions, Integrated Gradients image overlays, Grad-CAM overlays, and modality Shapley summaries.
- The final XAI summary contains 41 local explanations: 40 classic explanations and 1 neural explanation.
- Mean local modality utilization from Shapley summaries:
  - Classic: 55.0% text and 45.0% image.
  - Neural: 30.1% text and 69.9% image.

## Per-genre Performance Analysis

### Highest-performing genres for the classic model
- Documentary: F1 0.849, support 315.
- War: F1 0.820, support 212.
- Horror: F1 0.815, support 429.
- Sci-Fi: F1 0.800, support 276.
- Drama: F1 0.799, support 2,110.

### Lowest-performing genres for the classic model
- Short: F1 0.473, support 68.
- Animation: F1 0.565, support 131.
- Film-Noir: F1 0.604, support 53.
- Mystery: F1 0.613, support 326.
- History: F1 0.626, support 176.

### Highest-performing genres for the neural model
- Drama: F1 0.732, support 2,095.
- Documentary: F1 0.651, support 312.
- Comedy: F1 0.647, support 1,289.
- Thriller: F1 0.573, support 779.
- Horror: F1 0.568, support 405.

### Lowest-performing genres for the neural model
- Film-Noir: F1 0.217, support 51.
- Short: F1 0.221, support 71.
- History: F1 0.298, support 171.
- Musical: F1 0.315, support 126.
- Biography: F1 0.323, support 201.

### Largest per-genre differences
- The classic model had the largest advantage for Musical, Film-Noir, Biography, Fantasy, and Music.
- Per-genre F1 differences:
  - Musical: classic 0.708, neural 0.315.
  - Film-Noir: classic 0.604, neural 0.217.
  - Biography: classic 0.701, neural 0.323.
  - Fantasy: classic 0.723, neural 0.369.
  - Music: classic 0.782, neural 0.431.
