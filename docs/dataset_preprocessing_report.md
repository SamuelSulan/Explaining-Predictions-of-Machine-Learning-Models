# MM-IMDb Dataset And Preprocessing Technical Report

## Task Definition

The project task is multilabel movie genre classification from two modalities: plot text and poster image. Final comparison models must be multimodal. Single-modality models are allowed as baselines or ablations only.

## Paper-Derived Notes

The supplied paper introduces MM-IMDb for genre prediction from plot and poster and proposes Gated Multimodal Units for multimodal fusion.

- Dataset/split note from paper: metadata information encourages other interesting tasks such as rating prediction and content-based retrieval, among others. 4.2 E XPERIMENTAL SETUP The MM-IMDb dataset has been split in three subsets. Train, development and test subsets contain 15552, 2608 and 7799 respectively. The distribution of samples is listed in Table 1. The sample was stratified so that training, dev and test sets comprises 60%, 10%, 30% samples of each genre respectively. In the multilabel classification the performance evaluation can be more complex than traditional multi-class classification and the differences can be significant among several measures (Madjarov et al., 2012). Herein, four averages of the f-score (f1) are reported: samples computes the f-score per sample and then averages the results,micro computes the f-score using all predictions at once,macro computes the f-score per genre and then averages the results. weighted is the same as macro with a weighted average based on the number of positive samples per genre. Concretely, we calculate them
- Text representation note from paper: espectively;tpj,fp jandfn j the number of true positives, false positives and false negatives for the j-th label respectively. TEXTUAL REPRESENTATION The pretrained Google Word2vec 4 embedding space was used. After intersecting the Google word2vec available words with the MM-IMDb plots, the final vocabulary contains 41,612 words. Other than lowercase, no text preprocessing was applied. Since it is our intention to measure how the network’s depth affects the performance of the model, we also evaluate the architecture with a single fully connected layer. In order to compare the performance of this textual representation, we evaluate it using two publicly available datasets:7genre dataset that comprises 1,400 web pages with 7 disjoint genres and ki-04 dataset that comprises 1,239 samples classified under 8 genres. We com- pare the model with the state of the art results (Kanaris & Stamatatos, 2009) which used character n-grams with structured information from the HTML tags to predict the genre of web pages. VISUAL REPRESENTATION Since th
- Visual representation note from paper: And thus, they require different representation strategies according to the nature of data. This work explored several strategies to address text and visual representation. For text information we evalu- ated word2vec models, n-grams models and RNN models. The details are discussed in Subsection 3.2. On the other hand, two different convolutional neural networks were evaluated for processing visual data and are presented in Subsection 3.3. 3.1 G ATED MULTIMODAL UNIT FOR MULTIMODAL FUSION Multimodal learning is closely related to data fusion. Data fusion looks for optimal ways of combin- ing different information sources into an integrated representation that provides more information than the individual sources (Bhatt & Kankanhalli, 2011). This fusion can be performed at different levels, that can be categorized into two broad categories: feature fusion and decision fusion. Feature fusion, also called early fusion, looks for a subset of features from different modalities, or combi- nations of them, that better represent the informatio
- Results note from paper: ICLR 2017 that the baseline uses additional information from the HTML structure from the web page, while this representation uses only the text data. Table 2: Summary of classification task on the MM-IMDb dataset Modality Representation F-Score weighted samples micro macro Multimodal GMU 0.617 0.630 0.630 0.541 Linear sum 0.600 0.607 0.607 0.530 Concatenate 0.597 0.605 0.606 0.521 A VGprobs 0.604 0.616 0.615 0.491 MoE MaxoutMLP 0.592 0.593 0.601 0.516 MoE MaxoutMLP (tied) 0.579 0.579 0.587 0.489 MoE Logistic 0.541 0.557 0.565 0.456 MoE Logistic (tied) 0.483 0.507 0.518 0.358 Text MaxoutMLP w2v 0.588 0.592 0.595 0.488 RNN transfer 0.570 0.580 0.580 0.480 MaxoutMLP w2v 1 hidden 0.540 0.540 0.550 0.440 Logistic w2v 0.530 0.540 0.550 0.420 MaxoutMLP 3grams 0.510 0.510 0.520 0.420 Logistic 3grams 0.510 0.520 0.530 0.400 RNN end2end 0.490 0.490 0.490 0.370 Visual VGG Transfer 0.410 0.429 0.437 0.284 CNN end2end 0.370 0.350 0.340 0.210 Table 2 shows the results in the proposed dataset. For the textual modality, the best performance is obtain

## HDF5 Schema

| key          | shape                | dtype   |
|:-------------|:---------------------|:--------|
| features     | (25959, 300)         | float32 |
| genres       | (25959, 23)          | int32   |
| images       | (25959, 3, 256, 160) | int32   |
| imdb_ids     | (25959,)             | |S7     |
| sequences    | (25959,)             | object  |
| three_grams  | (25959, 9946)        | float32 |
| vgg_features | (25959, 4096)        | float32 |
| word_grams   | (25959, 13253)       | float32 |

## Metadata Schema

- Metadata keys: `ix_to_word, lookup, vocab_size, word_to_ix`.
- `vocab_size`: `69980`.
- `word_to_ix` size: `69980`.
- `ix_to_word` size: `69980`.
- `lookup` shape: `(41611, 300)`.
- `lookup` provides pretrained 300-dimensional Word2Vec-style vectors for the intersected vocabulary.

## Label Statistics

- Samples: `25959`.
- Labels: `23`.
- Zero-label rows: `0`.
- Labels per movie: min `1`, mean `2.485`, median `2.000`, max `10`.

| genre       |   count |   prevalence |
|:------------|--------:|-------------:|
| Drama       |   13967 |    0.538041  |
| Comedy      |    8592 |    0.330983  |
| Romance     |    5364 |    0.206634  |
| Thriller    |    5192 |    0.200008  |
| Crime       |    3838 |    0.147849  |
| Action      |    3550 |    0.136754  |
| Adventure   |    2710 |    0.104395  |
| Horror      |    2703 |    0.104126  |
| Documentary |    2082 |    0.0802034 |
| Mystery     |    2057 |    0.0792403 |
| Sci-Fi      |    1991 |    0.0766979 |
| Fantasy     |    1933 |    0.0744636 |
| Family      |    1668 |    0.0642552 |
| Biography   |    1343 |    0.0517354 |
| War         |    1335 |    0.0514273 |
| History     |    1143 |    0.044031  |
| Music       |    1045 |    0.0402558 |
| Animation   |     997 |    0.0384067 |
| Musical     |     841 |    0.0323972 |
| Western     |     705 |    0.0271582 |
| Sport       |     634 |    0.0244231 |
| Short       |     471 |    0.018144  |
| Film-Noir   |     338 |    0.0130205 |

![Genre label counts](C:/Users/samue/Documents/diplomka/outputs/figures/genre_label_counts.png)

![Genre co-occurrence](C:/Users/samue/Documents/diplomka/outputs/figures/genre_cooccurrence.png)

## Split Strategy

Generated split sizes: train `18171`, validation `3894`, test `3894`. Random state: `42`.

The intended project split is a new iterative multilabel stratified split: 70% train, 15% validation, 15% test. The original paper split is documented for comparison but is not used as the main split.

## Reversible Text Preprocessing

- `sequences` contains token IDs with sequence length stats: min `1`, mean `124.413`, median `107.0`, p95 `301.0`, max `1887`.
- Token IDs are mapped back to words through `metadata['ix_to_word']`.
- Neural models pad/truncate sequences to a configurable maximum length, default 512.
- Padding masks are retained so future XAI methods can ignore padding tokens.
- The embedding layer is initialized from `metadata['lookup']`; vocabulary rows not covered by lookup are initialized randomly.

## Reversible Image Preprocessing

- `images` contains poster tensors shaped `(3, 256, 160)`.
- Stored values match Caffe/VGG-style BGR pixels with channel mean subtraction.
- The restoration mean is `(103.939, 116.779, 123.68)`.
- On a 128-image sample, restored BGR pixel range before clipping is approximately `0.680` to `254.939`.
- For visualization and later XAI overlays, add the mean, convert BGR to RGB, and clip to 0..255.
- For pretrained CNNs, restored RGB pixels are normalized with ImageNet mean/std after deterministic resizing.

## Model Choices

- Classic multimodal final model: reconstructed plot TF-IDF plus reversible poster descriptors, fused by concatenation, classified with ClassifierChain Logistic Regression by default and One-vs-Rest as a baseline option.
- Neural multimodal final model: Word2Vec-initialized text encoder plus pretrained `torchvision` image encoder, fused by concatenation or GMU-style gated fusion, trained with 23 sigmoid outputs.
- Recommended neural image branch starts with ResNet18/ResNet50 because these are stable pretrained CNNs and straightforward to explain later with Grad-CAM.
- Recommended text branch starts with BiGRU-attention, with TextCNN and lightweight Transformer variants retained as ablations because token attributions can be mapped back to words from `ix_to_word`.

## XAI Readiness

- Text explanations can use Integrated Gradients, token occlusion, or SHAP-style token attribution.
- Image explanations can use Grad-CAM, Integrated Gradients, or occlusion sensitivity.
- Multimodal explanations can compare text-only, image-only, and fused predictions, and can inspect GMU gate values if gated fusion is used.
