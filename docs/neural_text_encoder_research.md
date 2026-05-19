# Neural Text Encoder Research

The original neural baseline used a Word2Vec-initialized TextCNN. The code now supports three in-repo text encoders over the existing MM-IMDb token IDs: TextCNN, BiGRU-attention, and a lightweight Transformer. The base `neural` config still names BiGRU-attention as the standalone default, while the active model-selection candidates in `configs/default.yaml` currently use the lightweight Transformer because recent full runs selected Transformer-GMU variants.

## Researched Options

### BiGRU With Attention

Hierarchical Attention Networks for document classification use bidirectional recurrent encoders plus attention to build document representations. This is a good fit for movie plots because the model can read token order, preserve long-range context better than local convolutions, and expose attention-pooled token representations.

Implementation in this project:

- `neural.text_encoder: bigru_attention`
- Word2Vec-initialized embedding layer from `metadata["lookup"]`
- bidirectional GRU over padded token sequences
- mask-aware attention pooling
- projection to the shared multimodal hidden size

This remains a supported replacement for TextCNN and the standalone base-config default.

### Lightweight Transformer Encoder

Transformer-based multimodal classifiers such as MMBT and two-stream vision-language models such as ViLBERT motivate testing self-attention for image-text classification. A full pretrained BERT/VilBERT route would require reconstructing raw text, adding Hugging Face dependencies, and changing tokenization, so this project first adds a lightweight in-repo Transformer encoder over the existing Word2Vec token IDs.

Implementation in this project:

- `neural.text_encoder: transformer`
- Word2Vec-initialized token embeddings
- learned positional embeddings
- PyTorch `TransformerEncoder`
- mask-aware mean pooling

This is the text encoder used by the current configured model-selection candidates.

### BERT / DistilBERT / Vision-Language Transformers

These are strong options for future work. They are not the new default because the current dataset representation already provides token IDs and a Word2Vec lookup, while BERT-style models would need raw text reconstruction, subword tokenization, and additional dependencies. A later extension could add a Hugging Face text encoder or an MMBT-style joint image-text model.

## Recommended Search

The current configured neural model-selection candidates compare:

- lightweight Transformer + GMU + ResNet18 at learning rate `2e-4`
- the same Transformer-GMU-ResNet18 setup with `ReduceLROnPlateau`
- lightweight Transformer + GMU + unfrozen EfficientNet-B0 at learning rate `1e-4`
- Transformer text-only ablation, marked `selection_eligible: false`
- unfrozen EfficientNet-B0 image-only ablation, marked `selection_eligible: false`
- Transformer-GMU-ResNet18 with focal loss and clipped positive weights

BiGRU-attention, TextCNN, ResNet50, concatenation fusion, and additional hidden-size variants remain supported by the code, but they are not in the active default candidate list. The current search is focused on Transformer text encoding, GMU fusion, ResNet18 versus EfficientNet-B0 image branches, scheduler/loss variants, and text-only/image-only ablations.

## References

- Arevalo et al., 2017, Gated Multimodal Units for Information Fusion: https://arxiv.org/abs/1702.01992
- Yang et al., 2016, Hierarchical Attention Networks for Document Classification: https://aclanthology.org/N16-1174/
- Kiela et al., 2019, Supervised Multimodal Bitransformers for Classifying Images and Text: https://arxiv.org/abs/1909.02950
- Lu et al., 2019, ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks: https://arxiv.org/abs/1908.02265
