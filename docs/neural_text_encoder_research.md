# Neural Text Encoder Research

The original neural baseline used a Word2Vec-initialized TextCNN. The current default is changed to a Word2Vec-initialized BiGRU with attention because it better matches document-classification research while still fitting the existing MM-IMDb token representation and XAI workflow.

## Researched Options

### BiGRU With Attention

Hierarchical Attention Networks for document classification use bidirectional recurrent encoders plus attention to build document representations. This is a good fit for movie plots because the model can read token order, preserve long-range context better than local convolutions, and expose attention-pooled token representations.

Implementation in this project:

- `neural.text_encoder: bigru_attention`
- Word2Vec-initialized embedding layer from `metadata["lookup"]`
- bidirectional GRU over padded token sequences
- mask-aware attention pooling
- projection to the shared multimodal hidden size

This is the recommended first replacement for TextCNN.

### Lightweight Transformer Encoder

Transformer-based multimodal classifiers such as MMBT and two-stream vision-language models such as ViLBERT motivate testing self-attention for image-text classification. A full pretrained BERT/VilBERT route would require reconstructing raw text, adding Hugging Face dependencies, and changing tokenization, so this project first adds a lightweight in-repo Transformer encoder over the existing Word2Vec token IDs.

Implementation in this project:

- `neural.text_encoder: transformer`
- Word2Vec-initialized token embeddings
- learned positional embeddings
- PyTorch `TransformerEncoder`
- mask-aware mean pooling

This is useful as an ablation, but it may train more slowly than BiGRU-attention.

### BERT / DistilBERT / Vision-Language Transformers

These are strong options for future work. They are not the new default because the current dataset representation already provides token IDs and a Word2Vec lookup, while BERT-style models would need raw text reconstruction, subword tokenization, and additional dependencies. A later extension could add a Hugging Face text encoder or an MMBT-style joint image-text model.

## Recommended Search

The default neural model-selection candidates now compare:

- BiGRU-attention + GMU + ResNet18
- BiGRU-attention + concatenation + ResNet18
- lightweight Transformer + GMU + ResNet18

ResNet50 and the original TextCNN channel/hidden-size variants remain supported by the code, but they are intentionally not active default candidates after being trained separately. The current default search is a small hyperparameter sweep over recurrent attention vs self-attention, gated vs concatenation fusion, learning rate, recurrent dropout, and hidden size.

## References

- Arevalo et al., 2017, Gated Multimodal Units for Information Fusion: https://arxiv.org/abs/1702.01992
- Yang et al., 2016, Hierarchical Attention Networks for Document Classification: https://aclanthology.org/N16-1174/
- Kiela et al., 2019, Supervised Multimodal Bitransformers for Classifying Images and Text: https://arxiv.org/abs/1909.02950
- Lu et al., 2019, ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks: https://arxiv.org/abs/1908.02265
