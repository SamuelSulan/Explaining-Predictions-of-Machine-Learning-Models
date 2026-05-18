"""Technical report generation."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mmimdb.constants import CAFFE_BGR_MEAN, GENRE_LABELS
from mmimdb.data import dataset_label_stats, h5_summary, load_labels, load_metadata
from mmimdb.text_utils import describe_sequence_lengths
from mmimdb.utils import ensure_dir, resolve_path


def extract_pdf_text(pdf_path: str | Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(resolve_path(pdf_path)))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return clean_pdf_text(text)
    except Exception as exc:
        return f"[PDF extraction failed: {type(exc).__name__}: {exc}]"


def clean_pdf_text(text: str) -> str:
    replacements = {
        "ï¬": "fi",
        "ï¬‚": "fl",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "Â": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _find_snippet(text: str, needle: str, chars: int = 900) -> str:
    idx = text.lower().find(needle.lower())
    if idx < 0:
        return ""
    snippet = text[max(0, idx - 150) : idx + chars]
    return " ".join(snippet.split())


def save_label_figures(y: np.ndarray, figures_dir: str | Path) -> dict[str, Path]:
    out = ensure_dir(figures_dir)
    counts = y.sum(axis=0)
    count_df = pd.DataFrame({"genre": GENRE_LABELS, "count": counts}).sort_values("count")

    plt.figure(figsize=(8, 7))
    sns.barplot(data=count_df, y="genre", x="count", color="#4C78A8")
    plt.title("Genre label counts")
    plt.tight_layout()
    label_path = out / "genre_label_counts.png"
    plt.savefig(label_path, dpi=160)
    plt.close()

    cooc = y.T @ y
    denom = np.maximum(np.diag(cooc), 1).reshape(-1, 1)
    cooc_norm = cooc / denom
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cooc_norm,
        xticklabels=GENRE_LABELS,
        yticklabels=GENRE_LABELS,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    plt.title("Genre co-occurrence normalized by row genre count")
    plt.tight_layout()
    cooc_path = out / "genre_cooccurrence.png"
    plt.savefig(cooc_path, dpi=160)
    plt.close()

    return {"label_counts": label_path, "cooccurrence": cooc_path}


def build_dataset_report(
    hdf5_path: str | Path,
    metadata_path: str | Path,
    article_pdf: str | Path,
    output_path: str | Path,
    figures_dir: str | Path,
    split_metadata_path: str | Path | None = None,
) -> Path:
    hdf5_path = resolve_path(hdf5_path)
    metadata_path = resolve_path(metadata_path)
    article_pdf = resolve_path(article_pdf)
    output_path = resolve_path(output_path)
    figures_dir = resolve_path(figures_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path)
    y = load_labels(hdf5_path)
    stats = dataset_label_stats(y)
    schema = h5_summary(hdf5_path)
    figures = save_label_figures(y, figures_dir)

    with h5py.File(hdf5_path, "r") as f:
        seq_stats = describe_sequence_lengths(f["sequences"])
        image_sample = f["images"][:128].astype(np.float32)
        restored_min = float((image_sample + np.asarray(CAFFE_BGR_MEAN).reshape(1, 3, 1, 1)).min())
        restored_max = float((image_sample + np.asarray(CAFFE_BGR_MEAN).reshape(1, 3, 1, 1)).max())

    article_text = extract_pdf_text(article_pdf)
    dataset_snippet = _find_snippet(article_text, "The MM-IMDb dataset has been split")
    text_snippet = _find_snippet(article_text, "The pretrained Google Word2vec")
    visual_snippet = _find_snippet(article_text, "VISUAL REPRESENTATION")
    table2_snippet = _find_snippet(article_text, "Table 2: Summary")

    schema_df = pd.DataFrame(schema)
    counts_df = pd.DataFrame(
        {
            "genre": GENRE_LABELS,
            "count": y.sum(axis=0).astype(int),
            "prevalence": y.mean(axis=0),
        }
    ).sort_values("count", ascending=False)

    split_section = "Split metadata has not been generated yet. Run `python scripts/create_splits.py`."
    if split_metadata_path is not None and resolve_path(split_metadata_path).exists():
        import json

        with resolve_path(split_metadata_path).open("r", encoding="utf-8") as f:
            split_meta = json.load(f)
        split_section = (
            f"Generated split sizes: train `{split_meta['n_train']}`, "
            f"validation `{split_meta['n_val']}`, test `{split_meta['n_test']}`. "
            f"Random state: `{split_meta['random_state']}`."
        )

    lines = [
        "# MM-IMDb Dataset And Preprocessing Technical Report",
        "",
        "## Task Definition",
        "",
        "The project task is multilabel movie genre classification from two modalities: plot text and poster image. Final comparison models must be multimodal. Single-modality models are allowed as baselines or ablations only.",
        "",
        "## Paper-Derived Notes",
        "",
        "The supplied paper introduces MM-IMDb for genre prediction from plot and poster and proposes Gated Multimodal Units for multimodal fusion.",
        "",
        f"- Dataset/split note from paper: {dataset_snippet or 'not extracted'}",
        f"- Text representation note from paper: {text_snippet or 'not extracted'}",
        f"- Visual representation note from paper: {visual_snippet or 'not extracted'}",
        f"- Results note from paper: {table2_snippet or 'not extracted'}",
        "",
        "## HDF5 Schema",
        "",
        schema_df.to_markdown(index=False),
        "",
        "## Metadata Schema",
        "",
        f"- Metadata keys: `{', '.join(sorted(metadata.keys()))}`.",
        f"- `vocab_size`: `{metadata['vocab_size']}`.",
        f"- `word_to_ix` size: `{len(metadata['word_to_ix'])}`.",
        f"- `ix_to_word` size: `{len(metadata['ix_to_word'])}`.",
        f"- `lookup` shape: `{metadata['lookup'].shape}`.",
        "- `lookup` provides pretrained 300-dimensional Word2Vec-style vectors for the intersected vocabulary.",
        "",
        "## Label Statistics",
        "",
        f"- Samples: `{stats['num_samples']}`.",
        f"- Labels: `{stats['num_labels']}`.",
        f"- Zero-label rows: `{stats['zero_label_rows']}`.",
        f"- Labels per movie: min `{stats['labels_per_movie_min']}`, mean `{stats['labels_per_movie_mean']:.3f}`, median `{stats['labels_per_movie_median']:.3f}`, max `{stats['labels_per_movie_max']}`.",
        "",
        counts_df.to_markdown(index=False),
        "",
        f"![Genre label counts]({figures['label_counts'].as_posix()})",
        "",
        f"![Genre co-occurrence]({figures['cooccurrence'].as_posix()})",
        "",
        "## Split Strategy",
        "",
        split_section,
        "",
        "The intended project split is a new iterative multilabel stratified split: 70% train, 15% validation, 15% test. The original paper split is documented for comparison but is not used as the main split.",
        "",
        "## Reversible Text Preprocessing",
        "",
        f"- `sequences` contains token IDs with sequence length stats: min `{seq_stats['min']}`, mean `{seq_stats['mean']:.3f}`, median `{seq_stats['median']}`, p95 `{seq_stats['p95']}`, max `{seq_stats['max']}`.",
        "- Token IDs are mapped back to words through `metadata['ix_to_word']`.",
        "- Neural models pad/truncate sequences to a configurable maximum length, default 512.",
        "- Padding masks are retained so future XAI methods can ignore padding tokens.",
        "- The embedding layer is initialized from `metadata['lookup']`; vocabulary rows not covered by lookup are initialized randomly.",
        "",
        "## Reversible Image Preprocessing",
        "",
        "- `images` contains poster tensors shaped `(3, 256, 160)`.",
        "- Stored values match Caffe/VGG-style BGR pixels with channel mean subtraction.",
        f"- The restoration mean is `{CAFFE_BGR_MEAN}`.",
        f"- On a 128-image sample, restored BGR pixel range before clipping is approximately `{restored_min:.3f}` to `{restored_max:.3f}`.",
        "- For visualization and later XAI overlays, add the mean, convert BGR to RGB, and clip to 0..255.",
        "- For pretrained CNNs, restored RGB pixels are normalized with ImageNet mean/std after deterministic resizing.",
        "",
        "## Model Choices",
        "",
        "- Classic multimodal final model: reconstructed plot TF-IDF plus reversible poster descriptors, fused by concatenation, classified with ClassifierChain Logistic Regression by default and One-vs-Rest as a baseline option.",
        "- Neural multimodal final model: Word2Vec-initialized text encoder plus pretrained `torchvision` image encoder, fused by concatenation or GMU-style gated fusion, trained with 23 sigmoid outputs.",
        "- Recommended neural image branch starts with ResNet18/ResNet50 because these are stable pretrained CNNs and straightforward to explain later with Grad-CAM.",
        "- Recommended text branch starts with BiGRU-attention, with TextCNN and lightweight Transformer variants retained as ablations because token attributions can be mapped back to words from `ix_to_word`.",
        "",
        "## XAI Readiness",
        "",
        "- Text explanations can use Integrated Gradients, token occlusion, or SHAP-style token attribution.",
        "- Image explanations can use Grad-CAM, Integrated Gradients, or occlusion sensitivity.",
        "- Multimodal explanations can compare text-only, image-only, and fused predictions, and can inspect GMU gate values if gated fusion is used.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
