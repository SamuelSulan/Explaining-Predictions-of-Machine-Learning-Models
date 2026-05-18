from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import h5py
import joblib
import numpy as np
from scipy import sparse

from mmimdb.constants import GENRE_LABELS
from mmimdb.data import DatasetPaths, load_metadata
from mmimdb.evaluation import predicted_label_indices, threshold_to_serializable
from mmimdb.image_utils import image_descriptor, restore_poster_rgb
from mmimdb.models.classic import load_reconstructed_texts
from mmimdb.models.neural import MMIMDBTorchDataset
from mmimdb.text_utils import sequence_to_text
from mmimdb.utils import load_config, resolve_path
from mmimdb.xai import load_neural_model, patch_classic_classifier_compat


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config("configs/default.yaml")
PATHS = DatasetPaths.from_config(CONFIG)
METADATA = load_metadata(PATHS.metadata)
XAI_CFG = CONFIG.get("xai", {})


def safe_path(raw_path: str | Path) -> Path:
    path = resolve_path(unquote(str(raw_path)))
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path is outside the project: {path}")
    return path


def json_response(handler: BaseHTTPRequestHandler, payload: dict | list, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, body: str, status: int = 200, content_type: str = "text/html") -> None:
    data = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", f"{content_type}; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def read_json(path: str | Path) -> dict:
    with safe_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def relative_path(path: str | Path) -> str:
    return str(safe_path(path).relative_to(ROOT)).replace("\\", "/")


def file_url(path: str | Path) -> str:
    return f"/file?path={relative_path(path)}"


def existing_report_roots() -> list[dict]:
    candidates = [
        ROOT / "outputs" / "xai",
        ROOT / "outputs" / "xai_smoke_conda",
    ]
    seen = set()
    roots = []
    for candidate in candidates + sorted((ROOT / "outputs").glob("xai*")):
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        classic = candidate / "classic" / "classic_xai_summary.json"
        neural = candidate / "neural" / "neural_xai_summary.json"
        if classic.exists() or neural.exists():
            roots.append(
                {
                    "label": candidate.relative_to(ROOT).as_posix(),
                    "path": candidate.relative_to(ROOT).as_posix(),
                    "classic": classic.exists(),
                    "neural": neural.exists(),
                }
            )
    return roots


def summary_path(root: str, model_type: str) -> Path:
    name = "classic_xai_summary.json" if model_type == "classic" else "neural_xai_summary.json"
    return safe_path(root) / model_type / name


def list_samples(root: str, model_type: str) -> list[dict]:
    path = summary_path(root, model_type)
    if not path.exists():
        return []
    summary = read_json(path)
    samples = []
    for sample in summary.get("samples", []):
        index = sample.get("index")
        genre = sample.get("target_genre", "unknown")
        sample_dir = path.parent / f"{model_type}_idx_{index}_{genre}"
        explanation_path = sample_dir / "explanation.json"
        samples.append(
            {
                "index": index,
                "target_genre": genre,
                "target_genres": sample.get("target_genres", [genre]),
                "probability": sample.get("probability"),
                "true_genres": sample.get("true_genres", []),
                "predicted_genres": sample.get("predicted_genres", []),
                "methods": sample.get("methods", [sample.get("method")]),
                "path": relative_path(explanation_path) if explanation_path.exists() else None,
            }
        )
    return samples


def explanation_payload(path: str) -> dict:
    explanation_path = safe_path(path)
    explanation = read_json(explanation_path)
    sample_dir = explanation_path.parent
    image_names = [
        "poster.png",
        "model_input_poster.png",
        "thumbnail_descriptor_heatmap.png",
        "integrated_gradients_image_overlay.png",
        "gradcam_overlay.png",
        "image_occlusion_sensitivity_overlay.png",
        "modality_shapley.png",
        "top_text_features.png",
        "top_token_attributions.png",
        "token_occlusion_attributions.png",
    ]
    images = [
        {
            "name": image_name,
            "url": file_url(sample_dir / image_name),
        }
        for image_name in image_names
        if (sample_dir / image_name).exists()
    ]
    return {"explanation": explanation, "images": images}


def top_probabilities(probabilities: np.ndarray, n: int = 8) -> list[dict]:
    order = np.argsort(probabilities)[::-1][:n]
    return [{"genre": GENRE_LABELS[i], "probability": float(probabilities[i])} for i in order]


def predicted_genres(probabilities: np.ndarray, threshold: float | list[float]) -> list[str]:
    return [GENRE_LABELS[i] for i in predicted_label_indices(probabilities, threshold)]


@lru_cache(maxsize=2)
def load_classic_artifact(model_path: str) -> dict:
    artifact = joblib.load(safe_path(model_path))
    patch_classic_classifier_compat(artifact["classifier"])
    return artifact


@lru_cache(maxsize=1)
def load_neural_artifact(checkpoint_path: str):
    return load_neural_model(safe_path(checkpoint_path), PATHS.metadata)


def classic_predict(model_path: str, index: int) -> dict:
    artifact = load_classic_artifact(model_path)
    classifier = artifact["classifier"]
    vectorizer = artifact["vectorizer"]
    image_scaler = artifact["image_scaler"]
    cfg = artifact.get("config", {})
    hist_bins = int(cfg.get("image_hist_bins", 16))
    thumbnail_size = tuple(cfg.get("image_thumbnail_size", (32, 20)))
    text = load_reconstructed_texts(PATHS.hdf5, PATHS.metadata, np.asarray([index]))[0]
    x_text = vectorizer.transform([text])

    with h5py.File(PATHS.hdf5, "r") as f:
        desc = image_descriptor(
            f["images"][index],
            hist_bins=hist_bins,
            thumbnail_size=thumbnail_size,
        ).reshape(1, -1)
        labels = f["genres"][index].astype(np.int8)

    desc_scaled = image_scaler.transform(desc)
    x = sparse.hstack([x_text, sparse.csr_matrix(desc_scaled)], format="csr")
    probabilities = np.asarray(classifier.predict_proba(x))[0]
    threshold = artifact.get("threshold", 0.5)
    return {
        "model_type": "classic",
        "model_path": relative_path(model_path),
        "top_probabilities": top_probabilities(probabilities),
        "predicted_genres": predicted_genres(probabilities, threshold),
        "true_genres": [GENRE_LABELS[i] for i in np.flatnonzero(labels)],
        "threshold": threshold_to_serializable(threshold),
        "text_preview": text[:1200],
    }


def neural_predict(checkpoint_path: str, index: int) -> dict:
    import torch

    model, cfg, metadata, checkpoint, device = load_neural_artifact(checkpoint_path)
    dataset = MMIMDBTorchDataset(
        PATHS.hdf5,
        np.asarray([index], dtype=np.int64),
        vocab_size=int(metadata["vocab_size"]),
        max_length=cfg.max_length,
        input_size=cfg.input_size,
    )
    item = dataset[0]
    tokens = item["tokens"].unsqueeze(0).to(device)
    mask = item["mask"].unsqueeze(0).to(device)
    image = item["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(tokens, mask, image)).detach().cpu().numpy()[0]
    threshold = checkpoint.get("threshold", 0.5)

    with h5py.File(PATHS.hdf5, "r") as f:
        real_len = int(mask.sum().detach().cpu().item())
        text = sequence_to_text(f["sequences"][index][:real_len], metadata["ix_to_word"])
        labels = f["genres"][index].astype(np.int8)

    return {
        "model_type": "neural",
        "model_path": relative_path(checkpoint_path),
        "top_probabilities": top_probabilities(probabilities),
        "predicted_genres": predicted_genres(probabilities, threshold),
        "true_genres": [GENRE_LABELS[i] for i in np.flatnonzero(labels)],
        "threshold": threshold_to_serializable(threshold),
        "text_preview": text[:1200],
    }


def sample_poster_data_url(index: int) -> str:
    with h5py.File(PATHS.hdf5, "r") as f:
        rgb = restore_poster_rgb(f["images"][index])
    from io import BytesIO
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def default_model_paths() -> dict:
    classic_default = resolve_path(XAI_CFG.get("classic", {}).get("model_path", "outputs/models/best/classic_multimodal_best.joblib"))
    neural_default = resolve_path(XAI_CFG.get("neural", {}).get("checkpoint_path", "outputs/models/best/neural_multimodal_best.pt"))
    fallback_classic = ROOT / "outputs" / "models" / "classic_multimodal_limit10.joblib"
    fallback_neural = ROOT / "outputs" / "models" / "multimodal_bigru_attention_resnet18_gmu_limit10.pt"
    if not fallback_neural.exists():
        fallback_neural = ROOT / "outputs" / "models" / "multimodal_textcnn_resnet18_gmu_limit10.pt"
    return {
        "classic": relative_path(classic_default if classic_default.exists() else fallback_classic),
        "neural": relative_path(neural_default if neural_default.exists() else fallback_neural),
    }


def infer_payload(query: dict[str, list[str]]) -> dict:
    index = int(query.get("index", ["0"])[0])
    model_type = query.get("model_type", ["both"])[0]
    classic_model = query.get("classic_model", [default_model_paths()["classic"]])[0]
    neural_checkpoint = query.get("neural_checkpoint", [default_model_paths()["neural"]])[0]

    results = []
    if model_type in {"classic", "both"}:
        results.append(classic_predict(classic_model, index))
    if model_type in {"neural", "both"}:
        results.append(neural_predict(neural_checkpoint, index))

    return {
        "index": index,
        "poster": sample_poster_data_url(index),
        "results": results,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                text_response(self, HTML)
            elif parsed.path == "/api/state":
                json_response(
                    self,
                    {
                        "roots": existing_report_roots(),
                        "default_models": default_model_paths(),
                        "genres": GENRE_LABELS,
                    },
                )
            elif parsed.path == "/api/samples":
                root = query.get("root", ["outputs/xai"])[0]
                model_type = query.get("model_type", ["classic"])[0]
                json_response(self, list_samples(root, model_type))
            elif parsed.path == "/api/explanation":
                json_response(self, explanation_payload(query["path"][0]))
            elif parsed.path == "/api/infer":
                json_response(self, infer_payload(query))
            elif parsed.path == "/file":
                self.send_file(query["path"][0])
            else:
                text_response(self, "Not found", HTTPStatus.NOT_FOUND, "text/plain")
        except Exception as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[xai-dashboard] {self.address_string()} - {fmt % args}")

    def send_file(self, raw_path: str) -> None:
        path = safe_path(raw_path)
        if not path.exists() or not path.is_file():
            text_response(self, "File not found", HTTPStatus.NOT_FOUND, "text/plain")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MM-IMDb XAI Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #19212a;
      --muted: #627180;
      --line: #d9e0e7;
      --accent: #146c94;
      --accent-2: #1f8a70;
      --warn: #bf5b30;
      --soft: #eef5f7;
      --shadow: 0 8px 28px rgba(22, 34, 45, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }
    header {
      background: #18242e;
      color: #fff;
      padding: 18px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
    }
    h1 { font-size: 22px; margin: 0; font-weight: 650; }
    main { padding: 22px; max-width: 1500px; margin: 0 auto; }
    .tabs { display: flex; gap: 8px; margin-bottom: 18px; }
    button, select, input {
      font: inherit;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      min-height: 38px;
    }
    button {
      border-radius: 6px;
      padding: 8px 13px;
      cursor: pointer;
    }
    button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    button.tab.active { background: #dceef4; border-color: #a8cad7; }
    select, input {
      border-radius: 5px;
      padding: 7px 9px;
      width: 100%;
    }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; font-weight: 650; }
    .toolbar {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr 1.4fr auto;
      gap: 12px;
      align-items: end;
      margin-bottom: 16px;
    }
    .infer-toolbar {
      display: grid;
      grid-template-columns: 0.45fr 0.7fr 1.3fr 1.3fr auto;
      gap: 12px;
      align-items: end;
      margin-bottom: 16px;
    }
    .layout { display: grid; grid-template-columns: 330px 1fr; gap: 16px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .panel h2 { font-size: 16px; margin: 0; padding: 14px 16px; border-bottom: 1px solid var(--line); }
    .panel-body { padding: 14px 16px; }
    .sample-list { display: grid; gap: 8px; max-height: calc(100vh - 210px); overflow: auto; padding: 12px; }
    .sample-item {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px;
      background: #fff;
      cursor: pointer;
    }
    .sample-item.active { border-color: var(--accent); background: var(--soft); }
    .sample-title { display: flex; justify-content: space-between; gap: 8px; font-weight: 650; }
    .muted { color: var(--muted); }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 10px; margin-bottom: 14px; }
    .metric { border: 1px solid var(--line); border-radius: 7px; padding: 10px; background: #fbfcfd; }
    .metric .value { font-size: 20px; font-weight: 700; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
    .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
    figure { margin: 0; border: 1px solid var(--line); border-radius: 7px; overflow: hidden; background: #fff; }
    figure img { width: 100%; display: block; }
    figcaption { padding: 8px 10px; border-top: 1px solid var(--line); font-size: 12px; color: var(--muted); }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-size: 12px; }
    .bar-row { display: grid; grid-template-columns: minmax(80px, 190px) 1fr 80px; gap: 8px; align-items: center; margin: 7px 0; }
    .bar-track { height: 12px; background: #edf1f4; border-radius: 20px; overflow: hidden; position: relative; }
    .bar { height: 100%; background: var(--accent-2); }
    .bar.neg { background: var(--warn); }
    .pill { display: inline-flex; align-items: center; border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; margin: 2px; background: #f8fafb; font-size: 12px; }
    .section { margin: 16px 0; }
    .section h3 { font-size: 14px; margin: 0 0 9px; }
    .poster { max-width: 300px; border: 1px solid var(--line); border-radius: 7px; }
    pre { white-space: pre-wrap; background: #f4f6f8; border: 1px solid var(--line); padding: 10px; border-radius: 7px; max-height: 240px; overflow: auto; }
    .hidden { display: none; }
    .error { color: #9a3412; background: #fff4ed; border: 1px solid #fed7aa; padding: 10px; border-radius: 7px; }
    @media (max-width: 980px) {
      .toolbar, .infer-toolbar, .layout, .grid-2 { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>MM-IMDb XAI Dashboard</h1>
    <div class="muted">Saved model inference and explanation reports</div>
  </header>
  <main>
    <div class="tabs">
      <button id="reportsTab" class="tab active" type="button">XAI Reports</button>
      <button id="inferTab" class="tab" type="button">Model Inference</button>
    </div>

    <section id="reportsView">
      <div class="toolbar">
        <label>Report Root<select id="rootSelect"></select></label>
        <label>Model Type<select id="modelType"><option value="classic">Classic</option><option value="neural">Neural</option></select></label>
        <label>Sample<select id="sampleSelect"></select></label>
        <button id="refreshReports" class="primary" type="button">Load</button>
      </div>
      <div class="layout">
        <aside class="panel">
          <h2>Samples</h2>
          <div id="sampleList" class="sample-list"></div>
        </aside>
        <section class="panel">
          <h2>Explanation</h2>
          <div id="reportContent" class="panel-body"></div>
        </section>
      </div>
    </section>

    <section id="inferView" class="hidden">
      <div class="infer-toolbar">
        <label>Dataset Index<input id="inferIndex" type="number" min="0" value="17"></label>
        <label>Model<select id="inferModel"><option value="both">Both</option><option value="classic">Classic</option><option value="neural">Neural</option></select></label>
        <label>Classic Model<input id="classicModel" type="text"></label>
        <label>Neural Checkpoint<input id="neuralModel" type="text"></label>
        <button id="runInfer" class="primary" type="button">Run</button>
      </div>
      <section class="panel">
        <h2>Inference Result</h2>
        <div id="inferContent" class="panel-body"></div>
      </section>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    let state = { roots: [], samples: [] };

    function fmt(value, digits = 4) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return Number(value).toFixed(digits);
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
    }

    async function getJson(url) {
      const res = await fetch(url);
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || res.statusText);
      return data;
    }

    function setTab(active) {
      $("reportsView").classList.toggle("hidden", active !== "reports");
      $("inferView").classList.toggle("hidden", active !== "infer");
      $("reportsTab").classList.toggle("active", active === "reports");
      $("inferTab").classList.toggle("active", active === "infer");
    }

    function methodPills(methods) {
      return (methods || []).filter(Boolean).map(m => `<span class="pill">${escapeHtml(m)}</span>`).join("");
    }

    function pills(items) {
      return (items || []).map(g => `<span class="pill">${escapeHtml(g)}</span>`).join("");
    }

    function bars(items, labelKey, valueKey) {
      const values = items.map(item => Math.abs(Number(item[valueKey] || 0)));
      const max = Math.max(...values, 1e-12);
      return items.map(item => {
        const value = Number(item[valueKey] || 0);
        const width = Math.max(2, Math.abs(value) / max * 100);
        return `<div class="bar-row">
          <div title="${escapeHtml(item[labelKey])}">${escapeHtml(item[labelKey])}</div>
          <div class="bar-track"><div class="bar ${value < 0 ? "neg" : ""}" style="width:${width}%"></div></div>
          <div>${fmt(value, 5)}</div>
        </div>`;
      }).join("");
    }

    function topProbTable(rows) {
      const normalized = Array.isArray(rows)
        ? rows
        : Object.entries(rows || {}).map(([genre, probability]) => ({ genre, probability }));
      return `<table><thead><tr><th>Genre</th><th>Probability</th></tr></thead><tbody>
        ${normalized.map(row =>
          `<tr><td>${escapeHtml(row.genre)}</td><td>${fmt(row.probability)}</td></tr>`).join("")}
      </tbody></table>`;
    }

    function featureTable(rows, columns) {
      if (!rows || !rows.length) return `<div class="muted">No data in this report.</div>`;
      return `<table><thead><tr>${columns.map(c => `<th>${escapeHtml(c.label)}</th>`).join("")}</tr></thead><tbody>
        ${rows.map(row => `<tr>${columns.map(c => `<td>${escapeHtml(c.format ? c.format(row[c.key], row) : row[c.key])}</td>`).join("")}</tr>`).join("")}
      </tbody></table>`;
    }

    function shapleyBlock(shapley) {
      if (!shapley) return "";
      const logit = shapley.logit?.shapley || {};
      const prob = shapley.probability?.shapley || {};
      const utilText = Number(logit.text_utilization || 0) * 100;
      const utilImage = Number(logit.image_utilization || 0) * 100;
      return `<div class="section">
        <h3>Modality Utilization</h3>
        <div class="bar-row"><div>Text</div><div class="bar-track"><div class="bar" style="width:${utilText}%"></div></div><div>${fmt(utilText, 1)}%</div></div>
        <div class="bar-row"><div>Image</div><div class="bar-track"><div class="bar" style="width:${utilImage}%"></div></div><div>${fmt(utilImage, 1)}%</div></div>
        <table><tbody>
          <tr><th>Logit text</th><td>${fmt(logit.text, 5)}</td><th>Probability text</th><td>${fmt(prob.text, 5)}</td></tr>
          <tr><th>Logit image</th><td>${fmt(logit.image, 5)}</td><th>Probability image</th><td>${fmt(prob.image, 5)}</td></tr>
          <tr><th>Interaction</th><td>${fmt(logit.interaction, 5)}</td><th>Probability interaction</th><td>${fmt(prob.interaction, 5)}</td></tr>
        </tbody></table>
      </div>`;
    }

    function renderImages(images) {
      if (!images.length) return "";
      const labels = {
        "poster.png": "Original poster",
        "model_input_poster.png": "Neural model input",
        "thumbnail_descriptor_heatmap.png": "Classic thumbnail/pixel descriptor heatmap",
        "integrated_gradients_image_overlay.png": "Integrated Gradients pixel attribution",
        "gradcam_overlay.png": "Grad-CAM region attribution",
        "image_occlusion_sensitivity_overlay.png": "Patch occlusion sensitivity",
        "modality_shapley.png": "Modality Shapley",
        "top_text_features.png": "Classic text feature chart",
        "top_token_attributions.png": "Layer IG token chart",
        "token_occlusion_attributions.png": "Token occlusion chart"
      };
      return `<div class="section"><h3>Visual Evidence</h3><div class="image-grid">
        ${images.map(img => `<figure><img src="${img.url}" alt="${escapeHtml(labels[img.name] || img.name)}"><figcaption>${escapeHtml(labels[img.name] || img.name)}</figcaption></figure>`).join("")}
      </div></div>`;
    }

    function renderExplanation(payload) {
      const exp = payload.explanation;
      const content = $("reportContent");
      const tokenOcc = exp.token_occlusion?.top_tokens || [];
      const imageOcc = exp.image_occlusion?.top_patches || [];
      const topTokens = exp.top_tokens || [];
      const topTextFeatures = exp.top_text_features || [];
      const topImageFeatures = exp.top_image_features || [];
      const targetGenres = exp.target_genres?.length ? exp.target_genres : [exp.target_genre];

      content.innerHTML = `
        <div class="metrics">
          <div class="metric"><div class="muted">Index</div><div class="value">${escapeHtml(exp.index)}</div></div>
          <div class="metric"><div class="muted">Target</div><div>${pills(targetGenres)}</div></div>
          <div class="metric"><div class="muted">Probability</div><div class="value">${fmt(exp.probability)}</div></div>
          <div class="metric"><div class="muted">Predicted Genres</div><div>${pills(exp.predicted_genres || [])}</div></div>
          <div class="metric"><div class="muted">True Genres</div><div>${pills(exp.true_genres || [])}</div></div>
        </div>
        <div class="section"><h3>Methods</h3>${methodPills(exp.methods || [exp.method])}</div>
        <div class="grid-2">
          <div class="section"><h3>Top Probabilities</h3>${topProbTable(exp.top_probabilities)}</div>
          ${shapleyBlock(exp.modality_shapley)}
        </div>
        ${renderImages(payload.images)}
        <div class="grid-2">
          <div class="section"><h3>Highest Text Impact</h3>
            ${topTokens.length ? bars(topTokens, "token", "attribution") : bars(topTextFeatures, "feature", "contribution")}
          </div>
          <div class="section"><h3>Token Occlusion Impact</h3>${bars(tokenOcc, "token", "probability_delta")}</div>
        </div>
        <div class="grid-2">
          <div class="section"><h3>Classic Image/Descriptor Features</h3>${featureTable(topImageFeatures, [
            { key: "feature", label: "Feature" },
            { key: "value", label: "Value", format: v => fmt(v, 5) },
            { key: "contribution", label: "Contribution", format: v => fmt(v, 5) }
          ])}</div>
          <div class="section"><h3>Patch / Superpixel-Style Regions</h3>${featureTable(imageOcc, [
            { key: "row", label: "Row" },
            { key: "col", label: "Col" },
            { key: "probability_delta", label: "Probability Delta", format: v => fmt(v, 6) },
            { key: "logit_delta", label: "Logit Delta", format: v => fmt(v, 6) },
            { key: "x0", label: "x0" },
            { key: "y0", label: "y0" },
            { key: "x1", label: "x1" },
            { key: "y1", label: "y1" }
          ])}</div>
        </div>
      `;
    }

    async function loadState() {
      state = await getJson("/api/state");
      $("rootSelect").innerHTML = state.roots.map(root => `<option value="${root.path}">${root.label}</option>`).join("");
      $("classicModel").value = state.default_models.classic;
      $("neuralModel").value = state.default_models.neural;
      await loadSamples();
    }

    async function loadSamples() {
      const root = $("rootSelect").value;
      const modelType = $("modelType").value;
      const samples = await getJson(`/api/samples?root=${encodeURIComponent(root)}&model_type=${modelType}`);
      state.samples = samples;
      $("sampleSelect").innerHTML = samples.map(s => {
        const targetLabel = (s.target_genres?.length ? s.target_genres : [s.target_genre]).join("+");
        return `<option value="${s.path || ""}">idx ${s.index} · ${escapeHtml(targetLabel)} · p=${fmt(s.probability)}</option>`;
      }).join("");
      $("sampleList").innerHTML = samples.map((s, i) => `<div class="sample-item" data-i="${i}">
        <div class="sample-title"><span>idx ${escapeHtml(s.index)}</span><span>${fmt(s.probability)}</span></div>
        <div>${pills(s.target_genres?.length ? s.target_genres : [s.target_genre])}</div>
        <div class="muted">${(s.true_genres || []).join(", ")}</div>
      </div>`).join("");
      document.querySelectorAll(".sample-item").forEach(el => el.addEventListener("click", () => loadExplanation(Number(el.dataset.i))));
      if (samples.length) await loadExplanation(0);
      else $("reportContent").innerHTML = `<div class="error">No samples found for this report root/model type.</div>`;
    }

    async function loadExplanation(i = null) {
      const sample = i === null ? state.samples[$("sampleSelect").selectedIndex] : state.samples[i];
      if (!sample?.path) return;
      $("sampleSelect").value = sample.path;
      document.querySelectorAll(".sample-item").forEach((el, idx) => el.classList.toggle("active", idx === state.samples.indexOf(sample)));
      const payload = await getJson(`/api/explanation?path=${encodeURIComponent(sample.path)}`);
      renderExplanation(payload);
    }

    async function runInference() {
      const qs = new URLSearchParams({
        index: $("inferIndex").value,
        model_type: $("inferModel").value,
        classic_model: $("classicModel").value,
        neural_checkpoint: $("neuralModel").value
      });
      $("inferContent").innerHTML = `<div class="muted">Running inference...</div>`;
      try {
        const data = await getJson(`/api/infer?${qs}`);
        $("inferContent").innerHTML = `<div class="grid-2">
          <div><img class="poster" src="${data.poster}" alt="Poster"><div class="muted">Dataset index ${escapeHtml(data.index)}</div></div>
          <div>${data.results.map(result => `<div class="section">
            <h3>${escapeHtml(result.model_type)} model</h3>
            <div class="muted">${escapeHtml(result.model_path)}</div>
            <div>Predicted: ${pills(result.predicted_genres || [])}</div>
            <div>True: ${pills(result.true_genres || [])}</div>
            ${topProbTable(result.top_probabilities)}
            <h3>Plot Preview</h3><pre>${escapeHtml(result.text_preview)}</pre>
          </div>`).join("")}</div>
        </div>`;
      } catch (err) {
        $("inferContent").innerHTML = `<div class="error">${escapeHtml(err.message)}</div>`;
      }
    }

    $("reportsTab").addEventListener("click", () => setTab("reports"));
    $("inferTab").addEventListener("click", () => setTab("infer"));
    $("refreshReports").addEventListener("click", loadSamples);
    $("rootSelect").addEventListener("change", loadSamples);
    $("modelType").addEventListener("change", loadSamples);
    $("sampleSelect").addEventListener("change", () => loadExplanation(null));
    $("runInfer").addEventListener("click", runInference);
    loadState().catch(err => $("reportContent").innerHTML = `<div class="error">${escapeHtml(err.message)}</div>`);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local MM-IMDb XAI dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"XAI dashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
