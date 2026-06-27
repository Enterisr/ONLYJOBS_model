#!/usr/bin/env python3
"""
Contrastive fine-tuning → browser-ready ONNX export.

Model  : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
         (384-dim, 50 languages incl. Hebrew, ~120 MB float32 / ~30 MB int8)
Pipeline:
  1. Parse labelled CSV (0=noise, 1=job post).
  2. Freeze backbone; train a single Linear(384 → 256) projection with
     Supervised Contrastive Loss.
  3. Every EVAL_EVERY epochs: fit a quick LR on current embeddings and record
     accuracy, F1, precision, recall on the training set (optimistic but shows trend).
  4. Extract 256-dim embeddings; fit final StandardScaler + LogisticRegression.
  5. Export base encoder via optimum → model_output/onnx/model.onnx.
  6. INT8 dynamic quantisation → model_quantized.onnx (~30 MB).
  7. Numeric parity check (PyTorch ≈ ONNX).
  8. Save projection + LR as classifier.json for lightweight JS inference.
  9. Generate plots → model_output/plots/.

Usage:
  pip install -r requirements.txt
  py train.py
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.manifold import TSNE
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

plt.rcParams.update(
    {"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False}
)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384
PROJ_DIM = 128  # bottleneck: forces more compressed representation than raw 384-dim
MAX_LEN = 512  # model's hard limit
TEMPERATURE = 0.08
EPOCHS = 500
LR = 2e-3
EVAL_EVERY = 25  # compute train metrics every N epochs (fast on 50 samples)
OUTPUT_DIR = Path("model_output")

# ── Data ──────────────────────────────────────────────────────────────────────


def load_data(csv_path: str):
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="warn")
    texts = df.iloc[:, 0].astype(str).tolist()
    labels = df.iloc[:, 1].astype(int).tolist()
    return texts, labels


class PostDataset(Dataset):
    def __init__(self, embeddings, labels):
        # embeddings: (N, EMBED_DIM) float tensor, already extracted from backbone
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Model ─────────────────────────────────────────────────────────────────────


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


@torch.no_grad()
def embed_texts(backbone, tokenizer, texts):
    """Encode texts into (N, EMBED_DIM) embeddings, truncating at MAX_LEN tokens."""
    backbone.eval()
    all_embs = []
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )
        out = backbone(enc["input_ids"], enc["attention_mask"])
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"]).squeeze(0)
        all_embs.append(emb)
    backbone.train()
    return torch.stack(all_embs)  # (N, EMBED_DIM)


class Embedder(nn.Module):
    """Trainable Linear projection + L2 normalisation on precomputed backbone embeddings."""

    def __init__(self, embed_dim=EMBED_DIM, proj_dim=PROJ_DIM):
        super().__init__()
        self.projector = nn.Linear(embed_dim, proj_dim, bias=True)
        nn.init.xavier_uniform_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)

    def forward(self, x):
        return F.normalize(self.projector(x), p=2, dim=-1)


# ── Loss ──────────────────────────────────────────────────────────────────────


class SupConLoss(nn.Module):
    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        self.T = temperature

    def forward(self, features, labels):
        N = features.shape[0]
        dev = features.device

        sim = torch.matmul(features, features.T) / self.T

        lab = labels.unsqueeze(0)
        pos = (lab == lab.T).float()
        pos.fill_diagonal_(0.0)

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim) * (1.0 - torch.eye(N, device=dev))
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        n_pos = pos.sum(dim=1).clamp(min=1.0)
        return (-(pos * log_prob).sum(dim=1) / n_pos).mean()


# ── Metrics helpers ───────────────────────────────────────────────────────────


@torch.no_grad()
def get_embeddings(model, loader):
    model.eval()
    embs, labs = [], []
    for batch in loader:
        embs.append(model(batch["embedding"]).numpy())
        labs.extend(batch["label"].tolist())
    model.train()
    return np.concatenate(embs, 0), np.array(labs)


def quick_lr_metrics(emb, labels):
    """Fit a throw-away LR on current embeddings; return train-set metrics."""
    pipe = Pipeline(
        [
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=300, class_weight="balanced")),
        ]
    )
    pipe.fit(emb, labels)
    preds = pipe.predict(emb)
    return {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }


# ── Training ──────────────────────────────────────────────────────────────────


def train_projection(model, loader, label_arr):
    """
    Returns:
      loss_history    : list[float]  — loss at every epoch
      metric_history  : list[dict]   — {epoch, acc, f1, precision, recall}
                        recorded at epoch 1, every EVAL_EVERY, and final epoch
    """
    criterion = SupConLoss()
    optimizer = torch.optim.Adam(model.projector.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_history = []
    metric_history = []

    def _maybe_eval(epoch):
        emb, _ = get_embeddings(model, loader)
        m = quick_lr_metrics(emb, label_arr)
        metric_history.append({"epoch": epoch, **m})
        return m

    model.train()
    for epoch in range(1, EPOCHS + 1):
        ep_loss = 0.0
        for batch in loader:
            emb = model(batch["embedding"])
            loss = criterion(emb, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()
        loss_history.append(ep_loss / len(loader))

        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            m = _maybe_eval(epoch)
            print(
                f"  epoch {epoch:3d}/{EPOCHS}  "
                f"loss={loss_history[-1]:.4f}  "
                f"acc={m['acc']:.3f}  f1={m['f1']:.3f}  "
                f"prec={m['precision']:.3f}  rec={m['recall']:.3f}"
            )

    return loss_history, metric_history


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_training_curves(loss_history, metric_history, plots_dir):
    epochs_all = list(range(1, len(loss_history) + 1))
    eval_epochs = [m["epoch"] for m in metric_history]
    acc_vals = [m["acc"] for m in metric_history]
    f1_vals = [m["f1"] for m in metric_history]
    prec_vals = [m["precision"] for m in metric_history]
    rec_vals = [m["recall"] for m in metric_history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(
        "Training dynamics (metrics on training set — optimistic)", fontsize=12
    )

    # Left: loss
    ax = axes[0]
    ax.plot(epochs_all, loss_history, color="#2563eb", linewidth=1.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SupCon loss")
    ax.set_title("Contrastive loss")
    ax.grid(axis="y", alpha=0.3)

    # Right: accuracy / F1 / precision / recall
    ax = axes[1]
    palette = {
        "Accuracy": "#16a34a",
        "F1": "#dc2626",
        "Precision": "#9333ea",
        "Recall": "#ea580c",
    }
    for name, vals in zip(palette, [acc_vals, f1_vals, prec_vals, rec_vals]):
        ax.plot(
            eval_epochs,
            vals,
            marker="o",
            markersize=4,
            label=name,
            color=palette[name],
            linewidth=1.4,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Classification metrics")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    out = plots_dir / "training_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  training_curves.png saved")


def plot_tsne(raw_emb, proj_emb, labels, plots_dir):
    """Side-by-side t-SNE: backbone-only vs SupCon-projected embeddings."""
    n = len(labels)
    perp = max(3, min(15, n // 4))

    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    coords_raw = tsne.fit_transform(raw_emb)
    tsne2 = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    coords_proj = tsne2.fit_transform(proj_emb)

    colors = ["#3b82f6" if l == 0 else "#ef4444" for l in labels]
    labels_str = ["non-job" if l == 0 else "job" for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("t-SNE of embeddings  (blue = non-job, red = job)", fontsize=12)

    for ax, coords, title in zip(
        axes,
        [coords_raw, coords_proj],
        ["Backbone only (frozen, no projection)", f"After SupCon projection ({PROJ_DIM}-dim)"],
    ):
        for label_val, color, name in [
            (0, "#3b82f6", "non-job"),
            (1, "#ef4444", "job"),
        ]:
            mask = np.array(labels) == label_val
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                label=name,
                alpha=0.75,
                s=55,
                edgecolors="white",
                linewidths=0.5,
            )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=9)

    fig.tight_layout()
    out = plots_dir / "tsne.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  tsne.png saved")


def plot_confusion_matrix(lr_pipe, proj_emb, labels, plots_dir):
    preds = lr_pipe.predict(proj_emb)
    cm = confusion_matrix(labels, preds)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        "Final LR head — evaluation on training set  (optimistic)", fontsize=11
    )

    # Confusion matrix
    disp = ConfusionMatrixDisplay(cm, display_labels=["non-job", "job"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion matrix")

    # Per-class bar chart
    ax = axes[1]
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)

    metric_names = ["Accuracy", "F1", "Precision", "Recall"]
    metric_vals = [acc, f1, prec, rec]
    bar_colors = ["#16a34a", "#dc2626", "#9333ea", "#ea580c"]

    bars = ax.bar(metric_names, metric_vals, color=bar_colors, alpha=0.85)
    for bar, val in zip(bars, metric_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, 1.15)
    ax.set_title("Final metrics (train set)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = plots_dir / "confusion_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  confusion_matrix.png saved")


def plot_cv_comparison(cv_base, cv_proj, plots_dir):
    """Bar chart comparing baseline vs SupCon 5-fold CV F1."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Backbone\n(no projection)", "SupCon\n(256-dim projection)"]
    means = [cv_base.mean(), cv_proj.mean()]
    stds = [cv_base.std(), cv_proj.std()]
    colors = ["#94a3b8", "#2563eb"]

    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        color=colors,
        alpha=0.85,
        capsize=6,
        error_kw={"linewidth": 1.5},
    )
    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + s + 0.012,
            f"{m:.3f} +/- {s:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 (5-fold CV, stratified)")
    ax.set_title(
        "Baseline vs SupCon — 5-fold CV F1\n(CV on SupCon space is optimistic)"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    out = plots_dir / "cv_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  cv_comparison.png saved")


# ── ONNX Export ───────────────────────────────────────────────────────────────


def save_model_locally(backbone, projector, tokenizer, output_dir):
    """Save fine-tuned backbone + projector to disk before ONNX export."""
    local_dir = output_dir / "pytorch_model"
    local_dir.mkdir(parents=True, exist_ok=True)
    backbone.save_pretrained(str(local_dir))
    tokenizer.save_pretrained(str(local_dir))
    torch.save(projector.state_dict(), local_dir / "projector.pt")
    print(f"  Fine-tuned model saved locally -> {local_dir}/")


def export_base_encoder(model_name, output_dir):
    from optimum.exporters.onnx import main_export

    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Exporting {model_name} via optimum ...")
    main_export(
        model_name_or_path=model_name,
        output=onnx_dir,
        task="feature-extraction",
        for_ort=True,
    )
    p = onnx_dir / "model.onnx"
    print(f"  model.onnx -> {p.stat().st_size // 1_000_000} MB")
    return p


def quantize_encoder(model_path):
    out = model_path.parent / "model_quantized.onnx"
    quantize_dynamic(str(model_path), str(out), weight_type=QuantType.QInt8)
    print(f"  model_quantized.onnx -> {out.stat().st_size // 1_000_000} MB")
    return out


def parity_check(backbone, tokenizer, onnx_path, text):
    enc = tokenizer(
        text, return_tensors="pt", max_length=64, truncation=True, padding="max_length"
    )
    with torch.no_grad():
        pt_out = backbone(**enc).last_hidden_state.numpy()

    sess = ort.InferenceSession(str(onnx_path))
    ort_inputs = {}
    for inp in sess.get_inputs():
        arr = enc.get(inp.name)
        ort_inputs[inp.name] = (
            arr.numpy().astype(np.int64)
            if arr is not None
            else np.zeros_like(enc["input_ids"].numpy(), dtype=np.int64)
        )
    ort_out = sess.run(None, ort_inputs)[0]
    np.testing.assert_allclose(pt_out, ort_out, rtol=1e-3, atol=1e-4)
    print("  ✓ parity check passed (PyTorch ≈ ONNX)")


def export_classifier(lr_pipe, out_path, projector=None):
    """
    JS inference pipeline:
      1. backbone ONNX -> last_hidden_state
      2. mean_pool -> (EMBED_DIM,)
      [if projector present]
      3. projected = L2_norm(proj_weight @ emb + proj_bias)  -> (PROJ_DIM,)
      4. z = (projected - scaler_mean) / scaler_std
      5. logit = dot(lr_coef, z) + lr_intercept
      6. p_job = sigmoid(logit)
    """
    sc = lr_pipe.steps[0][1]
    lr = lr_pipe.steps[1][1]
    payload = {
        "model_name":   MODEL_NAME,
        "embed_dim":    EMBED_DIM,
        "scaler_mean":  sc.mean_.tolist(),
        "scaler_std":   sc.scale_.tolist(),
        "lr_coef":      lr.coef_[0].tolist(),
        "lr_intercept": float(lr.intercept_[0]),
        "threshold":    0.5,
    }
    if projector is not None:
        with torch.no_grad():
            w = projector.projector.weight.detach().cpu().numpy()  # (PROJ_DIM, EMBED_DIM)
            b = projector.projector.bias.detach().cpu().numpy()    # (PROJ_DIM,)
        payload["proj_weight"] = w.tolist()
        payload["proj_bias"]   = b.tolist()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  classifier.json -> {out_path.stat().st_size // 1024} KB  ({out_path})")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("Loading data ...")
    texts, labels = load_data("dataset.csv")
    label_arr = np.array(labels)
    print(
        f"  {len(texts)} samples   job=1: {label_arr.sum()}   non-job=0: {(label_arr == 0).sum()}"
    )

    # ── 2. Load backbone ──────────────────────────────────────────────────────
    print(f"Loading tokeniser + backbone ({MODEL_NAME}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME)

    # ── 3. Precompute backbone embeddings with sliding-window chunking ─────────

    raw_emb_tensor = embed_texts(backbone, tokenizer, texts)
    raw_emb = raw_emb_tensor.numpy()

    dataset = PostDataset(raw_emb_tensor, labels)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    base_pipe = Pipeline(
        [
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
        ]
    )
    cv_base = cross_val_score(
        base_pipe,
        raw_emb,
        label_arr,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1",
    )
    print(f"  baseline 5-fold CV F1: {cv_base.mean():.3f} +/- {cv_base.std():.3f}")
    base_pipe.fit(raw_emb, label_arr)  # fit on full set for export

    # ── 4. Contrastive fine-tuning ────────────────────────────────────────────
    print(f"\nTraining projection head ({EPOCHS} epochs) ...")
    model = Embedder()
    loss_history, metric_history = train_projection(model, loader, label_arr)

    # ── 5. Final LR head on projected embeddings ──────────────────────────────
    print("\nFitting final LR head ...")
    proj_emb, _ = get_embeddings(model, loader)

    lr_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
        ]
    )
    lr_pipe.fit(proj_emb, label_arr)

    cv_proj = cross_val_score(
        lr_pipe,
        proj_emb,
        label_arr,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1",
    )
    print(f"  projected 5-fold CV F1: {cv_proj.mean():.3f} +/- {cv_proj.std():.3f}")
    print(f"  (CV on SupCon space is optimistic - use delta over baseline as signal)")
    print(f"  SupCon delta: {cv_proj.mean() - cv_base.mean():+.3f}")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_training_curves(loss_history, metric_history, plots_dir)
    plot_tsne(raw_emb, proj_emb, label_arr, plots_dir)
    plot_confusion_matrix(lr_pipe, proj_emb, label_arr, plots_dir)
    plot_cv_comparison(cv_base, cv_proj, plots_dir)
    print(f"  All plots -> {plots_dir}/")

    # ── 7. Save PyTorch model locally ─────────────────────────────────────────
    print("\nSaving fine-tuned model locally ...")
    save_model_locally(backbone, model.projector, tokenizer, OUTPUT_DIR)

    # ── 8. Save classifier.json ───────────────────────────────────────────────
    print("Saving classifier.json ...")
    clf_path = OUTPUT_DIR / "classifier.json"
    # Export SupCon pipeline: projector weights + LR trained on projected embeddings.
    # The baseline is only used as a reference; the SupCon model is what goes to JS.
    export_classifier(lr_pipe, clf_path, projector=model)

    # Copy classifier.json to extension dir if it exists
    ext_dir = Path(__file__).parent.parent / "linkedin-job-filter"
    if ext_dir.exists():
        import shutil
        shutil.copy(clf_path, ext_dir / "classifier.json")
        print(f"  copied -> {ext_dir / 'classifier.json'}")

    # ── 9. ONNX export (optional — extension uses Xenova CDN model) ───────────
    print("\nExporting encoder ONNX ...")
    onnx_path = export_base_encoder(MODEL_NAME, OUTPUT_DIR)
    quant_path = quantize_encoder(onnx_path)

    print("Running parity check ...")
    try:
        parity_check(
            backbone,
            tokenizer,
            quant_path,
            "We are hiring a software engineer to join our team.",
        )
    except AssertionError as e:
        print(f"  ⚠ parity check failed (quantization drift): {e}")

    print("Saving tokeniser ...")
    (OUTPUT_DIR / "tokenizer").mkdir(exist_ok=True)
    tokenizer.save_pretrained(str(OUTPUT_DIR / "tokenizer"))

    # ── 10. Summary ───────────────────────────────────────────────────────────
    print("\n-- Done --")
    print(f"  {OUTPUT_DIR}/classifier.json  <- SupCon projection + LR weights for JS")
    print(f"  {OUTPUT_DIR}/pytorch_model/   <- backbone + projector.pt")
    print(f"  {OUTPUT_DIR}/plots/           <- training curves, t-SNE, confusion matrix")
    print()
    print("Extension setup (one-time):")
    print("  cd ..\\linkedin-job-filter")
    print("  npm install")
    print("  Then load the folder as unpacked extension in chrome://extensions")


if __name__ == "__main__":
    main()
