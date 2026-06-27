#!/usr/bin/env python3
"""
Evaluate baseline vs SupCon-trained model on held-out test.csv.

Baseline : frozen backbone mean-pool -> StandardScaler -> LogisticRegression
Trained  : same backbone -> projector.pt projection -> StandardScaler -> LogisticRegression

Both LR heads are fitted on dataset.csv and scored on test.csv.
Backbone and projector are loaded from model_output/pytorch_model/.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

from train import (
    MODEL_NAME, PROJ_DIM, EMBED_DIM,
    PostDataset, Embedder, embed_texts,
    get_embeddings,
    OUTPUT_DIR,
)

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

TRAIN_CSV = "dataset.csv"
TEST_CSV  = "test.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(csv_path):
    df     = pd.read_csv(csv_path, engine="python", on_bad_lines="warn")
    texts  = df.iloc[:, 0].astype(str).tolist()
    labels = df.iloc[:, 1].astype(int).tolist()
    return texts, labels


def make_loader(emb_tensor, labels):
    ds = PostDataset(emb_tensor, labels)
    return DataLoader(ds, batch_size=len(ds), shuffle=False)


def metrics(y_true, y_pred):
    return {
        "acc":       accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_bar_comparison(base_m, trained_m, plots_dir):
    names  = ["Accuracy", "F1", "Precision", "Recall"]
    base_v = [base_m["acc"], base_m["f1"], base_m["precision"], base_m["recall"]]
    sup_v  = [trained_m["acc"], trained_m["f1"], trained_m["precision"], trained_m["recall"]]
    x      = np.arange(len(names))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, base_v, w, label="Baseline (backbone + LR)", color="#94a3b8", alpha=0.85)
    b2 = ax.bar(x + w/2, sup_v,  w, label="SupCon (projection + LR)", color="#2563eb", alpha=0.85)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs SupCon - test.csv (held-out)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    p = plots_dir / "test_comparison.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {p}")


def plot_cms(y_true, base_preds, trained_preds, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Confusion matrices - test.csv", fontsize=12)
    for ax, preds, title in zip(
        axes,
        [base_preds, trained_preds],
        ["Baseline (backbone + LR)", "SupCon (projection + LR)"],
    ):
        cm = confusion_matrix(y_true, preds)
        ConfusionMatrixDisplay(cm, display_labels=["non-job", "job"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title)
    fig.tight_layout()
    p = plots_dir / "test_confusion_matrices.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plots_dir  = OUTPUT_DIR / "plots"
    model_dir  = OUTPUT_DIR / "pytorch_model"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("Loading data...")
    train_texts, train_labels = load_data(TRAIN_CSV)
    test_texts,  test_labels  = load_data(TEST_CSV)
    train_arr = np.array(train_labels)
    test_arr  = np.array(test_labels)
    print(f"  train : {len(train_texts)} samples  job={train_arr.sum()}  non-job={(train_arr==0).sum()}")
    print(f"  test  : {len(test_texts)} samples   job={test_arr.sum()}  non-job={(test_arr==0).sum()}")

    # 2. Load backbone + tokenizer from saved model_output
    print(f"\nLoading tokeniser + backbone from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    backbone  = AutoModel.from_pretrained(str(model_dir))

    # 3. Precompute backbone embeddings with chunking (shared by both branches)
    print("Extracting backbone embeddings with chunking...")
    raw_train_t = embed_texts(backbone, tokenizer, train_texts)
    raw_test_t  = embed_texts(backbone, tokenizer, test_texts)
    raw_train   = raw_train_t.numpy()
    raw_test    = raw_test_t.numpy()

    train_loader = make_loader(raw_train_t, train_labels)
    test_loader  = make_loader(raw_test_t,  test_labels)

    # Baseline
    print("\nBaseline: fit on raw train -> score on test ...")
    base_pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
    ])
    base_pipe.fit(raw_train, train_arr)
    base_preds = base_pipe.predict(raw_test)
    base_m     = metrics(test_arr, base_preds)
    print(f"  acc={base_m['acc']:.3f}  f1={base_m['f1']:.3f}  "
          f"prec={base_m['precision']:.3f}  rec={base_m['recall']:.3f}")

    # SupCon trained: load saved projector weights
    proj_path = model_dir / "projector.pt"
    print(f"\nLoading trained projector from {proj_path}...")
    model = Embedder()
    state = torch.load(str(proj_path), map_location="cpu", weights_only=True)
    model.projector.load_state_dict(state)

    proj_train, _ = get_embeddings(model, train_loader)
    proj_test,  _ = get_embeddings(model, test_loader)

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
    ])
    lr_pipe.fit(proj_train, train_arr)
    trained_preds = lr_pipe.predict(proj_test)
    trained_m = metrics(test_arr, trained_preds)
    print(f"  acc={trained_m['acc']:.3f}  f1={trained_m['f1']:.3f}  "
          f"prec={trained_m['precision']:.3f}  rec={trained_m['recall']:.3f}")

    # Summary
    print("\n-- Test-set results --------------------------------------------------")
    print(f"  {'':20}  {'Acc':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    print(f"  {'Baseline':20}  {base_m['acc']:6.3f}  {base_m['f1']:6.3f}  "
          f"{base_m['precision']:6.3f}  {base_m['recall']:6.3f}")
    print(f"  {'SupCon':20}  {trained_m['acc']:6.3f}  {trained_m['f1']:6.3f}  "
          f"{trained_m['precision']:6.3f}  {trained_m['recall']:6.3f}")
    delta = trained_m["f1"] - base_m["f1"]
    print(f"\n  dF1 (SupCon - baseline): {delta:+.3f}")

    # Plots
    print("\nGenerating plots...")
    plot_bar_comparison(base_m, trained_m, plots_dir)
    plot_cms(test_arr, base_preds, trained_preds, plots_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
