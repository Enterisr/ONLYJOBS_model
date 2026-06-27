#!/usr/bin/env python3
"""
Fast export: load saved backbone -> embed dataset.csv -> fit baseline LR ->
write classifier.json to model_output/ and copy to ../linkedin-job-filter/.

Run this instead of the full train.py when you just need to refresh classifier.json.
"""
import json, shutil
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from train import (
    MODEL_NAME, EMBED_DIM, OUTPUT_DIR,
    load_data, embed_texts, export_classifier,
)

MODEL_DIR = OUTPUT_DIR / "pytorch_model"
EXT_DIR   = Path(__file__).parent.parent / "linkedin-job-filter"


def main():
    print("Loading tokenizer + backbone ...")
    if MODEL_DIR.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        backbone  = AutoModel.from_pretrained(str(MODEL_DIR))
        print(f"  loaded from {MODEL_DIR}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        backbone  = AutoModel.from_pretrained(MODEL_NAME)
        print(f"  loaded from HuggingFace ({MODEL_NAME})")

    print("Embedding dataset.csv ...")
    texts, labels = load_data("dataset.csv")
    label_arr = np.array(labels)
    raw_emb = embed_texts(backbone, tokenizer, texts).numpy()
    print(f"  {len(texts)} samples, embedding shape {raw_emb.shape}")

    print("Fitting baseline LR ...")
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(raw_emb, label_arr)

    out_path = OUTPUT_DIR / "classifier.json"
    OUTPUT_DIR.mkdir(exist_ok=True)
    export_classifier(pipe, out_path)

    if EXT_DIR.exists():
        shutil.copy(out_path, EXT_DIR / "classifier.json")
        print(f"  copied -> {EXT_DIR / 'classifier.json'}")
    else:
        print(f"  (extension dir not found at {EXT_DIR} — copy manually)")

    print("\nDone. classifier.json uses raw backbone embeddings (no projection).")


if __name__ == "__main__":
    main()
