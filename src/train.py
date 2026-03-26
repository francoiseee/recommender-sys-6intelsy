"""Training entrypoint for baseline and Text-CNN click-ranker models."""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml

from data_pipeline import build_vocab, get_dataloader, load_raw_data, preprocess, split_data
from models.text_cnn import TextCNNClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dirs():
    os.makedirs("experiments/logs", exist_ok=True)
    os.makedirs("experiments/models", exist_ok=True)
    os.makedirs("experiments/results", exist_ok=True)


def _load_dataframes():
    items, impressions = load_raw_data("data")
    expanded = preprocess(items=items, impressions=impressions)
    train_df, val_df, test_df = split_data(expanded)
    return train_df, val_df, test_df


def _validate(model: torch.nn.Module, loader, device: torch.device, loss_fn) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(inputs).squeeze(1)
            loss = loss_fn(logits, labels)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def train_popularity_baseline(cfg: dict):
    train_df, _, test_df = _load_dataframes()
    popularity = train_df.groupby("item_id")["label"].mean().to_dict()
    global_ctr = float(train_df["label"].mean())

    baseline_artifact = {
        "model": "popularity_baseline",
        "global_ctr": global_ctr,
        "item_popularity": {str(k): float(v) for k, v in popularity.items()},
        "seed": int(cfg.get("seed", 42)),
    }

    model_path = "experiments/models/popularity_baseline.json"
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(baseline_artifact, f, indent=2)

    summary = {
        "model": "popularity_baseline",
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "global_ctr": global_ctr,
    }
    out_path = "experiments/results/train_summary_popularity_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] Saved popularity baseline to {model_path}")


def train_cnn_ranker(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df = _load_dataframes()

    max_len = int(cfg.get("max_len", 48))
    batch_size = int(cfg.get("batch_size", 64))
    epochs = int(cfg.get("epochs", 5))
    lr = float(cfg.get("lr", 1e-3))

    vocab = build_vocab(train_df["combined_text"])
    train_loader = get_dataloader(train_df, vocab=vocab, batch_size=batch_size, shuffle=True, max_len=max_len)
    val_loader = get_dataloader(val_df, vocab=vocab, batch_size=batch_size, shuffle=False, max_len=max_len)

    model = TextCNNClassifier(
        vocab_size=len(vocab),
        num_classes=1,
        embed_dim=int(cfg.get("embed_dim", 128)),
        num_filters=int(cfg.get("num_filters", 128)),
        filter_sizes=list(cfg.get("filter_sizes", [2, 3, 4])),
        output_dim=int(cfg.get("output_dim", 256)),
        dropout=float(cfg.get("dropout", 0.3)),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    patience = int(cfg.get("early_stopping_patience", 2))
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        total_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader, start=1):
            inputs = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(inputs).squeeze(1)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if batch_idx % 200 == 0 or batch_idx == total_batches:
                print(
                    f"[train] epoch={epoch:02d} batch={batch_idx}/{total_batches} "
                    f"loss={loss.item():.4f}"
                )

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = _validate(model, val_loader, device, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[train] epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print("[train] Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = "experiments/models/cnn_ranker.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "config": cfg,
            "best_val_loss": best_val_loss,
        },
        model_path,
    )

    summary = {
        "model": "cnn_ranker",
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "vocab_size": int(len(vocab)),
        "best_val_loss": float(best_val_loss),
        "history": history,
    }
    out_path = "experiments/results/train_summary_cnn_ranker.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] Saved CNN ranker checkpoint to {model_path}")


def train(cfg: dict):
    """Main training dispatcher."""
    _ensure_dirs()
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    model_name = cfg.get("model", "cnn_ranker")
    print(f"[train] model={model_name} seed={seed}")
    if model_name == "popularity_baseline":
        train_popularity_baseline(cfg)
    elif model_name == "cnn_ranker":
        train_cnn_ranker(cfg)
    else:
        raise ValueError(f"Unsupported model '{model_name}'.")

    log = {
        "model": model_name,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "config": cfg,
    }
    log_path = f"experiments/logs/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[train] Config logged to {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train recommendation model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
