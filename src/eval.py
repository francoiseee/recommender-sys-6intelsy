"""Evaluation script for recommendation ranking metrics."""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from data_pipeline import build_vocab, encode_text, load_raw_data, preprocess, split_data
from models.text_cnn import TextCNNClassifier


def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """Compute nDCG@K."""
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def hit_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """Compute Hit@K."""
    return float(any(item in relevant for item in recommended[:k]))


def _prepare_splits():
    items, impressions = load_raw_data("data")
    rows = preprocess(items=items, impressions=impressions)
    return split_data(rows)


def _rank_metrics(pred_df: pd.DataFrame, k: int = 10) -> dict:
    ndcgs = []
    hits = []
    grouped = pred_df.groupby("impression_id")
    for _, grp in grouped:
        ranked = grp.sort_values("score", ascending=False)
        recommended = ranked["item_id"].tolist()
        relevant = set(grp.loc[grp["label"] > 0.5, "item_id"].tolist())
        ndcgs.append(ndcg_at_k(recommended, relevant, k=k))
        hits.append(hit_at_k(recommended, relevant, k=k))
    return {
        f"nDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"Hit@{k}": float(np.mean(hits)) if hits else 0.0,
        "num_impressions": int(len(grouped)),
    }


def _predict_random(test_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = test_df[["impression_id", "item_id", "label"]].copy()
    out["score"] = rng.random(len(out))
    return out


def _predict_popularity(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    popularity = train_df.groupby("item_id")["label"].mean().to_dict()
    default_score = float(train_df["label"].mean())
    out = test_df[["impression_id", "item_id", "label"]].copy()
    out["score"] = out["item_id"].map(lambda x: popularity.get(int(x), default_score))
    return out


def _predict_cnn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    ckpt_path = "experiments/models/cnn_ranker.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Missing experiments/models/cnn_ranker.pt. Run training first.")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = checkpoint.get("config", {})
    vocab = checkpoint.get("vocab") or build_vocab(train_df["combined_text"])

    model = TextCNNClassifier(
        vocab_size=len(vocab),
        num_classes=1,
        embed_dim=int(cfg.get("embed_dim", 128)),
        num_filters=int(cfg.get("num_filters", 128)),
        filter_sizes=list(cfg.get("filter_sizes", [2, 3, 4])),
        output_dim=int(cfg.get("output_dim", 256)),
        dropout=float(cfg.get("dropout", 0.3)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    max_len = int(cfg.get("max_len", 48))
    scores = []
    with torch.no_grad():
        for row in test_df.itertuples(index=False):
            token_ids = torch.tensor([encode_text(row.combined_text, vocab=vocab, max_len=max_len)])
            logit = model(token_ids).squeeze().item()
            score = 1.0 / (1.0 + np.exp(-logit))
            scores.append(float(score))

    out = test_df[["impression_id", "item_id", "label"]].copy()
    out["score"] = scores
    return out


def evaluate_all(k: int = 10):
    """Evaluate all models and export JSON/CSV summaries."""
    os.makedirs("experiments/results", exist_ok=True)

    train_df, _, test_df = _prepare_splits()
    eval_table = []

    random_preds = _predict_random(test_df)
    random_metrics = _rank_metrics(random_preds, k=k)
    random_metrics["model"] = "random_baseline"
    eval_table.append(random_metrics)
    random_preds.to_csv("experiments/results/predictions_random_baseline.csv", index=False)

    pop_preds = _predict_popularity(train_df, test_df)
    pop_metrics = _rank_metrics(pop_preds, k=k)
    pop_metrics["model"] = "popularity_baseline"
    eval_table.append(pop_metrics)
    pop_preds.to_csv("experiments/results/predictions_popularity_baseline.csv", index=False)

    try:
        cnn_preds = _predict_cnn(train_df, test_df)
        cnn_metrics = _rank_metrics(cnn_preds, k=k)
        cnn_metrics["model"] = "cnn_ranker"
        eval_table.append(cnn_metrics)
        cnn_preds.to_csv("experiments/results/predictions_cnn_ranker.csv", index=False)
    except FileNotFoundError as exc:
        print(f"[eval] Skipping cnn_ranker: {exc}")

    result_df = pd.DataFrame(eval_table)
    result_df.to_csv("experiments/results/metrics_summary.csv", index=False)
    with open("experiments/results/metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_table, f, indent=2)

    print("[eval] Saved metrics to experiments/results/metrics_summary.csv")
    print(result_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Evaluate recommendation models")
    parser.add_argument("--all", action="store_true", help="Evaluate all models")
    parser.add_argument("--k", type=int, default=10, help="Top-K cutoff for ranking metrics")
    args = parser.parse_args()

    if args.all:
        evaluate_all(k=args.k)
    else:
        evaluate_all(k=args.k)


if __name__ == "__main__":
    main()
