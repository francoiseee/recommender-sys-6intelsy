from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.data_pipeline import BanditEvent, build_news_feature_store, load_bandit_events
from src.eval import evaluate_bandit
from src.rl_agent import LinUCBAgent, LinUCBConfig


@dataclass
class MLPConfig:
    hidden_dim: int = 64
    epochs: int = 3
    lr: float = 0.03
    l2: float = 1e-4
    max_train_samples: int = 40000
    batch_size: int = 256


class NumpyMLP:
    def __init__(self, input_dim: int, config: MLPConfig, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.config = config
        self.W1 = (rng.normal(0.0, 0.05, size=(input_dim, config.hidden_dim))).astype(np.float32)
        self.b1 = np.zeros(config.hidden_dim, dtype=np.float32)
        self.W2 = (rng.normal(0.0, 0.05, size=(config.hidden_dim, 1))).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.maximum(0.0, X @ self.W1 + self.b1)
        p = self._sigmoid(h @ self.W2 + self.b2)
        return h, p

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n = X.shape[0]
        batch_size = max(1, self.config.batch_size)

        for _ in range(self.config.epochs):
            perm = np.random.permutation(n)
            X_shuf = X[perm]
            y_shuf = y[perm]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = X_shuf[start:end]
                yb = y_shuf[start:end].reshape(-1, 1)

                h, p = self._forward(xb)
                grad_logits = (p - yb) / max(1, xb.shape[0])

                grad_W2 = h.T @ grad_logits + self.config.l2 * self.W2
                grad_b2 = np.sum(grad_logits, axis=0)

                grad_h = grad_logits @ self.W2.T
                grad_h[h <= 0] = 0

                grad_W1 = xb.T @ grad_h + self.config.l2 * self.W1
                grad_b1 = np.sum(grad_h, axis=0)

                self.W2 -= self.config.lr * grad_W2
                self.b2 -= self.config.lr * grad_b2
                self.W1 -= self.config.lr * grad_W1
                self.b1 -= self.config.lr * grad_b1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, p = self._forward(X)
        return p.reshape(-1)


def _split_events(events: Sequence[BanditEvent], train_ratio: float) -> tuple[List[BanditEvent], List[BanditEvent]]:
    split_idx = max(1, int(len(events) * train_ratio))
    return list(events[:split_idx]), list(events[split_idx:])


def _build_popularity_stats(train_events: Iterable[BanditEvent]) -> Dict[str, float]:
    clicks: Dict[str, float] = {}
    views: Dict[str, float] = {}

    for event in train_events:
        for news_id, reward in zip(event.arm_news_ids, event.rewards):
            views[news_id] = views.get(news_id, 0.0) + 1.0
            clicks[news_id] = clicks.get(news_id, 0.0) + float(reward)

    scores: Dict[str, float] = {}
    for news_id, v in views.items():
        scores[news_id] = clicks.get(news_id, 0.0) / max(1.0, v)
    return scores


def _evaluate_popularity(val_events: Iterable[BanditEvent], popularity_score: Dict[str, float]) -> Dict[str, float]:
    rewards: List[float] = []
    regrets: List[float] = []

    count = 0
    for event in val_events:
        score_vec = np.asarray([popularity_score.get(nid, 0.0) for nid in event.arm_news_ids], dtype=np.float32)
        idx = int(np.argmax(score_vec))
        chosen_reward = float(event.rewards[idx])
        rewards.append(chosen_reward)
        regrets.append(float(np.max(event.rewards)) - chosen_reward)
        count += 1

    if count == 0:
        return {"events": 0.0, "ctr_at_1": 0.0, "avg_reward": 0.0, "avg_regret": 0.0}

    return {
        "events": float(count),
        "ctr_at_1": float(np.mean(rewards)),
        "avg_reward": float(np.mean(rewards)),
        "avg_regret": float(np.mean(regrets)),
    }


def _evaluate_content_similarity(val_events: Iterable[BanditEvent]) -> Dict[str, float]:
    rewards: List[float] = []
    regrets: List[float] = []

    count = 0
    for event in val_events:
        d = int(event.arm_features.shape[1] // 3)
        candidate = event.arm_features[:, :d]
        user = event.arm_features[:, d : 2 * d]

        num = np.sum(candidate * user, axis=1)
        den = np.linalg.norm(candidate, axis=1) * np.linalg.norm(user, axis=1)
        score = num / np.maximum(den, 1e-9)

        idx = int(np.argmax(score))
        chosen_reward = float(event.rewards[idx])
        rewards.append(chosen_reward)
        regrets.append(float(np.max(event.rewards)) - chosen_reward)
        count += 1

    if count == 0:
        return {"events": 0.0, "ctr_at_1": 0.0, "avg_reward": 0.0, "avg_regret": 0.0}

    return {
        "events": float(count),
        "ctr_at_1": float(np.mean(rewards)),
        "avg_reward": float(np.mean(rewards)),
        "avg_regret": float(np.mean(regrets)),
    }


def _collect_training_pairs(events: Sequence[BanditEvent], max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pairs: List[tuple[np.ndarray, float]] = []
    for event in events:
        for x, r in zip(event.arm_features, event.rewards):
            pairs.append((x, float(r)))

    rng = random.Random(seed)
    if len(pairs) > max_samples:
        pairs = rng.sample(pairs, max_samples)

    X = np.vstack([p[0] for p in pairs]).astype(np.float32)
    y = np.asarray([p[1] for p in pairs], dtype=np.float32)
    return X, y


def _evaluate_mlp(mlp: NumpyMLP, val_events: Iterable[BanditEvent]) -> Dict[str, float]:
    rewards: List[float] = []
    regrets: List[float] = []

    count = 0
    for event in val_events:
        prob = mlp.predict_proba(event.arm_features)
        idx = int(np.argmax(prob))
        chosen_reward = float(event.rewards[idx])
        rewards.append(chosen_reward)
        regrets.append(float(np.max(event.rewards)) - chosen_reward)
        count += 1

    if count == 0:
        return {"events": 0.0, "ctr_at_1": 0.0, "avg_reward": 0.0, "avg_regret": 0.0}

    return {
        "events": float(count),
        "ctr_at_1": float(np.mean(rewards)),
        "avg_reward": float(np.mean(rewards)),
        "avg_regret": float(np.mean(regrets)),
    }


def _svg_polyline(path: Path, title: str, xs: Sequence[float], ys: Sequence[float], x_label: str, y_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 900, 420
    left, right, top, bottom = 70, 40, 40, 60
    plot_w = w - left - right
    plot_h = h - top - bottom

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1e-6

    points = []
    for x, y in zip(xs, ys):
        px = left + ((x - x_min) / (x_max - x_min)) * plot_w
        py = top + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h
        points.append(f"{px:.2f},{py:.2f}")

    content = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'>
  <rect x='0' y='0' width='{w}' height='{h}' fill='white'/>
  <text x='{w/2:.0f}' y='25' text-anchor='middle' font-family='Arial' font-size='18'>{title}</text>
  <line x1='{left}' y1='{top+plot_h}' x2='{left+plot_w}' y2='{top+plot_h}' stroke='black'/>
  <line x1='{left}' y1='{top}' x2='{left}' y2='{top+plot_h}' stroke='black'/>
  <polyline fill='none' stroke='#0b57d0' stroke-width='2' points='{' '.join(points)}'/>
  <text x='{w/2:.0f}' y='{h-15}' text-anchor='middle' font-family='Arial' font-size='14'>{x_label}</text>
  <text x='18' y='{h/2:.0f}' transform='rotate(-90 18,{h/2:.0f})' text-anchor='middle' font-family='Arial' font-size='14'>{y_label}</text>
</svg>
"""
    path.write_text(content, encoding="utf-8")


def _write_metrics_csv(path: Path, rows: List[Dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "events", "ctr_at_1", "avg_reward", "avg_regret"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recommender baselines and export checkpoint artifacts.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-news", type=int, default=50000)
    parser.add_argument("--max-events", type=int, default=30000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--category-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curve-every", type=int, default=500)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-epochs", type=int, default=3)
    parser.add_argument("--mlp-lr", type=float, default=0.03)
    parser.add_argument("--mlp-max-train-samples", type=int, default=40000)
    parser.add_argument("--mlp-batch-size", type=int, default=256)
    parser.add_argument("--output-table", type=Path, default=Path("results") / "tables" / "baseline_metrics.csv")
    parser.add_argument(
        "--output-curve",
        type=Path,
        default=Path("results") / "tables" / "linucb_learning_curve.csv",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("results") / "figures" / "linucb_learning_curve.svg",
    )
    parser.add_argument(
        "--output-bar-figure",
        type=Path,
        default=Path("results") / "figures" / "baseline_ctr_comparison.svg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    news_features = build_news_feature_store(
        news_path=args.data_dir / "news.tsv",
        entity_embedding_path=args.data_dir / "entity_embedding.vec",
        text_dim=args.text_dim,
        category_dim=args.category_dim,
        max_news=args.max_news,
        show_progress=True,
    )
    events = load_bandit_events(
        behaviors_path=args.data_dir / "behaviors.tsv",
        news_features=news_features,
        max_events=args.max_events,
        show_progress=True,
    )

    train_events, val_events = _split_events(events, args.train_ratio)
    if not val_events:
        raise RuntimeError("Validation split is empty; reduce train-ratio or increase max-events")

    # Baseline 1: popularity
    popularity_scores = _build_popularity_stats(train_events)
    popularity_metrics = _evaluate_popularity(val_events, popularity_scores)

    # Baseline 2: content similarity
    content_metrics = _evaluate_content_similarity(val_events)

    # Baseline 3: LinUCB contextual bandit
    context_dim = int(train_events[0].arm_features.shape[1])
    linucb = LinUCBAgent(LinUCBConfig(context_dim=context_dim, alpha=1.0, l2_reg=1.0))

    curve_rows = []
    cumulative = 0.0
    for i, event in enumerate(train_events, start=1):
        cumulative += linucb.learn_from_logged_sample(event.arm_features, event.rewards)
        if i % max(1, args.curve_every) == 0 or i == len(train_events):
            curve_rows.append({"step": i, "avg_reward": cumulative / i})

    linucb_metrics = evaluate_bandit(linucb, val_events)

    # Baseline 4: shallow DL (NumPy MLP)
    mlp_config = MLPConfig(
        hidden_dim=args.mlp_hidden_dim,
        epochs=args.mlp_epochs,
        lr=args.mlp_lr,
        max_train_samples=args.mlp_max_train_samples,
        batch_size=args.mlp_batch_size,
    )
    X_train, y_train = _collect_training_pairs(train_events, mlp_config.max_train_samples, args.seed)
    mlp = NumpyMLP(input_dim=X_train.shape[1], config=mlp_config, seed=args.seed)
    mlp.fit(X_train, y_train)
    mlp_metrics = _evaluate_mlp(mlp, val_events)

    rows: List[Dict[str, float | str]] = [
        {"model": "popularity", **popularity_metrics},
        {"model": "content_similarity", **content_metrics},
        {"model": "linucb", **linucb_metrics},
        {"model": "mlp_dl_baseline", **mlp_metrics},
    ]
    _write_metrics_csv(args.output_table, rows)

    args.output_curve.parent.mkdir(parents=True, exist_ok=True)
    with args.output_curve.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "avg_reward"])
        writer.writeheader()
        for row in curve_rows:
            writer.writerow(row)

    _svg_polyline(
        path=args.output_figure,
        title="LinUCB Learning Curve (Training)",
        xs=[float(r["step"]) for r in curve_rows],
        ys=[float(r["avg_reward"]) for r in curve_rows],
        x_label="Training Step",
        y_label="Cumulative Avg Reward",
    )

    # Compare validation CTR across models.
    ctr_rows = [(str(r["model"]), float(r["ctr_at_1"])) for r in rows]
    ctr_rows = sorted(ctr_rows, key=lambda x: x[1])
    x_idx = list(range(1, len(ctr_rows) + 1))
    y_vals = [r[1] for r in ctr_rows]
    _svg_polyline(
        path=args.output_bar_figure,
        title="Validation CTR Comparison",
        xs=[float(v) for v in x_idx],
        ys=y_vals,
        x_label="Model Index (sorted by CTR)",
        y_label="CTR@1",
    )

    print("\n=== Baseline Metrics ===")
    for row in rows:
        print(
            f"{row['model']}: ctr_at_1={float(row['ctr_at_1']):.4f}, "
            f"avg_reward={float(row['avg_reward']):.4f}, avg_regret={float(row['avg_regret']):.4f}"
        )
    print(f"Metrics table saved to: {args.output_table}")
    print(f"Learning curve saved to: {args.output_curve}")
    print(f"Figure saved to: {args.output_figure}")
    print(f"Figure saved to: {args.output_bar_figure}")


if __name__ == "__main__":
    main()
