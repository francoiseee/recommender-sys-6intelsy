"""
eval.py — Evaluation script: nDCG, Hit@K, cumulative reward.
"""

import argparse
import numpy as np


def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """Compute nDCG@K."""
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def hit_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """Compute Hit@K."""
    return float(any(item in relevant for item in recommended[:k]))


def evaluate_all():
    """Evaluate all trained models and save results. TODO: Implement in Week 2/3."""
    print("[eval] TODO: Implement full evaluation in Week 2/3.")
    print("[eval] Metrics to compute: nDCG@10, Hit@10, Cumulative RL Reward")


def main():
    parser = argparse.ArgumentParser(description="Evaluate recommendation models")
    parser.add_argument("--all", action="store_true", help="Evaluate all models")
    parser.add_argument("--model", type=str, default=None, help="Evaluate specific model")
    args = parser.parse_args()

    if args.all:
        evaluate_all()
    else:
        print(f"[eval] Evaluating model: {args.model} — TODO")


if __name__ == "__main__":
    main()
