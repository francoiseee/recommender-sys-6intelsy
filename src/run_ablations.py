"""Run ablation experiments for the Text-CNN ranker."""

import argparse
import copy
import json
import os

from eval import evaluate_all
from train import train


def run_ablations(base_cfg: dict):
    os.makedirs("experiments/results", exist_ok=True)

    ablations = [
        {
            "name": "ablation_small_filters",
            "overrides": {"filter_sizes": [2, 3], "num_filters": 64},
        },
        {
            "name": "ablation_high_dropout",
            "overrides": {"dropout": 0.5},
        },
    ]

    summary = []
    for spec in ablations:
        cfg = copy.deepcopy(base_cfg)
        cfg.update(spec["overrides"])
        cfg["model"] = "cnn_ranker"
        cfg["epochs"] = int(cfg.get("ablation_epochs", 3))
        cfg["batch_size"] = int(cfg.get("ablation_batch_size", 64))

        print(f"[ablations] Running {spec['name']} with overrides={spec['overrides']}")
        train(cfg)
        evaluate_all(k=10)

        metrics_path = "experiments/results/metrics_summary.json"
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        cnn_row = next((m for m in metrics if m.get("model") == "cnn_ranker"), None)
        summary.append(
            {
                "ablation": spec["name"],
                "overrides": spec["overrides"],
                "cnn_metrics": cnn_row,
            }
        )

    out_path = "experiments/results/ablation_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[ablations] Saved summary to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Text-CNN ablations")
    parser.add_argument("--config", type=str, default="experiments/configs/cnn_ranker.yaml")
    args = parser.parse_args()

    import yaml

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_ablations(cfg)


if __name__ == "__main__":
    main()
