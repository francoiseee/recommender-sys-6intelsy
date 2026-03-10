"""
train.py — Training script for the recommendation model.
"""

import argparse
import random
import numpy as np
import torch
import yaml
import os
import json
from datetime import datetime


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg: dict):
    """Main training loop. TODO: Implement in Week 2."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    model_name = cfg.get("model", "cnn_ranker")
    epochs = cfg.get("epochs", 10)
    lr = cfg.get("lr", 1e-3)
    batch_size = cfg.get("batch_size", 32)

    print(f"[train] Model: {model_name} | Epochs: {epochs} | LR: {lr} | Batch: {batch_size}")
    print(f"[train] Seed: {seed}")
    print("[train] TODO: Implement full training loop in Week 2.")

    # Log hyperparameters
    os.makedirs("experiments/logs", exist_ok=True)
    log = {
        "model": model_name,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }
    log_path = f"experiments/logs/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
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
