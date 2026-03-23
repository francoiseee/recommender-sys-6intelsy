"""Download or generate local data for the recommendation pipeline.

This script creates a deterministic synthetic dataset suitable for offline ranking
and contextual bandit simulation.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"


def _build_items(rng: np.random.Generator, n_items_per_category: int = 120) -> pd.DataFrame:
    categories = {
        "sports": ["match", "player", "league", "coach", "season"],
        "politics": ["election", "policy", "senate", "vote", "minister"],
        "technology": ["ai", "chip", "startup", "cloud", "software"],
        "health": ["hospital", "diet", "vaccine", "doctor", "wellness"],
        "business": ["market", "stocks", "inflation", "trade", "investor"],
    }

    rows = []
    item_id = 0
    for category, words in categories.items():
        for _ in range(n_items_per_category):
            headline = f"{category} update {rng.integers(10000)}"
            body_tokens = rng.choice(words, size=8, replace=True)
            text = f"{headline} {' '.join(body_tokens)}"
            rows.append({"item_id": item_id, "category": category, "text": text})
            item_id += 1
    return pd.DataFrame(rows)


def _build_impressions(
    rng: np.random.Generator,
    items: pd.DataFrame,
    n_users: int = 120,
    impressions_per_user: int = 20,
    candidates_per_impression: int = 10,
) -> pd.DataFrame:
    categories = sorted(items["category"].unique().tolist())
    item_ids_by_cat = {
        cat: items.loc[items["category"] == cat, "item_id"].to_numpy() for cat in categories
    }
    all_item_ids = items["item_id"].to_numpy()

    rows = []
    impression_id = 0
    for user_id in range(n_users):
        user_pref = categories[user_id % len(categories)]
        for _ in range(impressions_per_user):
            candidates = rng.choice(all_item_ids, size=candidates_per_impression, replace=False)

            # Bias clicked item toward preferred category so models have signal.
            if rng.random() < 0.8:
                clicked_item = int(rng.choice(item_ids_by_cat[user_pref]))
                replace_idx = int(rng.integers(0, candidates_per_impression))
                candidates[replace_idx] = clicked_item
            else:
                clicked_item = int(rng.choice(candidates))

            rows.append(
                {
                    "impression_id": impression_id,
                    "user_id": user_id,
                    "user_pref": user_pref,
                    "candidate_item_ids": " ".join(map(str, candidates.tolist())),
                    "clicked_item_id": clicked_item,
                }
            )
            impression_id += 1
    return pd.DataFrame(rows)


def generate_synthetic_data(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    items = _build_items(rng=rng)
    impressions = _build_impressions(rng=rng, items=items)

    items.to_csv(PROCESSED_DIR / "items.csv", index=False)
    impressions.to_csv(PROCESSED_DIR / "impressions.csv", index=False)

    metadata = {
        "dataset_name": "synthetic_newsrec_v1",
        "source": "generated",
        "license": "MIT (project-generated)",
        "contains_pii": False,
        "seed": seed,
        "num_items": int(len(items)),
        "num_impressions": int(len(impressions)),
        "num_users": int(impressions["user_id"].nunique()),
    }
    with open(PROCESSED_DIR / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[get_data] Generated data at {PROCESSED_DIR}")
    print(
        "[get_data] "
        f"items={metadata['num_items']} impressions={metadata['num_impressions']} users={metadata['num_users']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Download/generate dataset for recommendation pipeline")
    parser.add_argument("--force", action="store_true", help="Re-download even if data exists")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation")
    args = parser.parse_args()

    items_path = PROCESSED_DIR / "items.csv"
    impressions_path = PROCESSED_DIR / "impressions.csv"

    if items_path.exists() and impressions_path.exists() and not args.force:
        print(f"[get_data] Data already exists at {PROCESSED_DIR}. Use --force to regenerate.")
    else:
        generate_synthetic_data(seed=args.seed)


if __name__ == "__main__":
    main()
