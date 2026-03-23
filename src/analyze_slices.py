"""Generate slice/error analysis tables and plots for rubric coverage."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from eval import hit_at_k, ndcg_at_k


RESULTS_DIR = Path("experiments/results")
PLOTS_DIR = RESULTS_DIR / "plots"


def _load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    impressions = pd.read_csv("data/processed/impressions.csv")
    items = pd.read_csv("data/processed/items.csv")
    return impressions, items


def _prediction_files() -> dict:
    return {
        "random_baseline": RESULTS_DIR / "predictions_random_baseline.csv",
        "popularity_baseline": RESULTS_DIR / "predictions_popularity_baseline.csv",
        "cnn_ranker": RESULTS_DIR / "predictions_cnn_ranker.csv",
    }


def _impression_level_metrics(pred_df: pd.DataFrame, impressions: pd.DataFrame, items: pd.DataFrame, model: str, k: int = 10) -> pd.DataFrame:
    item_to_category = dict(zip(items["item_id"], items["category"]))
    imp_meta = impressions[["impression_id", "user_id", "user_pref", "clicked_item_id"]].copy()

    rows = []
    grouped = pred_df.groupby("impression_id")
    for impression_id, grp in grouped:
        ranked = grp.sort_values("score", ascending=False).reset_index(drop=True)
        recommended = ranked["item_id"].tolist()
        relevant = set(grp.loc[grp["label"] > 0.5, "item_id"].tolist())

        clicked_item = int(next(iter(relevant))) if relevant else int(ranked.iloc[0]["item_id"])
        ranks = ranked.index[ranked["item_id"] == clicked_item].tolist()
        click_rank = int(ranks[0] + 1) if ranks else len(ranked) + 1

        ndcg = ndcg_at_k(recommended, relevant, k=k)
        hit = hit_at_k(recommended, relevant, k=k)
        miss = float(1.0 - hit)

        meta_row = imp_meta.loc[imp_meta["impression_id"] == impression_id].iloc[0]
        clicked_category = item_to_category.get(int(meta_row["clicked_item_id"]), "unknown")

        rows.append(
            {
                "impression_id": int(impression_id),
                "user_id": int(meta_row["user_id"]),
                "user_pref": str(meta_row["user_pref"]),
                "clicked_category": str(clicked_category),
                "nDCG@10": float(ndcg),
                "Hit@10": float(hit),
                "miss_rate@10": float(miss),
                "click_rank": int(click_rank),
                "is_miss": int(click_rank > k),
                "model": model,
            }
        )

    return pd.DataFrame(rows)


def _aggregate_slices(impression_metrics: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = (
        impression_metrics
        .groupby([group_col, "model"], as_index=False)
        .agg(
            {
                "nDCG@10": "mean",
                "Hit@10": "mean",
                "miss_rate@10": "mean",
                "impression_id": "count",
            }
        )
        .rename(columns={"impression_id": "num_impressions"})
        .sort_values(["model", group_col])
    )
    return out


def _make_plots(slice_user: pd.DataFrame, slice_cat: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=slice_user, x="user_pref", y="nDCG@10", hue="model")
    plt.title("nDCG@10 by User Preference")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ndcg_by_user_pref.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=slice_cat, x="clicked_category", y="Hit@10", hue="model")
    plt.title("Hit@10 by Clicked Category")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hit_by_clicked_category.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=slice_user, x="user_pref", y="miss_rate@10", hue="model")
    plt.title("Miss Rate@10 by User Preference")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "miss_rate_by_user_pref.png", dpi=160)
    plt.close()


def analyze_slices(k: int = 10):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    impressions, items = _load_metadata()

    all_impression_metrics = []
    files = _prediction_files()
    for model, path in files.items():
        if not path.exists():
            print(f"[analyze_slices] Skipping {model}: missing {path}")
            continue
        pred = pd.read_csv(path)
        all_impression_metrics.append(
            _impression_level_metrics(pred, impressions, items, model=model, k=k)
        )

    if not all_impression_metrics:
        raise FileNotFoundError("No prediction files found. Run `python src/eval.py --all` first.")

    metrics_df = pd.concat(all_impression_metrics, ignore_index=True)
    metrics_df.to_csv(RESULTS_DIR / "impression_level_metrics.csv", index=False)

    by_user_pref = _aggregate_slices(metrics_df, "user_pref")
    by_clicked_category = _aggregate_slices(metrics_df, "clicked_category")

    by_user_pref.to_csv(RESULTS_DIR / "slice_metrics_by_user_pref.csv", index=False)
    by_clicked_category.to_csv(RESULTS_DIR / "slice_metrics_by_clicked_category.csv", index=False)

    hard_cases = metrics_df.sort_values(["is_miss", "click_rank"], ascending=[False, False]).head(30)
    hard_cases.to_csv(RESULTS_DIR / "error_cases_top_rank_misses.csv", index=False)

    _make_plots(by_user_pref, by_clicked_category)

    summary = {
        "k": k,
        "rows_impression_level": int(len(metrics_df)),
        "generated_files": [
            "impression_level_metrics.csv",
            "slice_metrics_by_user_pref.csv",
            "slice_metrics_by_clicked_category.csv",
            "error_cases_top_rank_misses.csv",
            "plots/ndcg_by_user_pref.png",
            "plots/hit_by_clicked_category.png",
            "plots/miss_rate_by_user_pref.png",
        ],
    }
    with open(RESULTS_DIR / "slice_error_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[analyze_slices] Slice/error outputs generated in experiments/results/")


if __name__ == "__main__":
    analyze_slices(k=10)
