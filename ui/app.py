"""Streamlit dashboard for experiment artifacts and defense assets."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
DOCS = ROOT / "docs"


def _read_csv(path: Path) -> pd.DataFrame:
	if not path.exists():
		return pd.DataFrame()
	return pd.read_csv(path)


def _show_table(title: str, path: Path):
	st.subheader(title)
	df = _read_csv(path)
	if df.empty:
		st.info(f"Missing or empty file: {path.relative_to(ROOT)}")
	else:
		st.dataframe(df, use_container_width=True)


def _show_json_table(title: str, path: Path):
	st.subheader(title)
	if not path.exists():
		st.info(f"Missing file: {path.relative_to(ROOT)}")
		return
	with open(path, "r", encoding="utf-8") as f:
		payload = json.load(f)
	df = pd.DataFrame(payload)
	if df.empty:
		st.info(f"No rows in: {path.relative_to(ROOT)}")
	else:
		st.dataframe(df, use_container_width=True)


def _show_image(title: str, path: Path):
	st.subheader(title)
	if path.exists():
		st.image(str(path), use_container_width=True)
	else:
		st.info(f"Missing image: {path.relative_to(ROOT)}")


st.set_page_config(page_title="6INTELSY Recommender Dashboard", layout="wide")
st.title("6INTELSY Final Dashboard")
st.caption("Metrics, slice/error analysis, and defense assets")

st.markdown("### Final Dataset")
st.success("Using final dataset: synthetic_newsrec_v1 (from data/get_data.py)")

left, right = st.columns(2)
with left:
	_show_table("Overall Metrics", RESULTS / "metrics_summary.csv")
	_show_json_table("Ablation Summary", RESULTS / "ablation_summary.json")
with right:
	_show_table("Slice by User Preference", RESULTS / "slice_metrics_by_user_pref.csv")
	_show_table("Slice by Clicked Category", RESULTS / "slice_metrics_by_clicked_category.csv")

st.markdown("### Error Analysis")
_show_table("Top Rank Misses / Hard Cases", RESULTS / "error_cases_top_rank_misses.csv")

st.markdown("### Slice Plots")
plot_col1, plot_col2, plot_col3 = st.columns(3)
with plot_col1:
	_show_image("nDCG@10 by User Preference", RESULTS / "plots" / "ndcg_by_user_pref.png")
with plot_col2:
	_show_image("Hit@10 by Clicked Category", RESULTS / "plots" / "hit_by_clicked_category.png")
with plot_col3:
	_show_image("Miss Rate@10 by User Preference", RESULTS / "plots" / "miss_rate_by_user_pref.png")

st.markdown("### Run Commands")
st.code(
	"\n".join(
		[
			"python data/get_data.py --force",
			"python src/train.py --config experiments/configs/baseline.yaml",
			"python src/train.py --config experiments/configs/cnn_ranker.yaml",
			"python src/eval.py --all",
			"python src/run_ablations.py --config experiments/configs/cnn_ranker.yaml",
			"python src/analyze_slices.py",
			"python src/generate_assets.py",
			"streamlit run ui/app.py",
		]
	),
	language="bash",
)
