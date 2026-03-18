from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.data_pipeline import build_news_feature_store, load_bandit_events
from src.eval import evaluate_bandit
from src.rl_agent import LinUCBAgent, LinUCBConfig


def _format_seconds(seconds: float) -> str:
	seconds = max(0, int(seconds))
	minutes, secs = divmod(seconds, 60)
	hours, minutes = divmod(minutes, 60)
	if hours > 0:
		return f"{hours:d}:{minutes:02d}:{secs:02d}"
	return f"{minutes:02d}:{secs:02d}"


def _progress_line(label: str, current: int, total: int, start_time: float) -> None:
	elapsed = time.perf_counter() - start_time
	percent = (100.0 * current) / float(max(1, total))
	rate = current / max(elapsed, 1e-9)
	remaining = max(0, total - current)
	eta = remaining / max(rate, 1e-9)
	print(
		f"\r{label}: {current}/{total} ({percent:5.1f}%) | elapsed {_format_seconds(elapsed)} | ETA {_format_seconds(eta)}",
		end="",
		flush=True,
	)


def _save_checkpoint(
	checkpoint_path: Path,
	agent: LinUCBAgent,
	step: int,
	total_steps: int,
	cumulative_reward: float,
	args: argparse.Namespace,
) -> None:
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	state = agent.state_dict()
	np_payload = {
		"A": state["A"],
		"A_inv": state["A_inv"],
		"b": state["b"],
		"theta": state["theta"],
	}
	np_path = checkpoint_path.with_suffix(".npz")
	json_path = checkpoint_path.with_suffix(".json")

	import numpy as np

	np.savez(np_path, **np_payload)
	metadata = {
		"step": step,
		"total_steps": total_steps,
		"cumulative_reward": cumulative_reward,
		"context_dim": state["context_dim"],
		"alpha": state["alpha"],
		"l2_reg": state["l2_reg"],
		"created_at_unix": time.time(),
		"train_args": vars(args),
	}
	json_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def _load_checkpoint(
	checkpoint_path: Path,
	agent: LinUCBAgent,
) -> tuple[int, float]:
	np_path = checkpoint_path.with_suffix(".npz")
	json_path = checkpoint_path.with_suffix(".json")
	if not np_path.exists() or not json_path.exists():
		raise FileNotFoundError(f"Checkpoint files not found for base path: {checkpoint_path}")

	import numpy as np

	np_data = np.load(np_path)
	metadata = json.loads(json_path.read_text(encoding="utf-8"))
	state = {
		"A": np_data["A"],
		"A_inv": np_data["A_inv"],
		"b": np_data["b"],
		"theta": np_data["theta"],
	}
	agent.load_state_dict(state)
	step = int(metadata.get("step", 0))
	cumulative_reward = float(metadata.get("cumulative_reward", 0.0))
	return step, cumulative_reward


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a contextual-bandit recommender on MIND-like data.")
	parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing TSV/VEC files.")
	parser.add_argument("--text-dim", type=int, default=256, help="Hashed text feature size.")
	parser.add_argument("--category-dim", type=int, default=32, help="Hashed category feature size per field.")
	parser.add_argument("--max-events", type=int, default=15000, help="Limit number of behavior events for faster runs.")
	parser.add_argument("--max-news", type=int, default=30000, help="Limit number of news rows for faster runs.")
	parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of events used for training.")
	parser.add_argument("--alpha", type=float, default=1.0, help="LinUCB exploration coefficient.")
	parser.add_argument("--l2", type=float, default=1.0, help="L2 regularization for LinUCB.")
	parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
	parser.add_argument(
		"--checkpoint-dir",
		type=Path,
		default=Path("experiments") / "checkpoints",
		help="Directory to store training checkpoints.",
	)
	parser.add_argument(
		"--checkpoint-every",
		type=int,
		default=1000,
		help="Save a checkpoint every N training events (0 disables periodic saves).",
	)
	parser.add_argument(
		"--resume-from",
		type=Path,
		default=None,
		help="Base checkpoint path to resume from (without .npz/.json extension).",
	)
	parser.add_argument("--run-name", type=str, default="linucb_run", help="Run name used for checkpoint files.")
	parser.add_argument(
		"--use-entity-embeddings",
		action="store_true",
		help="Include entity_embedding.vec features (slower but richer).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	news_path = args.data_dir / "news.tsv"
	behaviors_path = args.data_dir / "behaviors.tsv"
	entity_vec_path = args.data_dir / "entity_embedding.vec"

	if not news_path.exists() or not behaviors_path.exists():
		raise FileNotFoundError("Expected news.tsv and behaviors.tsv inside the provided data directory")

	selected_entity_path = None
	if args.use_entity_embeddings and entity_vec_path.exists():
		selected_entity_path = entity_vec_path

	show_progress = not args.no_progress
	total_start = time.perf_counter()
	if show_progress:
		print("Starting pipeline...")

	news_stage_start = time.perf_counter()
	news_features = build_news_feature_store(
		news_path=news_path,
		entity_embedding_path=selected_entity_path,
		text_dim=args.text_dim,
		category_dim=args.category_dim,
		max_news=args.max_news,
		show_progress=show_progress,
	)
	news_elapsed = time.perf_counter() - news_stage_start
	if not news_features:
		raise RuntimeError("No news features were produced. Check your news.tsv format.")

	events_stage_start = time.perf_counter()
	events = load_bandit_events(
		behaviors_path=behaviors_path,
		news_features=news_features,
		max_events=args.max_events,
		show_progress=show_progress,
	)
	events_elapsed = time.perf_counter() - events_stage_start
	if not events:
		raise RuntimeError("No valid behavior events were produced. Check your behaviors.tsv format.")

	split_idx = max(1, int(len(events) * args.train_ratio))
	train_events = events[:split_idx]
	val_events = events[split_idx:] if split_idx < len(events) else []

	context_dim = int(train_events[0].arm_features.shape[1])
	agent = LinUCBAgent(LinUCBConfig(context_dim=context_dim, alpha=args.alpha, l2_reg=args.l2))

	checkpoint_base = args.checkpoint_dir / args.run_name
	start_step = 0
	cumulative_reward = 0.0
	if args.resume_from is not None:
		start_step, cumulative_reward = _load_checkpoint(args.resume_from, agent)
		if show_progress:
			print(f"Resumed checkpoint: {args.resume_from} at step {start_step}")
		if start_step >= len(train_events):
			if show_progress:
				print("Checkpoint step is already at or beyond total train events; skipping training loop.")
			start_step = len(train_events)

	train_stage_start = time.perf_counter()
	for index, event in enumerate(train_events[start_step:], start=start_step + 1):
		cumulative_reward += agent.learn_from_logged_sample(event.arm_features, event.rewards)
		if show_progress and (index == 1 or index % 500 == 0 or index == len(train_events)):
			_progress_line("Training", index, len(train_events), train_stage_start)

		if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
			periodic_path = checkpoint_base.parent / f"{checkpoint_base.name}_step_{index}"
			_save_checkpoint(
				checkpoint_path=periodic_path,
				agent=agent,
				step=index,
				total_steps=len(train_events),
				cumulative_reward=cumulative_reward,
				args=args,
			)
			if show_progress:
				print(f"\nCheckpoint saved: {periodic_path}")

	final_checkpoint = checkpoint_base.parent / f"{checkpoint_base.name}_final"
	_save_checkpoint(
		checkpoint_path=final_checkpoint,
		agent=agent,
		step=len(train_events),
		total_steps=len(train_events),
		cumulative_reward=cumulative_reward,
		args=args,
	)

	train_elapsed = time.perf_counter() - train_stage_start

	if show_progress:
		print()

	print("=== Training Summary ===")
	print(f"news items: {len(news_features)}")
	print(f"events used: {len(events)}")
	print(f"train events: {len(train_events)}")
	print(f"val events: {len(val_events)}")
	print(f"train cumulative reward: {cumulative_reward:.2f}")
	print(f"train avg reward: {cumulative_reward / max(1, len(train_events)):.4f}")
	print(f"final checkpoint base path: {final_checkpoint}")
	print(f"stage news feature build elapsed: {_format_seconds(news_elapsed)}")
	print(f"stage event loading elapsed: {_format_seconds(events_elapsed)}")
	print(f"stage training elapsed: {_format_seconds(train_elapsed)}")
	print(f"total elapsed: {_format_seconds(time.perf_counter() - total_start)}")

	if val_events:
		metrics = evaluate_bandit(agent, val_events)
		print("=== Validation Metrics ===")
		for key, value in metrics.items():
			print(f"{key}: {value:.4f}")
	else:
		print("No validation split available (increase max-events or adjust train-ratio).")


if __name__ == "__main__":
	main()
