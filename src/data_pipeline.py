from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _format_seconds(seconds: float) -> str:
	seconds = max(0, int(seconds))
	minutes, secs = divmod(seconds, 60)
	hours, minutes = divmod(minutes, 60)
	if hours > 0:
		return f"{hours:d}:{minutes:02d}:{secs:02d}"
	return f"{minutes:02d}:{secs:02d}"


def _progress_line(label: str, current: int, total: Optional[int], start_time: float) -> None:
	elapsed = time.perf_counter() - start_time
	if total is not None and total > 0:
		percent = (100.0 * current) / float(total)
		rate = current / max(elapsed, 1e-9)
		remaining = max(0, total - current)
		eta = remaining / max(rate, 1e-9)
		print(
			f"\r{label}: {current}/{total} ({percent:5.1f}%) | elapsed {_format_seconds(elapsed)} | ETA {_format_seconds(eta)}",
			end="",
			flush=True,
		)
	else:
		print(f"\r{label}: {current} | elapsed {_format_seconds(elapsed)} | ETA --:--", end="", flush=True)


@dataclass
class BanditEvent:
	impression_id: str
	user_id: str
	arm_news_ids: List[str]
	arm_features: np.ndarray
	rewards: np.ndarray


def _tokenize(text: str) -> List[str]:
	return TOKEN_PATTERN.findall((text or "").lower())


def _hashed_bow(text: str, dim: int) -> np.ndarray:
	vec = np.zeros(dim, dtype=np.float32)
	tokens = _tokenize(text)
	if not tokens:
		return vec

	for token in tokens:
		vec[hash(token) % dim] += 1.0

	vec /= float(len(tokens))
	return vec


def _hashed_one_hot(value: str, dim: int) -> np.ndarray:
	vec = np.zeros(dim, dtype=np.float32)
	if value:
		vec[hash(value.lower()) % dim] = 1.0
	return vec


def _safe_json_loads(value: str) -> List[dict]:
	if not value:
		return []
	try:
		loaded = json.loads(value)
		if isinstance(loaded, list):
			return loaded
		return []
	except json.JSONDecodeError:
		return []


def _extract_wikidata_ids(entity_blob: str) -> List[str]:
	entities = _safe_json_loads(entity_blob)
	ids: List[str] = []
	for entry in entities:
		wikidata_id = entry.get("WikidataId")
		if isinstance(wikidata_id, str) and wikidata_id:
			ids.append(wikidata_id)
	return ids


def load_embedding_file(vec_path: Optional[Path], max_rows: Optional[int] = None) -> Dict[str, np.ndarray]:
	if vec_path is None or not vec_path.exists():
		return {}

	embeddings: Dict[str, np.ndarray] = {}
	with vec_path.open("r", encoding="utf-8") as handle:
		for index, line in enumerate(handle):
			if max_rows is not None and index >= max_rows:
				break

			parts = line.rstrip("\n").split("\t")
			if len(parts) < 2:
				continue

			key = parts[0]
			try:
				values = np.asarray([float(v) for v in parts[1:]], dtype=np.float32)
			except ValueError:
				continue

			embeddings[key] = values

	return embeddings


def _mean_entity_embedding(entity_ids: Iterable[str], entity_embeddings: Dict[str, np.ndarray], entity_dim: int) -> np.ndarray:
	if entity_dim <= 0:
		return np.zeros(0, dtype=np.float32)

	vectors: List[np.ndarray] = []
	for entity_id in entity_ids:
		vec = entity_embeddings.get(entity_id)
		if vec is not None:
			vectors.append(vec)

	if not vectors:
		return np.zeros(entity_dim, dtype=np.float32)

	return np.mean(vectors, axis=0).astype(np.float32)


def build_news_feature_store(
	news_path: Path,
	entity_embedding_path: Optional[Path] = None,
	text_dim: int = 256,
	category_dim: int = 32,
	max_news: Optional[int] = None,
	show_progress: bool = False,
) -> Dict[str, np.ndarray]:
	entity_embeddings = load_embedding_file(entity_embedding_path)
	entity_dim = 0
	if entity_embeddings:
		first_key = next(iter(entity_embeddings))
		entity_dim = int(entity_embeddings[first_key].shape[0])

	feature_store: Dict[str, np.ndarray] = {}
	news_start_time = time.perf_counter()
	processed_rows = 0

	with news_path.open("r", encoding="utf-8") as handle:
		reader = csv.reader(handle, delimiter="\t")
		for index, row in enumerate(reader):
			if max_news is not None and index >= max_news:
				break

			processed_rows += 1

			if show_progress and (index == 0 or (index + 1) % 1000 == 0):
				_progress_line("Building news features", processed_rows, max_news, news_start_time)

			if len(row) < 8:
				continue

			news_id, category, subcategory, title, abstract, _, title_entities, abstract_entities = row[:8]

			text_vec = _hashed_bow(f"{title} {abstract}", text_dim)
			category_vec = np.concatenate(
				[
					_hashed_one_hot(category, category_dim),
					_hashed_one_hot(subcategory, category_dim),
				]
			).astype(np.float32)

			entity_ids = _extract_wikidata_ids(title_entities) + _extract_wikidata_ids(abstract_entities)
			entity_vec = _mean_entity_embedding(entity_ids, entity_embeddings, entity_dim)

			news_feature = np.concatenate([text_vec, category_vec, entity_vec]).astype(np.float32)
			feature_store[news_id] = news_feature

	if show_progress:
		_progress_line("Building news features", processed_rows, max_news, news_start_time)
		print()

	return feature_store


def _average_news_vectors(news_ids: Sequence[str], news_features: Dict[str, np.ndarray], feature_dim: int) -> np.ndarray:
	vectors = [news_features[nid] for nid in news_ids if nid in news_features]
	if not vectors:
		return np.zeros(feature_dim, dtype=np.float32)
	return np.mean(vectors, axis=0).astype(np.float32)


def load_bandit_events(
	behaviors_path: Path,
	news_features: Dict[str, np.ndarray],
	max_events: Optional[int] = None,
	show_progress: bool = False,
) -> List[BanditEvent]:
	if not news_features:
		return []

	one_news_dim = int(next(iter(news_features.values())).shape[0])
	events: List[BanditEvent] = []
	events_start_time = time.perf_counter()

	with behaviors_path.open("r", encoding="utf-8") as handle:
		reader = csv.reader(handle, delimiter="\t")
		for row in reader:
			if len(row) < 5:
				continue

			impression_id, user_id, _, history, impressions = row[:5]
			history_ids = history.split() if history else []
			user_profile = _average_news_vectors(history_ids, news_features, one_news_dim)

			arm_news_ids: List[str] = []
			arm_features: List[np.ndarray] = []
			rewards: List[int] = []

			for token in impressions.split():
				if "-" not in token:
					continue
				candidate_id, label_str = token.rsplit("-", 1)
				if candidate_id not in news_features:
					continue

				candidate_vec = news_features[candidate_id]
				# Candidate + user profile + interaction term.
				x = np.concatenate([candidate_vec, user_profile, candidate_vec - user_profile]).astype(np.float32)

				arm_news_ids.append(candidate_id)
				arm_features.append(x)
				rewards.append(1 if label_str == "1" else 0)

			if not arm_features:
				continue

			events.append(
				BanditEvent(
					impression_id=impression_id,
					user_id=user_id,
					arm_news_ids=arm_news_ids,
					arm_features=np.vstack(arm_features).astype(np.float32),
					rewards=np.asarray(rewards, dtype=np.float32),
				)
			)

			if show_progress and (len(events) == 1 or len(events) % 500 == 0):
				_progress_line("Loading bandit events", len(events), max_events, events_start_time)

			if max_events is not None and len(events) >= max_events:
				break

	if show_progress:
		_progress_line("Loading bandit events", len(events), max_events, events_start_time)
		print()

	return events
