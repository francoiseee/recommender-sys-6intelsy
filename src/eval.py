from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.data_pipeline import BanditEvent
from src.rl_agent import LinUCBAgent


def evaluate_bandit(agent: LinUCBAgent, events: List[BanditEvent]) -> Dict[str, float]:
	if not events:
		return {"events": 0.0, "ctr_at_1": 0.0, "avg_reward": 0.0, "avg_regret": 0.0}

	hits = 0.0
	rewards = []
	regrets = []

	for event in events:
		choice = agent.select_arm(event.arm_features)
		chosen_reward = float(event.rewards[choice])
		optimal_reward = float(np.max(event.rewards))

		hits += chosen_reward
		rewards.append(chosen_reward)
		regrets.append(optimal_reward - chosen_reward)

	return {
		"events": float(len(events)),
		"ctr_at_1": float(hits / len(events)),
		"avg_reward": float(np.mean(rewards)),
		"avg_regret": float(np.mean(regrets)),
	}
