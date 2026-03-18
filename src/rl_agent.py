from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LinUCBConfig:
	context_dim: int
	alpha: float = 1.0
	l2_reg: float = 1.0


class LinUCBAgent:
	"""Disjoint-arm LinUCB implemented as a shared linear model over arm contexts."""

	def __init__(self, config: LinUCBConfig) -> None:
		self.config = config
		d = config.context_dim
		if config.l2_reg <= 0:
			raise ValueError("l2_reg must be > 0 for a stable inverse")

		self.A = np.eye(d, dtype=np.float64) * config.l2_reg
		self.A_inv = np.eye(d, dtype=np.float64) / config.l2_reg
		self.b = np.zeros(d, dtype=np.float64)
		self.theta = np.zeros(d, dtype=np.float64)

	def _theta(self) -> np.ndarray:
		return self.theta

	def state_dict(self) -> dict:
		return {
			"context_dim": self.config.context_dim,
			"alpha": self.config.alpha,
			"l2_reg": self.config.l2_reg,
			"A": self.A,
			"A_inv": self.A_inv,
			"b": self.b,
			"theta": self.theta,
		}

	def load_state_dict(self, state: dict) -> None:
		self.A = state["A"].astype(np.float64)
		self.A_inv = state["A_inv"].astype(np.float64)
		self.b = state["b"].astype(np.float64)
		self.theta = state["theta"].astype(np.float64)

	def predict(self, arm_contexts: np.ndarray) -> np.ndarray:
		"""Return UCB scores for each arm context in shape [n_arms, context_dim]."""
		if arm_contexts.ndim != 2:
			raise ValueError("arm_contexts must be a 2D matrix")

		theta = self._theta()

		mean = arm_contexts @ theta
		# x^T A^-1 x for each row x.
		uncertainty = np.sqrt(np.einsum("ij,jk,ik->i", arm_contexts, self.A_inv, arm_contexts))
		return mean + self.config.alpha * uncertainty

	def select_arm(self, arm_contexts: np.ndarray) -> int:
		scores = self.predict(arm_contexts)
		return int(np.argmax(scores))

	def update(self, chosen_context: np.ndarray, reward: float) -> None:
		x = chosen_context.astype(np.float64)
		# Rank-1 inverse update: (A + xx^T)^-1 via Sherman-Morrison.
		A_inv_x = self.A_inv @ x
		denom = 1.0 + float(x @ A_inv_x)
		if denom <= 1e-12:
			denom = 1e-12
		self.A_inv -= np.outer(A_inv_x, A_inv_x) / denom

		self.A += np.outer(x, x)
		self.b += reward * x
		self.theta = self.A_inv @ self.b

	def learn_from_logged_sample(
		self,
		arm_contexts: np.ndarray,
		rewards: np.ndarray,
		forced_arm: Optional[int] = None,
	) -> float:
		"""
		Choose an arm (or use forced_arm), update the model, and return observed reward.
		"""
		arm_index = forced_arm if forced_arm is not None else self.select_arm(arm_contexts)
		reward = float(rewards[arm_index])
		self.update(arm_contexts[arm_index], reward)
		return reward
