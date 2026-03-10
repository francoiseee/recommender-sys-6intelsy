"""
rl_agent.py — Contextual Bandit for adaptive recommendation.
RL Requirement: Satisfies the RL component of the project.

Implements:
  - EpsilonGreedyBandit: simple ε-greedy exploration
  - LinUCBBandit: linear upper confidence bound bandit
"""

import numpy as np
import argparse
import yaml
import json
import os


# ---------------------------------------------------------------------------
# Environment Spec
# ---------------------------------------------------------------------------
# State:    User context vector (e.g., from text embedding or user history)
# Actions:  Item indices to recommend
# Reward:   Click / engagement signal (1 = clicked, 0 = not clicked)
# Episode:  One recommendation round per user session
# ---------------------------------------------------------------------------


class EpsilonGreedyBandit:
    """
    ε-Greedy Contextual Bandit.
    Explores randomly with probability epsilon, otherwise exploits best known arm.
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: int = 42):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(n_arms)       # times each arm was pulled
        self.values = np.zeros(n_arms)       # estimated reward per arm

    def select_arm(self, context=None) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n  # incremental mean


class LinUCBBandit:
    """
    LinUCB Contextual Bandit.
    Uses a linear model per arm with Upper Confidence Bound exploration.
    Reference: Li et al. (2010) "A Contextual-Bandit Approach to Personalized News Article Recommendation"
    """

    def __init__(self, n_arms: int, context_dim: int, alpha: float = 0.5, seed: int = 42):
        self.n_arms = n_arms
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        d = context_dim
        self.A = [np.eye(d) for _ in range(n_arms)]        # d x d
        self.b = [np.zeros((d, 1)) for _ in range(n_arms)] # d x 1

    def select_arm(self, context: np.ndarray) -> int:
        x = context.reshape(-1, 1)
        ucb_scores = []
        for i in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            ucb = float(theta.T @ x + self.alpha * np.sqrt(x.T @ A_inv @ x))
            ucb_scores.append(ucb)
        return int(np.argmax(ucb_scores))

    def update(self, arm: int, context: np.ndarray, reward: float):
        x = context.reshape(-1, 1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x


def simulate_bandit(agent, n_steps: int = 1000, context_dim: int = 32, seed: int = 42):
    """
    Offline simulation of bandit policy.
    Returns cumulative rewards and per-step reward log.
    """
    rng = np.random.default_rng(seed)
    rewards = []
    cumulative = []
    total = 0.0

    for step in range(n_steps):
        context = rng.standard_normal(context_dim)
        arm = agent.select_arm(context)
        # Simulated reward: arm 0 has higher true reward to test learning
        true_probs = np.linspace(0.3, 0.7, agent.n_arms)
        reward = float(rng.random() < true_probs[arm])

        if hasattr(agent, 'A'):  # LinUCB
            agent.update(arm, context, reward)
        else:
            agent.update(arm, reward)

        total += reward
        rewards.append(reward)
        cumulative.append(total)

    return rewards, cumulative


def main():
    parser = argparse.ArgumentParser(description="Run RL Bandit simulation")
    parser.add_argument("--config", type=str, default="experiments/configs/bandit.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n_arms = cfg.get("n_arms", 10)
    n_steps = cfg.get("n_steps", 1000)
    context_dim = cfg.get("context_dim", 32)
    seed = cfg.get("seed", 42)
    agent_type = cfg.get("agent", "epsilon_greedy")

    print(f"[rl_agent] Running {agent_type} bandit | arms={n_arms} | steps={n_steps}")

    if agent_type == "linucb":
        agent = LinUCBBandit(n_arms=n_arms, context_dim=context_dim, seed=seed)
    else:
        agent = EpsilonGreedyBandit(n_arms=n_arms, epsilon=cfg.get("epsilon", 0.1), seed=seed)

    rewards, cumulative = simulate_bandit(agent, n_steps=n_steps, context_dim=context_dim, seed=seed)

    os.makedirs("experiments/results", exist_ok=True)
    out_path = f"experiments/results/bandit_{agent_type}_rewards.json"
    with open(out_path, "w") as f:
        json.dump({"rewards": rewards, "cumulative": cumulative}, f)

    print(f"[rl_agent] Total reward: {cumulative[-1]:.2f} / {n_steps}")
    print(f"[rl_agent] Results saved to {out_path}")


if __name__ == "__main__":
    main()
