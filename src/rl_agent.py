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
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    out = np.convolve(values, kernel, mode="same")
    return out


def _save_single_seed_artifacts(agent_type: str, rewards: list[float], cumulative: list[float], out_dir: Path, window: int):
    rewards_arr = np.array(rewards, dtype=float)
    cumulative_arr = np.array(cumulative, dtype=float)
    steps = np.arange(1, len(rewards_arr) + 1)
    moving_avg = _moving_average(rewards_arr, window=window)

    # Per-step CSV for plotting/reporting.
    curve_df = pd.DataFrame(
        {
            "step": steps,
            "reward": rewards_arr,
            "cumulative_reward": cumulative_arr,
            "moving_avg_reward": moving_avg,
        }
    )
    curve_df.to_csv(out_dir / f"bandit_{agent_type}_learning_curve.csv", index=False)

    # Existing JSON output retained for compatibility.
    with open(out_dir / f"bandit_{agent_type}_rewards.json", "w", encoding="utf-8") as f:
        json.dump({"rewards": rewards, "cumulative": cumulative}, f)

    # Learning curve plot.
    plt.figure(figsize=(10, 5))
    plt.plot(steps, cumulative_arr, label="cumulative reward")
    plt.xlabel("step")
    plt.ylabel("cumulative reward")
    plt.title(f"Bandit Learning Curve ({agent_type})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"bandit_{agent_type}_learning_curve.png", dpi=160)
    plt.close()

    # Reward + moving average plot.
    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards_arr, alpha=0.3, label="instant reward")
    plt.plot(steps, moving_avg, linewidth=2, label=f"moving avg (window={window})")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.title(f"Bandit Reward Trend ({agent_type})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"bandit_{agent_type}_reward_trend.png", dpi=160)
    plt.close()

    summary = {
        "agent": agent_type,
        "num_steps": int(len(rewards_arr)),
        "total_reward": float(cumulative_arr[-1]) if len(cumulative_arr) else 0.0,
        "mean_reward": float(rewards_arr.mean()) if len(rewards_arr) else 0.0,
        "std_reward": float(rewards_arr.std()) if len(rewards_arr) else 0.0,
        "moving_avg_window": int(window),
        "generated_files": [
            f"bandit_{agent_type}_rewards.json",
            f"bandit_{agent_type}_learning_curve.csv",
            f"bandit_{agent_type}_learning_curve.png",
            f"bandit_{agent_type}_reward_trend.png",
        ],
    }
    with open(out_dir / f"bandit_{agent_type}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _save_multiseed_artifacts(agent_type: str, all_rewards: list[list[float]], out_dir: Path):
    if not all_rewards:
        return

    rewards_mat = np.array(all_rewards, dtype=float)  # [num_seeds, n_steps]
    mean_reward = rewards_mat.mean(axis=0)
    std_reward = rewards_mat.std(axis=0)
    mean_cumulative = np.cumsum(mean_reward)
    steps = np.arange(1, rewards_mat.shape[1] + 1)

    multi_df = pd.DataFrame(
        {
            "step": steps,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_cumulative_reward": mean_cumulative,
        }
    )
    multi_df.to_csv(out_dir / f"bandit_{agent_type}_multiseed_curve.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, mean_reward, label="mean reward")
    plt.fill_between(steps, mean_reward - std_reward, mean_reward + std_reward, alpha=0.2, label="±1 std")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.title(f"Bandit Reward Across Seeds ({agent_type})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"bandit_{agent_type}_multiseed_reward.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(steps, mean_cumulative, label="mean cumulative reward")
    plt.xlabel("step")
    plt.ylabel("mean cumulative reward")
    plt.title(f"Bandit Mean Cumulative Reward Across Seeds ({agent_type})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"bandit_{agent_type}_multiseed_cumulative.png", dpi=160)
    plt.close()

    summary = {
        "agent": agent_type,
        "num_seeds": int(rewards_mat.shape[0]),
        "num_steps": int(rewards_mat.shape[1]),
        "final_mean_cumulative_reward": float(mean_cumulative[-1]) if len(mean_cumulative) else 0.0,
        "final_mean_reward": float(mean_reward[-1]) if len(mean_reward) else 0.0,
        "final_std_reward": float(std_reward[-1]) if len(std_reward) else 0.0,
        "generated_files": [
            f"bandit_{agent_type}_multiseed_curve.csv",
            f"bandit_{agent_type}_multiseed_reward.png",
            f"bandit_{agent_type}_multiseed_cumulative.png",
        ],
    }
    with open(out_dir / f"bandit_{agent_type}_multiseed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


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
    num_seeds = int(cfg.get("num_seeds", 3))
    moving_avg_window = int(cfg.get("moving_avg_window", 25))
    agent_type = cfg.get("agent", "epsilon_greedy")

    print(f"[rl_agent] Running {agent_type} bandit | arms={n_arms} | steps={n_steps} | seeds={num_seeds}")

    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rewards = []
    all_cumulative = []
    for run_seed in [seed + i for i in range(num_seeds)]:
        if agent_type == "linucb":
            agent = LinUCBBandit(
                n_arms=n_arms,
                context_dim=context_dim,
                alpha=float(cfg.get("alpha", 0.5)),
                seed=run_seed,
            )
        else:
            agent = EpsilonGreedyBandit(
                n_arms=n_arms,
                epsilon=float(cfg.get("epsilon", 0.1)),
                seed=run_seed,
            )

        rewards, cumulative = simulate_bandit(
            agent,
            n_steps=n_steps,
            context_dim=context_dim,
            seed=run_seed,
        )
        all_rewards.append(rewards)
        all_cumulative.append(cumulative)

    # Save first-seed artifacts for backward compatibility and reporting simplicity.
    _save_single_seed_artifacts(
        agent_type=agent_type,
        rewards=all_rewards[0],
        cumulative=all_cumulative[0],
        out_dir=out_dir,
        window=moving_avg_window,
    )
    _save_multiseed_artifacts(agent_type=agent_type, all_rewards=all_rewards, out_dir=out_dir)

    print(f"[rl_agent] Seed0 total reward: {all_cumulative[0][-1]:.2f} / {n_steps}")
    print(f"[rl_agent] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
