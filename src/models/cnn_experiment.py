from __future__ import annotations

"""Prototype CNN experiment scaffold for text-based recommendation scoring.

This module is intentionally lightweight so the repository has a concrete CNN
component scaffold that can be expanded into full experiments.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass
class CNNExperimentConfig:
    vocab_size: int = 50000
    embed_dim: int = 128
    num_filters: int = 64
    kernel_size: int = 3
    hidden_dim: int = 64
    dropout: float = 0.2


def torch_available() -> bool:
    try:
        import_module("torch")

        return True
    except Exception:
        return False


def build_torch_model(config: CNNExperimentConfig):
    try:
        torch = import_module("torch")
        nn = import_module("torch.nn")
    except Exception as exc:
        raise RuntimeError("PyTorch is not installed. Install torch to run the CNN experiment.") from exc

    class TextCNNRanker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
            self.conv = nn.Conv1d(
                in_channels=config.embed_dim,
                out_channels=config.num_filters,
                kernel_size=config.kernel_size,
                padding=config.kernel_size // 2,
            )
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.head = nn.Sequential(
                nn.Linear(config.num_filters, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )

        def forward(self, token_ids: Any) -> Any:
            # token_ids: [batch, seq_len]
            x = self.embedding(token_ids)  # [batch, seq_len, embed_dim]
            x = x.transpose(1, 2)  # [batch, embed_dim, seq_len]
            x = torch.relu(self.conv(x))
            x = self.pool(x).squeeze(-1)
            return self.head(x).squeeze(-1)

    return TextCNNRanker()


def smoke_test() -> str:
    if not torch_available():
        return "PyTorch not installed; CNN scaffold created but not executed."

    torch: Any = import_module("torch")

    cfg = CNNExperimentConfig()
    model = build_torch_model(cfg)
    tokens = torch.randint(low=0, high=cfg.vocab_size, size=(4, 20))
    with torch.no_grad():
        out = model(tokens)
    return f"CNN scaffold smoke test ok. Output shape: {tuple(out.shape)}"


if __name__ == "__main__":
    print(smoke_test())
