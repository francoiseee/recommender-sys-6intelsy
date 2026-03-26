"""
ranker.py — Embedding-based item ranker (Core DL model).
Uses TextCNN or a transformer encoder to embed items, then ranks by similarity.
"""

import torch
import torch.nn as nn
from src.models.text_cnn import TextCNN


class EmbeddingRanker(nn.Module):
    """
    Embedding-based ranker for personalized recommendation.
    Core DL model: encodes user context and item text, then scores by dot product.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, output_dim: int = 256):
        super().__init__()
        self.item_encoder = TextCNN(vocab_size, embed_dim=embed_dim, output_dim=output_dim)
        self.user_encoder = nn.Linear(output_dim, output_dim)  # placeholder for user features

    def encode_items(self, item_texts):
        """Encode a batch of item texts into embedding vectors."""
        return self.item_encoder(item_texts)  # (B, output_dim)

    def forward(self, user_context, item_texts):
        """
        Score items given user context.
        Returns relevance scores (higher = more relevant).
        """
        item_embs = self.encode_items(item_texts)   # (B, D)
        user_emb = self.user_encoder(user_context)  # (B, D)
        scores = (user_emb * item_embs).sum(dim=-1)  # dot product
        return scores
