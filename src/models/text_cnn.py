"""
text_cnn.py — Text-CNN for item/news text feature extraction.
CNN Requirement: Satisfies the CNN component of the project.
Reference: Kim (2014) "Convolutional Neural Networks for Sentence Classification"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    Text-CNN encoder for extracting features from item/news text.
    Used as the CNN component of the recommendation pipeline.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 128,
        filter_sizes: list = [2, 3, 4],
        output_dim: int = 256,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x).unsqueeze(1)  # (B, 1, seq_len, embed_dim)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(embedded)).squeeze(3)   # (B, num_filters, seq_len - fs + 1)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)  # (B, num_filters)
            pooled.append(c)
        cat = self.dropout(torch.cat(pooled, dim=1))  # (B, num_filters * len(filter_sizes))
        return self.fc(cat)  # (B, output_dim)


class TextCNNClassifier(nn.Module):
    """Text-CNN with classification head — for NLP auxiliary task."""

    def __init__(self, vocab_size, num_classes, embed_dim=128, **kwargs):
        super().__init__()
        self.encoder = TextCNN(vocab_size, embed_dim=embed_dim, **kwargs)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)
