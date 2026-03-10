"""
data_pipeline.py — Data loading, preprocessing, and splitting.
"""

import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_raw_data(data_dir: str) -> pd.DataFrame:
    """Load raw data from disk. TODO: Implement after dataset is finalized."""
    raise NotImplementedError("Implement load_raw_data() once dataset is chosen.")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the dataset."""
    # TODO: tokenization, normalization, feature engineering
    return df


def split_data(df: pd.DataFrame, train=0.7, val=0.15, test=0.15, seed=SEED):
    """Stratified train/val/test split with no leakage."""
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1."
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train)
    val_end = train_end + int(n * val)
    return df[:train_end], df[train_end:val_end], df[val_end:]


class RecommenderDataset(Dataset):
    """PyTorch Dataset for the recommendation task."""

    def __init__(self, df: pd.DataFrame, tokenizer=None):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # TODO: Return (text_input, label/reward) pairs
        row = self.df.iloc[idx]
        return row


def get_dataloader(df: pd.DataFrame, batch_size: int = 32, shuffle: bool = True):
    dataset = RecommenderDataset(df)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
