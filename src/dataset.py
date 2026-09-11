'''
# Data & Tokenizer
-We use a character-level tokenizer (simplest, no dependencies) since the assignment 
-targets small datasets.

'''

import os
import torch
import requests
from torch.utils.data import Dataset

class CharTokenizer:
    """
    Minimal character-level tokenizer.
    - Maps every unique character to an integer id and vice versa.
    """
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}   # string -> int
        self.itos = {i: ch for ch, i in self.stoi.items()}   # int -> string

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def download_dataset(cfg):
    """Download the text dataset if not already present."""
    os.makedirs(os.path.dirname(cfg.data_path), exist_ok=True)
    if not os.path.exists(cfg.data_path):
        print(f"Downloading {cfg.data_url} ...")
        r = requests.get(cfg.data_url)
        with open(cfg.data_path, "w", encoding="utf-8") as f:
            f.write(r.text)
    with open(cfg.data_path, "r", encoding="utf-8") as f:
        return f.read()


class LMDataset(Dataset):
    """
    Language-modeling dataset.
    Each item returns (x, y) where:
      - x = tokens [i, i+1, ..., i+block_size-1]
      - y = tokens [i+1, i+2, ..., i+block_size]   (shifted by 1 for next-token prediction)
    """
    def __init__(self, token_ids, block_size):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def build_datasets(cfg):
    """Download text, build tokenizer, split into train/val, return Datasets + tokenizer."""
    text = download_dataset(cfg)
    tokenizer = CharTokenizer(text)

    # Encode entire corpus into a list of integers
    ids = tokenizer.encode(text)

    # 90 / 10 train-validation split
    n = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]

    train_ds = LMDataset(train_ids, cfg.block_size)
    val_ds   = LMDataset(val_ids,   cfg.block_size)
    return train_ds, val_ds, tokenizer