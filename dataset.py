from torch.utils.data import Dataset, random_split
from feature_extraction import extract_features
from pathlib import Path
import termcolor
import numpy as np
from typing import Tuple
import argparse


class EEGDataset(Dataset):
    def __init__(self, data_dir: Path):
        try:
            X = np.load(data_dir / "X.npy")
            y = np.load(data_dir / "Y.npy")
        except FileNotFoundError:
            print(termcolor.colored("Data files not found. Please ensure the data files are in the correct directory.", "red"))
            exit(1)

        features = extract_features(X)
        self.features = features
        self.labels = y.flatten()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
