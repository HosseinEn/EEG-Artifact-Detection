from torch.utils.data import Dataset, random_split
from feature_extraction import extract_features
from pathlib import Path
import numpy as np
from typing import Tuple
import argparse


class EEGDataset(Dataset):
    def __init__(self, data_dir: Path):
        X = np.load(data_dir / "X.npy")
        y = np.load(data_dir / "Y.npy")

        features = extract_features(X)
        self.features = features
        self.labels = y.flatten()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

#     @staticmethod
#     def load_data_splits(data_path, config):
#
#         train_dataset = EEGDataset(data_path / "train_val", config=config)
#
#         # Split the training dataset into training and validation datasets
#         train_size = int(config.train_ratio * len(train_dataset))
#         val_size = len(train_dataset) - train_size
#         train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
#
#         # Load all test datasets into a dictionary
#         test_datasets = {}
#         test_dir = data_path / "test"
#         for snr_dir in test_dir.iterdir():
#             if snr_dir.is_dir():
#                 snr_value = snr_dir.name.split('_')[-1]
#                 test_datasets[snr_value] = EEGDataset(snr_dir, config=config)
#
#         return train_dataset, val_dataset, test_datasets
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process some integers.")
#
#     # Add arguments
#     parser.add_argument('--datapath', type=str, default='./data', help='Path to the data folder.')
#     parser.add_argument('--test_ratio', type=float, default=0.25,
#                         help='Proportion of the dataset to include in the test split.')
#     parser.add_argument('--val_ratio', type=float, default=0.2,
#                         help='Proportion of the dataset to include in the validation split.')
#     parser.add_argument('--train_ratio', type=float, default=0.6)
#
#     args = parser.parse_args()
#
#     config = argparse.Namespace(
#         datapath=args.datapath,
#         train_ratio=args.train_ratio,
#         test_ratio=args.test_ratio,
#         val_ratio=args.val_ratio,
#     )
#
#     train_dataset, val_dataset, test_datasets = load_data_splits(Path(args.datapath), config)
#
#     print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")