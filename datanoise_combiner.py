from pathlib import Path
from random import seed
import numpy as np
from numpy import array, genfromtxt
from scipy.stats import zscore
from torch.utils.data import Dataset
import scipy.io as sio
import os
import argparse
from utils import *

seed(2408)
np.random.seed(1305)

class DataNoiseCombiner:
    def __init__(self, config):
        self.config = config
        self.data_clean = self.load_samples(os.path.join(config.datapath, "filtered80Hz_EEG_all_epochs.mat"))
        self.data_eog = self.load_samples(os.path.join(config.datapath, "filtered80Hz_EOG_all_epochs.mat"))
        self.data_eog = np.repeat(self.data_eog, np.ceil(len(self.data_clean) / len(self.data_eog)), axis=0)[: len(self.data_clean), :]
        self.data_emg = self.load_samples(os.path.join(config.datapath, "filtered80Hz_EMG_all_epochs.mat"))
        self.data_emg = np.repeat(self.data_emg, np.ceil(len(self.data_clean) / len(self.data_emg)), axis=0)[: len(self.data_clean), :]
        self.white_noise = self.load_samples(os.path.join(config.datapath, "white_noise.mat"))
        self.clean_indices = self.shuffle_indices(len(self.data_clean))
        self.eog_indices = self.shuffle_indices(len(self.data_eog))
        self.emg_indices = self.shuffle_indices(len(self.data_emg))
        self.white_noise_indices = self.shuffle_indices(len(self.white_noise))
        self.process_and_save_data()

    def load_samples(self, path):
        path = Path(path)
        data = sio.loadmat(path) if path.suffix == ".mat" else (
            genfromtxt(path, delimiter=",") if path.suffix == ".csv" else (
                np.load(path) if path.suffix == ".npy" else None))
        if data is None:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        X = next(value for value in data.values() if isinstance(value, np.ndarray)) if path.suffix == ".mat" else data
        # X = self.augment_data(X, len(X)*2)
        return X

    def augment_data(self, data, num_samples):
        data = np.repeat(data, 4, axis=0)
        return data[:num_samples]

    @staticmethod
    def shuffle_indices(length):
        indices = np.arange(length)
        np.random.shuffle(indices)
        return indices

    def split_indices(self, indices, test_size, val_size):
        test_size, val_size = int(test_size * len(indices)), int(val_size * len(indices))
        return indices[:test_size], indices[test_size:test_size + val_size], indices[test_size + val_size:]

    def save_data(self, X, y, data_type, snr_type=None):
        directory = Path(self.config.datapath) / data_type / (snr_type or "")
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "X.npy", zscore(X, axis=1))
        np.save(directory / "Y.npy", y)


    def process_and_save_data(self):
        clean_test_indices, clean_val_indices, clean_training_indices = self.split_indices(self.clean_indices, self.config.test_size, self.config.val_size)
        eog_test_indices, eog_val_indices, eog_training_indices = self.split_indices(self.eog_indices, self.config.test_size, self.config.val_size)
        emg_test_indices, emg_val_indices, emg_training_indices = self.split_indices(self.emg_indices, self.config.test_size, self.config.val_size)
        wn_test_indices, wn_val_indices, wn_training_indices = self.split_indices(self.white_noise_indices, self.config.test_size, self.config.val_size)
        train_noises = {
            'White_noise': self.white_noise[wn_training_indices],
            'EOG': self.data_eog[eog_training_indices],
            'EMG': self.data_emg[emg_training_indices]
        }
        X_train, y_train = combine_data(clean_data=self.data_clean[clean_training_indices], noises=train_noises)
        self.save_data(X_train, y_train, "train")
        val_noises = {
            'White_noise': self.white_noise[wn_val_indices],
            'EOG': self.data_eog[eog_val_indices],
            'EMG': self.data_emg[emg_val_indices]
        }
        X_val, y_val = combine_data(clean_data=self.data_clean[clean_val_indices], noises=val_noises)
        self.save_data(X_val, y_val, "val")
        test_noises = {
            'White_noise': self.white_noise[wn_test_indices],
            'EOG': self.data_eog[eog_test_indices],
            'EMG': self.data_emg[emg_test_indices]
        }
        X_test, y_test = combine_data(clean_data=self.data_clean[clean_test_indices], noises=test_noises)
        self.save_data(X_test, y_test, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--lower_snr', type=float, default=-7)
    parser.add_argument('--higher_snr', type=float, default=4.5)
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--val_size', type=float, default=0.1)
    args = parser.parse_args()
    DataNoiseCombiner(args)