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
        self.data_clean = self.load_samples(os.path.join(config.datapath, "filtered80Hz_EEG_all_epochs.mat"), 0)
        self.data_eog = self.load_samples(os.path.join(config.datapath, "filtered80Hz_EOG_all_epochs.mat"), 1)
        self.data_emg = self.load_samples(os.path.join(config.datapath, "filtered80Hz_EMG_all_epochs.mat"), 2)
        self.white_noise = self.load_samples(os.path.join(config.datapath, "white_noise.mat"), 3)
        self.clean_indices = self.shuffle_indices(len(self.data_clean[0]))
        self.eog_indices = self.shuffle_indices(len(self.data_eog[0]))
        self.emg_indices = self.shuffle_indices(len(self.data_emg[0]))
        self.white_noise_indices = self.shuffle_indices(len(self.white_noise[0]))
        self.process_and_save_data()

    def load_samples(self, path, label=None):
        path = Path(path)
        data = sio.loadmat(path) if path.suffix == ".mat" else (
            genfromtxt(path, delimiter=",") if path.suffix == ".csv" else (
                np.load(path) if path.suffix == ".npy" else None))
        if data is None:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        X = next(value for value in data.values() if isinstance(value, np.ndarray)) if path.suffix == ".mat" else data
        X = np.repeat(X, np.ceil(len(self.data_clean[0]) / len(X)), axis=0)[: len(self.data_clean[0]), :] if label in [1, 2] else X
        return X, array([label] * len(X)) if label is not None else None

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

    def combine_and_save(self, clean_indices, noise_indices, data_clean, data_noise, snr):
        combined_data = combine_waveforms((data_clean[0][clean_indices], data_clean[0][clean_indices]),
                                          (data_noise[0][noise_indices], data_noise[1][noise_indices]), snr_db=snr)
        return combined_data[0], combined_data[1]

    def process_and_save_data(self):
        clean_test_indices, clean_val_indices, clean_training_indices = self.split_indices(self.clean_indices, self.config.test_size, self.config.val_size)
        eog_test_indices, eog_val_indices, eog_training_indices = self.split_indices(self.eog_indices, self.config.test_size, self.config.val_size)
        emg_test_indices, emg_val_indices, emg_training_indices = self.split_indices(self.emg_indices, self.config.test_size, self.config.val_size)
        wn_test_indices, wn_val_indices, wn_training_indices = self.split_indices(self.white_noise_indices, self.config.test_size, self.config.val_size)

        for snr in np.arange(self.config.lower_snr, self.config.higher_snr, 0.5):
            X_eog, y_eog = self.combine_and_save(clean_test_indices, eog_test_indices, self.data_clean, self.data_eog, snr)
            X_emg, y_emg = self.combine_and_save(clean_test_indices, emg_test_indices, self.data_clean, self.data_emg, snr)
            X_wn, y_wn = self.combine_and_save(clean_test_indices, wn_test_indices, self.data_clean, self.white_noise, snr)
            X_clean, y_clean = self.data_clean[0][clean_test_indices], self.data_clean[1][clean_test_indices]
            X = np.concatenate((X_eog, X_emg, X_wn, X_clean), axis=0)
            y = np.concatenate((y_eog, y_emg, y_wn, y_clean), axis=0)
            self.save_data(X, y, "test", f"snr {snr}")

        subsets = [("val", clean_val_indices, wn_val_indices, eog_val_indices, emg_val_indices),
         ("train", clean_training_indices, wn_training_indices, eog_training_indices, emg_training_indices)]
        for subset_name, clean_indices, wn_indices, eog_indices, emg_indices in subsets:
            X_eog, y_eog = self.combine_and_save(clean_indices, eog_indices, self.data_clean, self.data_eog, None)
            X_emg, y_emg = self.combine_and_save(clean_indices, emg_indices, self.data_clean, self.data_emg, None)
            X_wn, y_wn = self.combine_and_save(clean_indices, wn_indices, self.data_clean, self.white_noise, None)
            X_clean, y_clean = self.data_clean[0][clean_indices], self.data_clean[1][clean_indices]
            self.save_data(np.concatenate((X_eog, X_emg, X_clean, X_wn), axis=0),
                           np.concatenate((y_eog, y_emg, y_clean, y_wn), axis=0), subset_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--lower_snr', type=float, default=-7)
    parser.add_argument('--higher_snr', type=float, default=4.5)
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--val_size', type=float, default=0.2)
    args = parser.parse_args()
    DataNoiseCombiner(args)