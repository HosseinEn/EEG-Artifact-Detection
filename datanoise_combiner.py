from pathlib import Path
from random import seed
from typing import Tuple
import numpy as np
from numpy import array, genfromtxt, ndarray
from scipy.stats import zscore
from torch.utils.data import Dataset
import scipy.io as sio
import os
import argparse
from utils import combine_waveforms

seed(2408)
np.random.seed(1305)

class DataNoiseCombiner:
    def __init__(self, config):
        self.config = config
        self.data_clean = self.load_samples(os.path.join(config.datapath, "EEG_all_epochs.mat"), 0)
        self.data_eog = self.load_samples(os.path.join(config.datapath, "EOG_all_epochs.mat"), 1)
        self.data_emg = self.load_samples(os.path.join(config.datapath, "EMG_all_epochs.mat"), 2)

        self.clean_indices = self.shuffle_indices(len(self.data_clean[0]))
        self.eog_indices = self.shuffle_indices(len(self.data_eog[0]))
        self.emg_indices = self.shuffle_indices(len(self.data_emg[0]))

        self.process_and_save_data()

    def load_samples(self, path, label=None):
        path = Path(path)
        if path.suffix == ".mat":
            data = sio.loadmat(path)
            X = next(value for value in data.values() if isinstance(value, np.ndarray))
        elif path.suffix == ".csv":
            X = genfromtxt(path, delimiter=",")
        elif path.suffix == ".npy":
            X = np.load(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        # resize eog and emg to match the size of the clean data
        if label == 1 or label == 2:
            rep = np.ceil(len(self.data_clean[0]) / len(X))
            X = np.repeat(X, rep, axis=0)[: len(self.data_clean[0]), :]

        y = array([label] * len(X)) if label is not None else None
        return X, y

    @staticmethod
    def shuffle_indices(length):
        indices = np.arange(length)
        # np.random.shuffle(indices)
        return indices

    def split_indices(self, indices, test_size, val_size):
        test_size = int(test_size * len(indices))
        val_size = int(val_size * len(indices))
        test_indices = indices[:test_size]
        val_indices = indices[test_size:test_size + val_size]
        training_indices = indices[test_size + val_size:]
        return test_indices, val_indices, training_indices

    def save_data(self, X, y, data_type, snr_type=None):
        if snr_type is not None:
            directory = Path(self.config.datapath) / data_type / snr_type
        else:
            directory = Path(self.config.datapath) / data_type
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "X.npy", X)
        np.save(directory / "Y.npy", y)

    def combine_and_save(self, clean_indices, noise_indices, data_clean, data_noise, snr, data_type):
        combined_data = combine_waveforms(
            (data_clean[0][clean_indices], data_clean[0][clean_indices]),
            (data_noise[0][noise_indices], data_noise[1][noise_indices]), snr_db=snr
        )
        X, y = combined_data[0], combined_data[1]
        return X, y

    def process_and_save_data(self):
        clean_test_indices, clean_val_indices, clean_training_indices = self.split_indices(
            self.clean_indices, self.config.test_size, self.config.val_size)
        eog_test_indices, eog_val_indices, eog_training_indices = self.split_indices(
            self.eog_indices, self.config.test_size, self.config.val_size)
        emg_test_indices, emg_val_indices, emg_training_indices = self.split_indices(
            self.emg_indices, self.config.test_size, self.config.val_size)

        for snr in np.arange(self.config.lower_snr, self.config.higher_snr, 0.5):
            X_eog, y_eog = self.combine_and_save(clean_test_indices, eog_test_indices, self.data_clean, self.data_eog, snr, "test")
            X_emg, y_emg = self.combine_and_save(clean_test_indices, emg_test_indices, self.data_clean, self.data_emg, snr, "test")
            X_clean, y_clean = self.data_clean[0][clean_test_indices], self.data_clean[1][clean_test_indices]

            X = np.concatenate((X_eog, X_emg, X_clean), axis=0)
            y = np.concatenate((y_eog, y_emg, y_clean), axis=0)
            self.save_data(X, y, "test", f"snr {snr}")

        X_eog, y_eog = self.combine_and_save(clean_val_indices, eog_val_indices, self.data_clean, self.data_eog, None, "val")
        X_emg, y_emg = self.combine_and_save(clean_val_indices, emg_val_indices, self.data_clean, self.data_emg, None, "val")
        X_clean, y_clean = self.data_clean[0][clean_val_indices], self.data_clean[1][clean_val_indices]

        X = np.concatenate((X_eog, X_emg, X_clean), axis=0)
        y = np.concatenate((y_eog, y_emg, y_clean), axis=0)
        self.save_data(X, y, "val")

        X_eog, y_eog = self.combine_and_save(clean_training_indices, eog_training_indices, self.data_clean, self.data_eog, None, "train")
        X_emg, y_emg = self.combine_and_save(clean_training_indices, emg_training_indices, self.data_clean, self.data_emg, None, "train")
        X_clean, y_clean = self.data_clean[0][clean_training_indices], self.data_clean[1][clean_training_indices]

        X = np.concatenate((X_eog, X_emg, X_clean), axis=0)
        y = np.concatenate((y_eog, y_emg, y_clean), axis=0)
        self.save_data(X, y, "train")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--lower_snr', type=float, default=-7)
    parser.add_argument('--higher_snr', type=float, default=4.5)
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--val_size', type=float, default=0.2)
    args = parser.parse_args()
    combiner = DataNoiseCombiner(args)


