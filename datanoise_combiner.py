from pathlib import Path
from random import seed
from typing import Tuple
import numpy as np
from numpy import array, genfromtxt, ndarray
from scipy.stats import zscore
from torch.utils.data import Dataset
import scipy.io as sio
import os
from utils import combine_waveforms

seed(2408)
np.random.seed(1305)

class DataNoiseCombiner:
    def load_samples(self, path, label=None) -> Tuple[ndarray, ndarray]:
        X = None
        path = Path(path)
        if path.suffix == ".mat":
            X = sio.loadmat(path)
            for key, value in X.items():
                if isinstance(value, np.ndarray):
                    X = value
                    break
        elif path.suffix == ".csv":
            X = genfromtxt(path, delimiter=",")
        elif path.suffix == ".npy":
            X = np.load(path)
        y = [label] * len(X) if label is not None else None

        return X, array(y) if y is not None else None


    def __init__(self, file_path: Path = Path("data/"), test_ratio = None, config = None):
        data_clean = self.load_samples(os.path.join(file_path, "EEG_all_epochs.mat"), 0)
        data_eog = self.load_samples(os.path.join(file_path, "EOG_all_epochs.mat"), 1)
        data_emg = self.load_samples(os.path.join(file_path, "EMG_all_epochs.mat"), 2)

        clean_indices = np.arange(len(data_clean[0]))
        np.random.shuffle(clean_indices)

        eog_indices = np.arange(len(data_eog[0]))
        np.random.shuffle(eog_indices)

        emg_indices = np.arange(len(data_emg[0]))
        np.random.shuffle(emg_indices)


        clean_test_size = int(test_ratio * len(clean_indices))
        clean_test_indices = clean_indices[:clean_test_size]
        remaining_clean_indices = clean_indices[clean_test_size:]

        eog_test_size = int(test_ratio * len(eog_indices))
        eog_test_indices = eog_indices[:eog_test_size]
        remaining_eog_indices = eog_indices[eog_test_size:]

        emg_test_size = int(test_ratio * len(emg_indices))
        emg_test_indices = emg_indices[:emg_test_size]
        remaining_emg_indices = emg_indices[emg_test_size:]

        test_dir = Path(file_path) / "test"
        test_dir.mkdir(exist_ok=True)
        for snr in np.arange(config.lower_snr, config.higher_snr, 0.5):
            combined_eog = combine_waveforms((data_clean[0][clean_test_indices], data_clean[0][clean_test_indices]),
                                             (data_eog[0][eog_test_indices],data_eog[1][eog_test_indices]), snr_db=snr)
            combined_emg = combine_waveforms((data_clean[0][clean_test_indices], data_clean[0][clean_test_indices]),
                                             (data_emg[0][emg_test_indices],data_emg[1][emg_test_indices]), snr_db=snr)
            combined_clean = (data_clean[0][clean_test_indices], data_clean[1][clean_test_indices])


            X = np.concatenate((combined_eog[0], combined_emg[0], combined_clean[0]), axis=0)
            y = np.concatenate((combined_eog[1], combined_emg[1], combined_clean[1]), axis=0)

            snr_dir = Path(file_path) / "test" / f"snr {snr}"
            snr_dir.mkdir(exist_ok=True)
            np.save(os.path.join(file_path, "test", f"snr {snr}", f"X.npy"), X)
            np.save(os.path.join(file_path, "test", f"snr {snr}", f"Y.npy"), y)

        combined_eog = combine_waveforms((data_clean[0][remaining_clean_indices], data_clean[0][remaining_clean_indices]),
                                         (data_eog[0][remaining_eog_indices],data_eog[1][remaining_eog_indices]), snr_db=None)
        combined_emg = combine_waveforms((data_clean[0][remaining_clean_indices], data_clean[0][remaining_clean_indices]),
                                         (data_emg[0][remaining_emg_indices],data_emg[1][remaining_emg_indices]), snr_db=None)
        combined_clean = (data_clean[0][remaining_clean_indices], data_clean[1][remaining_clean_indices])

        X = np.concatenate((combined_eog[0], combined_emg[0], combined_clean[0]), axis=0)
        y = np.concatenate((combined_eog[1], combined_emg[1], combined_clean[1]), axis=0)

        train_dir = Path(file_path) / "train_val"
        train_dir.mkdir(exist_ok=True)
        np.save(os.path.join(file_path, "train_val", "X.npy"), X)
        np.save(os.path.join(file_path, "train_val", "Y.npy"), y)


if __name__ == "__main__":
    DataNoiseCombiner()
