from pathlib import Path
from random import seed
from typing import List, Tuple
import numpy as np
import torch
from numpy import array, genfromtxt, ndarray
from pytorch_lightning import LightningDataModule
from scipy.stats import zscore
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import argparse
import os

seed(2408)
np.random.seed(1305)


class EEGDenoiseDataset(Dataset):
    @staticmethod
    def load_samples(path, label) -> Tuple[ndarray, ndarray]:
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
        y = [label] * len(X)

        return X, array(y)

    @staticmethod
    def combine_waveforms(
        clean: Tuple[ndarray, ndarray], noise: Tuple[ndarray, ndarray], snr_db: float
    ) -> Tuple[ndarray, ndarray]:
        rms = lambda x: np.sqrt(np.mean(x ** 2, axis=1))

        clean_EEG = clean[0]
        noise_EEG = noise[0]

        rep = np.ceil(len(clean_EEG) / len(noise_EEG))
        noise_EEG = np.repeat(noise_EEG, rep, axis=0)[: len(clean_EEG), :]

        if snr_db is None:
            snr_db = np.random.choice(np.arange(-7, 4.5, 0.5), (noise_EEG.shape[0],))

        lambda_snr = rms(clean_EEG) / rms(noise_EEG) / 10 ** (snr_db / 20)
        lambda_snr = np.expand_dims(lambda_snr, 1)

        combined_data = zscore(clean_EEG + lambda_snr * noise_EEG, axis=1)
        labels = array([noise[1][0]] * len(noise_EEG))

        return (noise_EEG, (combined_data, labels))

    def __init__(
        self,
        file_path: Path = Path("data/"),
        split: str = "",
        snr_db: float = None,
    ) -> None:
        # Labels are served as follows:
        # - 0 clean sample
        # - 1 EOG artifact on clean samples
        # - 2 EMG artifact on clean samples
        data_clean = self.load_samples(os.path.join(file_path, "EEG_all_epochs.mat"), 0)
        data_eog = self.load_samples(os.path.join(file_path, "EOG_all_epochs.mat"), 1)
        data_emg = self.load_samples(os.path.join(file_path, "EMG_all_epochs.mat"), 2)

        noise_eog, data_eog = self.combine_waveforms(data_clean, data_eog, snr_db)
        noise_emg, data_emg = self.combine_waveforms(data_clean, data_emg, snr_db)
        data_clean = (zscore(data_clean[0], axis=1), data_clean[1])

        self.X = np.concatenate((data_eog[0], data_emg[0], data_clean[0]), axis=0)
        # data_eog is all zeros, data_emg is all ones, data_clean is all twos
        self.y = np.concatenate((data_eog[1], data_emg[1], data_clean[1]), axis=0)
        self.clean_samples = np.concatenate(
            (data_clean[0], data_clean[0], data_clean[0]), axis=0
        )
        self.noise_samples = np.concatenate(
            (noise_eog, noise_emg, np.zeros(shape=data_clean[0].shape)), axis=0
        )

    def __getitem__(self, index: int) -> Tuple[ndarray, ndarray]:
        return (
            self.clean_samples[index],
            self.noise_samples[index],
            self.X[index],
            self.y[index],
        )

    def __len__(self):
        return len(self.X)


class EEGDenoiseDM(LightningDataModule):
    def __init__(
        self,
        config
    ) -> None:
        super().__init__()
        data = EEGDenoiseDataset(config.datapath, snr_db=config.snr_db)
        test_size = config.test_size
        val_size = config.val_size
        train, self.test = torch.utils.data.random_split(
            data,
            [int(np.floor(len(data) * (1 - test_size))), int(np.ceil(len(data) * test_size))],
            generator=torch.Generator().manual_seed(1305),
        )

        self.train, self.val = torch.utils.data.random_split(
            train,
            [int(np.floor(len(train) * (1 - val_size))), int(np.ceil(len(train) * val_size))],
            generator=torch.Generator().manual_seed(2408),
        )
        print(f"Train size: {len(self.train)}", f"Val size: {len(self.val)}", f"Test size: {len(self.test)}")



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--snr_db", type=float, default=None)
    args = args.parse_args()
    a = EEGDenoiseDM(snr_db=args.snr_db)

    clean, noise, x, y = [], [], [], []
    for s in a.val:
        clean.append(s[0])
        noise.append(s[1])
        x.append(s[2])
        y.append(s[3])

    clean = array(clean)
    noise = array(noise)
    y = array(y)

    from scipy.io import savemat

    savemat("val_set.mat", {"clean": clean, "noise": noise, "x": x, "y": y})
