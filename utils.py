import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path
from random import seed
from typing import Tuple
import numpy as np
from numpy import array, genfromtxt, ndarray
from scipy.stats import zscore
from torch.utils.data import Dataset
import scipy.io as sio
import os

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif val_loss >= self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

def setup_logging(log_file, log_level):
    logging.basicConfig(filename=log_file, level=log_level, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    return acc, f1, precision, recall

def combine_waveforms(clean, noise, snr_db):
    rms = lambda x: np.sqrt(np.mean(x ** 2, axis=1))
    clean_EEG = clean[0]
    noise_EEG = noise[0]

    if snr_db is None:
        snr_db = np.random.choice(np.arange(-7, 4.5, 0.5), (noise_EEG.shape[0],))

    lambda_snr = rms(clean_EEG) / rms(noise_EEG) / 10 ** (snr_db / 20)
    lambda_snr = np.expand_dims(lambda_snr, 1)

    combined_data = clean_EEG + lambda_snr * noise_EEG
    labels = array([noise[1][0]] * len(noise_EEG))

    return (combined_data, labels)
