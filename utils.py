import logging
import numpy as np
from scipy.stats import zscore

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
                        format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')


def combine_data(clean_data, noises):
    P_signal = np.mean(clean_data ** 2, axis=1)
    num_samples = clean_data.shape[0]
    combined_data = clean_data.copy()
    total_noise = np.zeros_like(clean_data)
    l = []

    for name, noise in noises.items():
        snr_db = np.random.choice(np.arange(-7, 6, 0.5), (num_samples,))
        l.append(snr_db)
        P_noise = np.mean(noise ** 2, axis=1)
        lambda_n = np.sqrt(P_signal / (10 ** (snr_db / 10)) / P_noise)
        lambda_n = lambda_n[:, np.newaxis]
        scaled_noise = noise * lambda_n
        total_noise += scaled_noise
    l = np.array(l).T
    combined_data += total_noise
    return combined_data, l
