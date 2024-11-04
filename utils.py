import logging
import numpy as np

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

def create_noisy_data(clean_data, noises_dict, SNR_total):
    P_signal = np.mean(clean_data ** 2)
    P_noise_total = P_signal / (10 ** (SNR_total / 10))
    noise_weights = np.random.dirichlet([1] * len(noises_dict))
    P_noises = noise_weights * P_noise_total
    lambdas = np.zeros(len(noises_dict))
    for i, noise in enumerate(noises_dict.values()):
        P_noise_orig = np.mean(noise ** 2)
        lambdas[i] = np.sqrt(P_noises[i] / P_noise_orig)
    noisy_data = clean_data.copy()
    for i, noise in enumerate(noises_dict.values()):
        noisy_data += lambdas[i] * noise
    P_noisy = np.mean((noisy_data - clean_data) ** 2)
    SNR_total_actual = 10 * np.log10(P_signal / P_noisy)
    return noisy_data, SNR_total_actual

def combine_data(clean_data, noises):
    combined_noises, labels = [], []
    for cln_data, white_noise, eog, emg in zip(clean_data, noises['White_noise'], noises['EOG'], noises['EMG']):
        random_SNR = np.random.choice(np.arange(-7, 6.5, 0.5))
        n, l = create_noisy_data(cln_data, {'White_noise': white_noise, 'EOG': eog, 'EMG': emg}, random_SNR)
        combined_noises.append(n)
        labels.append(l)
    return combined_noises, labels



