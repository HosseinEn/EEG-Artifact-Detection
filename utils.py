import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from numpy import array
from scipy.signal import firwin, filtfilt


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
    return combined_data, labels


def combine_noise_simultaneously(clean, noises, snrs_db, config):
    rms = lambda x: np.sqrt(np.mean(x ** 2, axis=1))
    clean_EEG = clean[0]
    num_samples = clean_EEG.shape[0]
    combined_data = clean_EEG.copy()
    total_noise = np.zeros_like(clean_EEG)
    if snrs_db is None:
        for noise in noises:
            snr_db = np.random.choice(np.arange(config.lower_snr, config.higher_snr), (num_samples,))
            noise_EEG = noise[0]
            rms_clean = rms(clean_EEG)
            rms_noise = rms(noise_EEG)
            lambda_n = (rms_clean / rms_noise) / (10 ** (snr_db / 20))
            lambda_n = lambda_n[:, np.newaxis]
            scaled_noise = noise_EEG * lambda_n
            total_noise += scaled_noise
    else:
        for noise, snr in zip(noises, snrs_db):
            noise_EEG = noise[0]
            rms_clean = rms(clean_EEG)
            rms_noise = rms(noise_EEG)
            lambda_n = (rms_clean / rms_noise) / (10 ** (snr / 20))
            lambda_n = lambda_n[:, np.newaxis]
            scaled_noise = noise_EEG * lambda_n
            total_noise += scaled_noise
    combined_data += total_noise
    labels = np.full((num_samples,), noises[np.argmin(snrs_db)][1][0])
    return combined_data, labels

def custom_bandpass_filter(data, lowcut, highcut, fs,
                           l_trans_bandwidth=0.5,
                           h_trans_bandwidth=0.5,
                           filter_length=101,
                           fir_window='hann',
                           pad_length=100):
    nyquist = 0.5 * fs
    low = (lowcut - l_trans_bandwidth) / nyquist
    high = (highcut + h_trans_bandwidth) / nyquist
    if filter_length % 2 == 0:
        filter_length += 1
    fir_coeff = firwin(filter_length, [low, high], pass_zero=False, window=fir_window)
    padded_data = np.pad(data, (pad_length, pad_length), mode='edge')
    filtered_data = filtfilt(fir_coeff, 1.0, padded_data)
    return filtered_data[pad_length:pad_length + len(data)]

def add_white_noise(data, snr_db):
    d = data.copy()
    l = data[1]
    signal_power = np.mean(d ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), d.shape)
    filtered_noise = np.zeros_like(noise)
    for i in range(d.shape[0]):
        filtered_noise[i] = custom_bandpass_filter(noise[i], 1, 80, 256)
    d = d + filtered_noise
    return d, l

def create_white_noise_data(data):
    SNRs = np.random.choice(np.arange(-7, 6.5, 0.5), (data.shape[0],))
    noise_powers = np.mean(data ** 2) / (10 ** (SNRs / 10))
    noises = np.zeros(data.shape)
    for i in range(data.shape[0]):
        noise = np.random.normal(0, np.sqrt(noise_powers[i]), data.shape[1])
        noises[i] = custom_bandpass_filter(noise, 1, 80, 256)
    d = data + noises
    return d, 3 * np.ones(data.shape[0])
