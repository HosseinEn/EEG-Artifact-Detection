import numpy as np
import pywt
from scipy.signal import welch

def wavelet_transform(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    approx = coeffs[0]
    details = coeffs[1:]
    return np.concatenate([approx.flatten()])

def power_spectral_density(signal, fs=512):
    freqs, psd = welch(signal, fs=fs)
    return psd

def extract_features(eeg_signals):
    features = []
    for signal in eeg_signals:
        wavelet_features = wavelet_transform(signal)
        psd = power_spectral_density(signal)
        feature_vector = np.concatenate([wavelet_features, psd])
        features.append(feature_vector)
    return np.array(features)