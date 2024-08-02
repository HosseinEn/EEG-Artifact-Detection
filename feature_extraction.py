import numpy as np
import pywt
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import antropy as ant

def wavelet_transform(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    approx = coeffs[0]
    details = coeffs[1:]
    return np.concatenate([approx.flatten()] + [d.flatten() for d in details[1:]])

def power_spectral_density(signal, fs=512):
    freqs, psd = welch(signal, fs=fs)
    return psd

def extract_features(eeg_signals):
    features = []
    for signal in eeg_signals:
        # wavelet_features = wavelet_transform(signal)
        #
        #
        var = np.var(signal)
        skewness = skew(signal)
        kurt = kurtosis(signal)
        rms = np.sqrt(np.mean(signal**2))
        entropy = ant.spectral_entropy(signal, sf=512, method='welch')
        #
        psd = power_spectral_density(signal)

        feature_vector = np.concatenate([signal, [var, skewness, kurt, rms, entropy], psd])
        features.append(feature_vector)


    return np.array(features)