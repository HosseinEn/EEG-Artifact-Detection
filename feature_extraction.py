import numpy as np
import pywt
from sklearn.decomposition import FastICA
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import antropy as ant

def wavelet_transform(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.concatenate([c.flatten() for c in coeffs])

def power_spectral_density(signal, fs=256):
    freqs, psd = welch(signal, fs=fs)
    return psd

def extract_features(eeg_signals):
    features = []
    for signal in eeg_signals:
        # Wavelet transform
        wavelet_features = wavelet_transform(signal)

        # ICA
        ica = FastICA(n_components=1, random_state=10)
        ica_features = ica.fit_transform(signal.reshape(-1, 1)).flatten()

        # Statistical features
        var = np.var(signal)
        skewness = skew(signal)
        kurt = kurtosis(signal)
        rms = np.sqrt(np.mean(signal**2))
        entropy = ant.spectral_entropy(signal, sf=256, method='welch')

        # Power spectral density
        psd = power_spectral_density(signal)

        # LOF
        lof = LocalOutlierFactor(n_neighbors=20)
        lof_score = lof.fit_predict(signal.reshape(-1, 1))

        # Combine features
        # feature_vector = np.concatenate([ica_features, [var, skewness, kurt, rms, entropy], psd])
        feature_vector = np.concatenate([ica_features, [], psd])
        features.append(feature_vector)

    return np.array(features)
