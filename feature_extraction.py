import numpy as np
import pywt
from sklearn.decomposition import FastICA
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import antropy as ant
from sklearn.decomposition import PCA

def wavelet_transform(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.concatenate([c.flatten() for c in coeffs])

def power_spectral_density(signal, fs=256):
    freqs, psd = welch(signal, fs=fs)
    return psd

def extract_features(eeg_signals):
    features = []
    for signal in eeg_signals:
        wavelet_features = wavelet_transform(signal)

        ica = FastICA(n_components=1, random_state=10)
        ica_features = ica.fit_transform(signal.reshape(-1, 1)).flatten()

        var = np.var(signal)
        skewness = skew(signal)
        kurt = kurtosis(signal)
        rms = np.sqrt(np.mean(signal**2))
        entropy = ant.spectral_entropy(signal, sf=256, method='welch')

        psd = power_spectral_density(signal)

        # LOF
        lof = LocalOutlierFactor(n_neighbors=20)
        lof_score = lof.fit_predict(signal.reshape(-1, 1))

        # Combine features
        feature_vector = np.concatenate([ica_features, [var, skewness, kurt, rms, entropy], psd]) # Acc: 92.57, F1: 92.59, Prec: 92.94
        # feature_vector = np.concatenate([ica_features, [var, rms, entropy], psd]) # Acc: 92.59, F1: 92.60, Prec: 92.59
        # feature_vector = np.concatenate([ica_features, [rms, entropy], psd]) # Acc: 92.61, F1: 92.63, Prec: 93.09
        # feature_vector = np.concatenate([ica_features, [], psd]) # Acc: 91.15, F1: 91.14, Prec: 91.35

        features.append(feature_vector)

    pca = PCA(n_components=0.95)
    pca_features = pca.fit_transform(features)

    return np.array(pca_features)
