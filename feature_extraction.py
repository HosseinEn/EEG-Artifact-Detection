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
    return np.concatenate([c.flatten() for c in coeffs]) # coeffs[0] is the approximation and the rest are details

def power_spectral_density(signal, fs=512):
    freqs, psd = welch(signal, fs=fs)
    return psd

def extract_features(eeg_signals):
    features = []
    for signal in eeg_signals:
        wavelet_features = wavelet_transform(signal)

        ica = FastICA(n_components=1, random_state=10)
        ica_component = ica.fit_transform(signal.reshape(-1, 1)).flatten()

        ica_var = np.var(ica_component)
        ica_skewness = skew(ica_component)
        ica_kurt = kurtosis(ica_component)
        ica_rms = np.sqrt(np.mean(ica_component**2))

        var = np.var(signal)
        skewness = skew(signal)
        kurt = kurtosis(signal)
        rms = np.sqrt(np.mean(signal**2))
        entropy = ant.spectral_entropy(signal, sf=512, method='welch')

        psd = power_spectral_density(signal)

        feature_vector = np.concatenate([wavelet_features, [var, skewness, kurt, rms, entropy, ica_var, ica_skewness, ica_kurt, ica_rms], psd])
        features.append(feature_vector)

    pca = PCA(n_components=0.95)
    pca_features = pca.fit_transform(features)
    return np.array(pca_features)