import numpy as np
import pywt
from sklearn.decomposition import FastICA
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import skew, kurtosis


def wavelet_transform(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.concatenate([c.flatten() for c in coeffs])


def extract_features(eeg_signals):
    features = []
    for signal in eeg_signals:
        # Wavelet transform
        wavelet_features = wavelet_transform(signal)

        # ICA
        ica = FastICA(n_components=1)
        ica_features = ica.fit_transform(signal.reshape(-1, 1)).flatten()

        # Statistical features
        var = np.var(signal)
        skewness = skew(signal)
        kurt = kurtosis(signal)

        # LOF
        lof = LocalOutlierFactor(n_neighbors=20)
        lof_score = lof.fit_predict(signal.reshape(-1, 1))

        # Combine features
        feature_vector = np.concatenate([ica_features])
        features.append(feature_vector)

    return np.array(features)
