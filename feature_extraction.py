import numpy as np
import pywt
from scipy.signal import welch

def wavelet_packets(signal, wavelet='db4', level=4):
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric')
    nodes = [node.path for node in wp.get_level(level, 'freq')]
    wavelet_features = []
    for node in nodes:
        wavelet_features.append(wp[node].data)
    return wavelet_features

def wavelet_transform(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    approx = coeffs[0]
    return approx

def wavelet_energy(coeffs):
    energies = []
    for coeff in coeffs:
        energies.append(np.mean(np.square(coeff)))
    return energies

def power_spectral_density(signal, fs=256):
    freqs, psd = welch(signal, fs=fs)
    return psd

def extract_features(eeg_signals):
    features = []
    for signal in eeg_signals:
        pkts = wavelet_packets(signal)
        pkts_energy = wavelet_energy(pkts)
        psd = power_spectral_density(signal)
        feature_vector = np.concatenate([pkts_energy, psd])
        features.append(feature_vector)
    return np.array(features)