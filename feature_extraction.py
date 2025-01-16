import numpy as np
import pywt
from scipy.signal import welch, butter, filtfilt

# def wavelet_transform(signal, wavelet='db4', level=4):
#     coeffs = pywt.wavedec(signal, wavelet, level=level)
#     approx = coeffs[0]
#     details = coeffs[1:]
#     return np.concatenate([approx.flatten()])

def power_spectral_density(signal, fs=256):
    freqs, psd = welch(signal, fs=fs)
    return psd

def design_lowpass_filter(cutoff, fs=256, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(b, a, signal):
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def decimate_signal(signal, decimation_factor=16):
    return signal[::decimation_factor]

def wavelet_4th_level_approximation_simulation(signal, level=4):
    fs = 256
    cutoff = fs / (2 ** (level + 1))
    b, a = design_lowpass_filter(cutoff, fs=fs)
    filtered_signal = apply_lowpass_filter(b, a, signal)
    decimated_signal = decimate_signal(filtered_signal, decimation_factor=2**level)
    return decimated_signal.flatten()

def extract_features(eeg_signals):
    features = []
    for signal in eeg_signals:
        low_pass_info = wavelet_4th_level_approximation_simulation(signal)
        psd = power_spectral_density(signal)
        feature_vector = np.concatenate([low_pass_info, psd])
        features.append(feature_vector)
    return np.array(features)